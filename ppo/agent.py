# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from scipy.special import softmax as softmax_sci
from torch.nn.utils import clip_grad_norm_
import GLOBAL_PRARM as gp

from ppo.basic_block import DQN
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
# https://openai.com/blog/openai-baselines-ppo/
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/distributions.py

class Agent:
    def __init__(self, args, env, index):
        self.action_space = env.get_action_size()
        self.atoms = args.atoms
        self.action_type = args.action_selection
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.device = args.device
        self.net_type = args.architecture
        self.reward_update_rate = args.reward_update_rate
        self.average_reward = 0
        self.neighbor_indice = np.zeros([])
        self.clip_param = 0.2
        # https://openai.com/blog/openai-baselines-ppo/

        self.online_net = DQN(args, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
            self.model_path = os.path.join(args.model, "model" + str(index) + ".pth")
            if os.path.isfile(self.model_path):
                state_dict = torch.load(self.model_path, map_location='cpu')
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                # if 'conv1.weight' in state_dict.keys():
                #     for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'),
                #                              ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'),
                #                              ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                #         state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
                #         del state_dict[old_key]  # Delete old keys for strict load_state_dict
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + self.model_path)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(self.model_path)

        self.online_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)

        self.online_dict = self.online_net.state_dict()
        self.target_dict = self.target_net.state_dict()

        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    def update_neighbor_indice(self, neighbor_indices):
        self.neighbor_indice = neighbor_indices

    def reload_step_state_dict(self, better=True):
        if better:
            self.online_dict = self.online_net.state_dict()
            self.target_dict = self.target_net.state_dict()
        else:
            self.online_net.load_state_dict(self.online_dict)
            self.target_net.load_state_dict(self.target_dict)

    def get_state_dict(self):
        return self.online_net.state_dict()

    def set_state_dict(self, new_state_dict):
        self.online_net.load_state_dict(new_state_dict)
        return

    def get_target_dict(self):
        return self.target_net.state_dict()

    def set_target_dict(self, new_state_dict):
        self.target_net.load_state_dict(new_state_dict)
        return

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, available=None, epsilon=0.3, action_type='greedy'):  # High ε can reduce evaluation scores drastically
        if action_type == 'greedy' or action_type == 'no_limit':
            raise ValueError("Greedy action selection is banned in PPO")
        elif action_type == 'boltzmann':
            return self.act_boltzmann(state, available)

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_boltzmann(self, state, avail):  # High ε can reduce evaluation scores drastically
        with torch.no_grad():
            res_policy, res_policy_log = self.online_net(state.unsqueeze(0))
            res_action = self.boltzmann(res_policy, [avail])
            return (res_action, (res_policy[0]).numpy())

    def boltzmann(self, res_policy, mask):
        sizeofres = res_policy.shape
        res = []
        res_policy = res_policy.numpy()
        for i in range(sizeofres[0]):
            action_probs = [res_policy[i][ind] * mask[i][ind] for ind in range(res_policy[i].shape[0])]
            count = np.sum(action_probs)
            if count == 0:
                action_probs = np.array([1.0 / np.sum(mask[i]) if _ != 0 else 0 for _ in mask[i]])
                print('Zero probs, random action')
            else:
                action_probs = np.array([x / count for x in action_probs])
            res.append(np.random.choice(self.action_space, p=action_probs))
        if sizeofres[0] == 1:
            return res[0]
        return np.array(res)

    def lookup_server(self, list_of_pipe):
        num_pro = len(list_of_pipe)
        list_pro = np.ones(num_pro, dtype=bool)
        with torch.no_grad():
            while list_pro.any():
                for key, pipes in enumerate(list_of_pipe):
                    if not pipes.closed and pipes.readable:
                        obs, avial = pipes.recv()
                        if len(obs) == 1:
                            if not obs[0]:
                                pipes.close()
                                list_pro[key] = False
                                continue
                        pipes.send(self.act_boltzmann(obs, avial))
                        # convert back to numpy or cpu-tensor, or it will cause error since cuda try to run in
                        # another thread. Keep the gpu resource inside main thread

    def lookup_server_loop(self, list_of_pipe):
        num_pro = len(list_of_pipe)
        list_pro = np.ones(num_pro, dtype=bool)
        for key, pipes in enumerate(list_of_pipe):
            if not pipes.closed and pipes.readable:
                if pipes.poll():
                    obs, avial = pipes.recv()
                    if type(obs) is np.ndarray:
                        pipes.close()
                        list_pro[key] = False
                        continue
                    pipes.send(self.act_boltzmann(obs, avial))
            else:
                list_pro[key] = False
            # convert back to numpy or cpu-tensor, or it will cause error since cuda try to run in
            # another thread. Keep the gpu resource inside main thread
        return list_pro.any()

    @staticmethod
    def _to_one_hot(y, num_classes):
        y = torch.as_tensor(y)
        y[y == -1] = num_classes
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes + 1, dtype=torch.float32)
        zeros = zeros.scatter(scatter_dim, y_tensor, 1)
        return zeros[..., 0:num_classes]

    def learn(self, mem):
        # Sample transitions
        if gp.ONE_EPISODE_RUN > 0:
            self.average_reward = 0
        idxs, states, actions, actions_p, _, _, avails, returns, next_states, nonterminals, weights = \
            mem.sample(self.batch_size, self.average_reward)

        # critic update
        self.optimiser.zero_grad()

        # Calculate current state probabilities (online network noise already sampled)
        ps_a, log_ps_a = self.online_net(states, False, log=True)
        # Log probabilities log p(s_t, ·; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            self.target_net.reset_noise()  # Sample new target net noise
            pns_a, _ = self.target_net(next_states, False)
            # Probabilities p(s_t+n, ·; θtarget)
            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().long(), b.ceil().long()
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
            # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))
            # m_u = m_u + p(s_t+n, a*)(b - l)

            # update the average reward
            self.average_reward = self.average_reward + \
                                  self.reward_update_rate * torch.mean(returns.unsqueeze(1) +
                                                                       torch.sum(pns_a * self.support, dim=1) -
                                                                       torch.sum(ps_a.detach() * self.support, dim=1))

        value_loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))

        # Actor update
        actions_p_s = actions_p.gather(-1, actions.unsqueeze(1))
        curr_pol_out, curr_pol_out_log = self.online_net(states)
        curr_pol_out_p = curr_pol_out.gather(-1, actions.unsqueeze(1))
        ratios = torch.div(curr_pol_out_p, actions_p_s + 1e-7)
        # in case of the situation of zero probability caused Nan
        # since ratio is prob/oldprob, so zero probability should be zero or one

        state_value_current, _ = self.online_net(states, False)
        advantage = returns - torch.sum(state_value_current.detach() * self.support, dim=-1)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-7)
        surr1 = ratios * advantage
        surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

        dist = torch.distributions.Categorical(curr_pol_out)
        entropy_loss = dist.entropy().mean()

        policy_loss = - torch.min(surr1, surr2).mean() - entropy_loss
        # log probs * advantage
        policy_loss += -(curr_pol_out ** 2).mean() * 1e-3

        loss = (weights * value_loss).mean() + policy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 0.5)
        self.optimiser.step()

        mem.update_priorities(idxs[0], value_loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def soft_update_target_net(self, tau):
        for target_param, param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, index=-1, name='model.pth'):
        if index == -1:
            torch.save(self.online_net.state_dict(), os.path.join(path, name))
        else:
            torch.save(self.online_net.state_dict(), os.path.join(path, name[0:-4] + str(index) + name[-4:]))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            temp, _ = self.online_net(state.unsqueeze(0), False)
            return (temp * self.support).sum(-1).item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
