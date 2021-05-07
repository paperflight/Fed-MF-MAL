# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from scipy.special import softmax as softmax_sci
from torch.nn.utils import clip_grad_norm_
import GLOBAL_PRARM as gp
import torch.nn.functional as F

from trpo.basic_block import DQN
# https://github.com/mjacar/pytorch-trpo/blob/master/trpo_agent.py
# https://arxiv.org/pdf/1502.05477.pdf
# https://github.com/TianhongDai/reinforcement-learning-algorithms/blob/master/rl_algorithms/trpo/trpo_agent.py


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
        self.damping = 0.1  # the damping coeffificent
        self.max_kl = 0.01  # the max kl divergence

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
            return (res_action, (res_policy[:, res_action]).numpy())

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

    # conjugated gradient
    def _conjugated_gradient(self, b, update_steps, obs, pi_old, residual_tol=1e-10):
        # the initial solution is zero
        x = torch.zeros(b.size(), dtype=torch.float32)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(update_steps):
            fv_product = self._fisher_vector_product(p, obs, pi_old)
            alpha = rdotr / torch.dot(p, fv_product)
            x = x + alpha * p
            r = r - alpha * fv_product
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            # if less than residual tot.. break
            if rdotr < residual_tol:
                break
        return x

    # line search
    def _line_search(self, x, full_step, expected_rate, obs, adv, actions, pi_old, max_backtracks=10, accept_ratio=0.1):
        fval = self._get_surrogate_loss(obs, adv, actions, pi_old).data
        for (_n_backtracks, stepfrac) in enumerate(0.5 ** np.arange(max_backtracks)):
            xnew = x + stepfrac * full_step
            self._set_flat_params_to(xnew)
            new_fval = self._get_surrogate_loss(obs, adv, actions, pi_old).data
            actual_improve = fval - new_fval
            expected_improve = expected_rate * stepfrac
            ratio = actual_improve / expected_improve
            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                return True, xnew
        return False, x

    def _set_flat_params_to(self, flat_params):
        prev_indx = 0
        for param in self.online_net.actor_parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_indx:prev_indx + flat_size].view(param.size()))
            prev_indx += flat_size

    # get the surrogate loss
    def _get_surrogate_loss(self, obs, adv, actions, pi_old):
        p, logp = self.online_net(obs)
        p = p.gather(-1, actions.unsqueeze(1))
        pi_old_s = pi_old.gather(-1, actions.unsqueeze(1))
        ratios = torch.div(p, pi_old_s + 1e-7)
        # mask the zero probility to zero
        surr_loss = - ratios * adv
        return surr_loss.mean()

    # the product of the fisher informaiton matrix and the nature gradient -> Ax
    def _fisher_vector_product(self, v, obs, pi_old):
        kl = self._get_kl(obs, pi_old)
        kl = kl.mean()
        # start to calculate the second order gradient of the KL
        kl_grads = torch.autograd.grad(kl, self.online_net.actor_parameters(), create_graph=True)
        flat_kl_grads = torch.cat([grad.view(-1) for grad in kl_grads])
        kl_v = (flat_kl_grads * torch.autograd.Variable(v)).sum()
        kl_second_grads = torch.autograd.grad(kl_v, self.online_net.actor_parameters())
        flat_kl_second_grads = torch.cat([grad.contiguous().view(-1) for grad in kl_second_grads]).data
        flat_kl_second_grads = flat_kl_second_grads + self.damping * v
        return flat_kl_second_grads

    # get the kl divergence between two distributions
    def _get_kl(self, obs, pi_old):
        p, logp = self.online_net(obs)
        kl = F.kl_div(pi_old, p, reduction='sum')
        # kl = torch.exp(pi_old) * (pi_old - logp)
        return kl

    def learn(self, mem):
        # Sample transitions
        if gp.ONE_EPISODE_RUN > 0:
            self.average_reward = 0
        idxs, states, actions, actions_logp, _, _, avails, returns, next_states, nonterminals, weights = \
            mem.sample(self.batch_size, self.average_reward)

        self.online_net.zero_grad()
        with torch.no_grad():
            state_value_current, _ = self.online_net(states, False)
            advantage = returns - torch.sum(state_value_current.detach() * self.support, dim=-1)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-7)

        # get the surr loss
        surr_loss = self._get_surrogate_loss(states, advantage, actions, actions_logp)
        # comupte the surrogate gardient -> g, Ax = g, where A is the fisher information matrix
        surr_loss.backward(retain_graph=True)
        flat_surr_grad = torch.cat([param.grad.view(-1) for param in self.online_net.actor_parameters()]).data
        # use the conjugated gradient to calculate the scaled direction vector (natural gradient)
        nature_grad = self._conjugated_gradient(-flat_surr_grad, 10, states, actions_logp)
        # calculate the scaleing ratio
        non_scale_kl = 0.5 * (nature_grad * self._fisher_vector_product(nature_grad, states, actions_logp)).sum(0, keepdim=True)
        scale_ratio = torch.sqrt(non_scale_kl / self.max_kl)
        final_nature_grad = nature_grad / scale_ratio[0]
        # calculate the expected improvement rate...
        expected_improve = (-flat_surr_grad * nature_grad).sum(0, keepdim=True) / scale_ratio[0]
        # get the flat param ...
        prev_params = torch.cat([param.data.view(-1) for param in self.online_net.actor_parameters()])
        # start to do the line search
        success, new_params = self._line_search(prev_params, final_nature_grad,
                                                expected_improve, states, advantage, actions, actions_logp)
        self._set_flat_params_to(new_params)

        # critic update
        self.online_net.zero_grad()
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

        loss = (weights * value_loss).mean()

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
