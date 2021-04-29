# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from scipy.special import softmax as softmax_sci
from torch.nn.utils import clip_grad_norm_
import GLOBAL_PRARM as gp
import math

from maddpg_sp.basic_block import Actor_Critic


# TODO: maddpg learns fast, set a small target network update rate or remove target nework for non-episodic cases


class Agent:
    def __init__(self, args, env, index):
        self.action_space = env.get_action_size()
        self.atoms = args.atoms
        self.action_type = args.action_selection
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.device = args.device
        self.net_type = args.architecture
        self.reward_update_rate = args.reward_update_rate
        self.average_reward = 0

        self.online_net = Actor_Critic(args, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
            self.model_path = os.path.join(args.model, "model" + str(index) + ".pth")
            if os.path.isfile(self.model_path):
                state_dict = torch.load(self.model_path, map_location='cpu')
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + self.model_path)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(self.model_path)

        self.online_net.train()

        self.target_net = Actor_Critic(args, self.action_space).to(device=args.device)

        self.online_dict = self.online_net.state_dict()
        self.target_dict = self.target_net.state_dict()

        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
        self.mseloss = torch.nn.MSELoss()

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

    # Acts based on single state (no batch)
    def act(self, state, avail=None):
        with torch.no_grad():
            if avail is None:
                return self.online_net(state.unsqueeze(0)).argmax(1).item()
            temp = self.online_net(state.unsqueeze(0)) * torch.tensor(avail)
            temp[:, avail == 0] = (torch.min(temp) - 100)
            return temp.argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, available=None, epsilon=0.3,
                     action_type='greedy'):  # High ε can reduce evaluation scores drastically
        if action_type == 'greedy':
            return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state,
                                                                                                         available)
        elif action_type == 'boltzmann':
            return self.act_boltzmann(state, available)
        elif action_type == 'no_limit':
            return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_boltzmann(self, state, avail):  # High ε can reduce evaluation scores drastically
        with torch.no_grad():
            res_policy = self.online_net(state.unsqueeze(0)).detach()
            return self.boltzmann(res_policy, [avail])

    def boltzmann(self, res_policy, mask):
        sizeofres = res_policy.shape
        res = []
        res_policy = softmax_sci(res_policy.numpy(), axis=1)
        for i in range(sizeofres[0]):
            action_probs = [res_policy[i][ind] * mask[i][ind] for ind in range(res_policy[i].shape[0])]
            count = np.sum(action_probs)
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
                        pipes.send(self.act_boltzmann(obs, avial).numpy())
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
    def _to_one_hot(temp, num_classes):
        y = temp.clone()
        y = torch.as_tensor(y)
        y[y == -1] = num_classes
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes + 1, dtype=y.dtype)
        zeros = zeros.scatter(scatter_dim, y_tensor, 1)
        return zeros[..., 0:num_classes]

    def blind_neighbor_observation(self, state, neighbor_action, target=True):
        with torch.no_grad():
            pad_width = math.floor(1 + ((gp.ACCESS_POINTS_FIELD - 1) / 2) / gp.SQUARE_STEP)
            one_side_length = int((state.size(-1) - 1) / 2)

            state_pad = torch.nn.functional.pad(state, (one_side_length, one_side_length, one_side_length, one_side_length),
                                                "constant", 0)

            returns = torch.zeros(*neighbor_action.size(), self.action_space, dtype=torch.float32)
            neighbor_ind = torch.where(state_pad[-1, state.size(-3) - gp.OBSERVATION_DIMS] != 0)
            neib_ind = 0
            for ap_index, aps_act in enumerate(neighbor_action[-1]):
                if aps_act == -1:
                    returns[:, ap_index] = 0
                else:
                    a = math.floor(neighbor_ind[0][neib_ind] / gp.SQUARE_STEP) + pad_width - one_side_length
                    b = math.floor(neighbor_ind[0][neib_ind] / gp.SQUARE_STEP) + pad_width + one_side_length + 1
                    c = math.floor(neighbor_ind[1][neib_ind] / gp.SQUARE_STEP) + pad_width - one_side_length
                    d = math.floor(neighbor_ind[1][neib_ind] / gp.SQUARE_STEP) + pad_width + one_side_length + 1
                    corp_state = state_pad[:, :, int(a):int(b), int(c):int(d)]
                    neib_ind += 1
                    if target:
                        returns[:, ap_index] = self.target_net(corp_state)
                    else:
                        returns[:, ap_index] = self.online_net(corp_state)
        return returns

    def learn(self, mem):
        # Sample transitions
        if gp.ONE_EPISODE_RUN > 0:
            self.average_reward = 0
        idxs, states, actions, neighbor_action, _, avails, returns, next_states, nonterminals, weights = \
            mem.sample(self.batch_size, self.average_reward)

        # critic update
        self.optimiser.zero_grad()

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_action = self.boltzmann(self.target_net(next_states), avails)
            next_nei_policy_out = self.blind_neighbor_observation(next_states, neighbor_action)
            next_q_values = self.online_net(next_states, False,
                                       torch.cat([next_nei_policy_out[:, 0:int((neighbor_action.size(1) - 1) / 2)],
                                                  self._to_one_hot(torch.tensor(next_q_action), self.action_space).unsqueeze(1),
                                                  next_nei_policy_out[:, int((neighbor_action.size(1) + 1) / 2)::]],
                                       dim=1))
            target_q_batch = returns.unsqueeze(1) + self.discount * nonterminals * next_q_values

        q_batch = self.online_net(states, False, self._to_one_hot(neighbor_action, self.action_space))
        value_loss = self.mseloss(q_batch, target_q_batch)

        # Actor update
        curr_policy_out = self.online_net(states)
        curr_nei_policy_out = self.blind_neighbor_observation(states, neighbor_action[0], False)
        policy_loss = -self.online_net(states, False,
                                       torch.cat([curr_nei_policy_out[:, 0:int((neighbor_action.size(1) - 1) / 2)],
                                                  curr_policy_out.unsqueeze(1),
                                                  curr_nei_policy_out[:, int((neighbor_action.size(1) + 1) / 2)::]],
                                       dim=1))
        policy_loss = policy_loss.mean()
        policy_loss += -(curr_policy_out ** 2).mean() * 1e-3

        (value_loss + policy_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 0.5)
        self.optimiser.step()

        self.average_reward = self.average_reward + \
                              self.reward_update_rate * torch.mean(target_q_batch.detach() - q_batch.detach())
        self.average_reward = self.average_reward.detach()
        if self.average_reward <= -1:
            self.average_reward = -1  # do some crop

        mem.update_priorities(idxs, value_loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def soft_update_target_net(self, tau):
        for target_param, param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, index=-1, name='model.pth'):
        if index == -1:
            torch.save(self.online_net.state_dict(), os.path.join(path, name))
        else:
            torch.save(self.online_net.state_dict(), os.path.join(path, name[0:-4] + str(index) + name[-4:]))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0))).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
