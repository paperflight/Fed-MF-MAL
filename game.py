import copy

import numpy as np
import GLOBAL_PRARM as gp
import math
import env
import torch
import time
import copy as cp
import mymatplotlib as myplt
from collections import defaultdict, deque
import typing

"""
    1) game.hash_observation:
        return games' current observations' string representation
        dtype = string
    2) game.get_observation:
        return [[num], [num]] observation of games' current state
        dtype = np.ndarray
    3) game.goto_next_state(action):
        take input action and move game to next state
        dtype = None
    4) game.end_game:
        return True if game end
        dtype = bool
    5) game.get_finial_reward:
        return the result reward if game end 
        dtype = float
    6) game.restart:
        restart a new round of game
    7) game.get_streng_observations(pi):
        input the policy
        rotate the observation and corresponding policy for data strengthern
        return Tuples of (observation, pi)
    8) game.convert_hash_observation:
        input observation
        return hashed_observation
        string
    9) game.get_valid_action:
        return valid action
        np.ndarray bool
    10) game.get_board_size:
        return board_x, board_y
        int, int
    11) game.get_action_size:
        return num of possible action
        int
"""


class Decentralized_Game:
    def __init__(self, args):
        self.args = args
        self.board_length_l = gp.LENGTH_OF_FIELD
        self.board_length_w = gp.WIDTH_OF_FIELD
        self.one_side_length = int(math.floor(gp.ACCESS_POINTS_FIELD - 1) / (2 * gp.SQUARE_STEP))
        self.environment = env.Channel(["square", gp.LENGTH_OF_FIELD, gp.WIDTH_OF_FIELD],
                                       ["PCP", [gp.DENSE_OF_USERS, gp.PCP_CLUSTER_NUM, gp.PCP_MAX_CLUSTER_SIZE]],
                                       ["Hex", gp.NUM_OF_ACCESSPOINT, gp.ACCESSPOINT_SPACE],
                                       [gp.ACCESS_POINT_TRANSMISSION_EIRP, 0, gp.AP_TRANSMISSION_CENTER_FREUENCY],
                                       [gp.ACCESS_POINT_TRANSMISSION_EIRP, 0, gp.AP_TRANSMISSION_CENTER_FREUENCY],
                                       ["3GPP-InH-LOS", "rayleigh_indirect", False, gp.AP_UE_ALPHA, gp.NAKAGAMI_M,
                                        'zero_forcing'],
                                       "Stronger First", gp.ACCESSPOINT_SPACE * 2 * np.sqrt(3) + 5)

        self.state_buffer = []
        self.history_buffer_length = args.history_length
        if args.history_length <= 1 and args.previous_action_observable:
            raise ValueError("Illegal setting avaliable previous action with less or equal than 1 history length")
        self.history_step = args.multi_step
        self.aps_observation = []

        # ---------reset replay buffer---------#
        self.state_buffer = []
        for _ in range(self.environment.ap_number):
            self.state_buffer.append(deque([], maxlen=self.args.history_length))

        for index in range(self.environment.ap_number):
            for _ in range(self.history_buffer_length):
                self.state_buffer[index].append(torch.zeros(gp.OBSERVATION_DIMS, int(self.one_side_length * 2 + 1),
                                                            int(self.one_side_length * 2 + 1), device=self.args.device))

    @staticmethod
    def get_action_size():
        return gp.ACTION_NUM

    def reset(self):
        np.random.seed(int(time.time() % 1 * 10e8))
        del self.environment
        self.environment = env.Channel(["square", gp.LENGTH_OF_FIELD, gp.WIDTH_OF_FIELD],
                                       ["PCP", [gp.DENSE_OF_USERS, gp.PCP_CLUSTER_NUM, gp.PCP_MAX_CLUSTER_SIZE]],
                                       ["Hex", gp.NUM_OF_ACCESSPOINT, gp.ACCESSPOINT_SPACE],
                                       [gp.ACCESS_POINT_TRANSMISSION_EIRP, 0, gp.AP_TRANSMISSION_CENTER_FREUENCY],
                                       [gp.ACCESS_POINT_TRANSMISSION_EIRP, 0, gp.AP_TRANSMISSION_CENTER_FREUENCY],
                                       ["3GPP-InH-LOS", "rayleigh_indirect", False, gp.AP_UE_ALPHA, gp.NAKAGAMI_M,
                                        'zero_forcing'],
                                       "Stronger First", gp.ACCESSPOINT_SPACE * 2 * np.sqrt(3) + 5)
        self.state_buffer = []
        self.aps_observation = []
        self.state_buffer = []
        for _ in range(self.environment.ap_number):
            self.state_buffer.append(deque([], maxlen=self.args.history_length))

        for index in range(self.environment.ap_number):
            for _ in range(self.history_buffer_length):
                self.state_buffer[index].append(torch.zeros(gp.OBSERVATION_DIMS, int(self.one_side_length * 2 + 1),
                                                            int(self.one_side_length * 2 + 1), device=self.args.device))
        return False

    def plot_grid_map(self, position_list):
        grid_map = np.zeros([int(self.board_length_l / gp.SQUARE_STEP), int(self.board_length_w / gp.SQUARE_STEP)],
                            dtype=bool)
        clusters_locations_norms = np.floor(position_list / gp.SQUARE_STEP).astype(int)
        for locations in clusters_locations_norms:
            grid_map[locations[0], locations[1]] = True
        return grid_map

    def get_observation_tensor(self):
        """:return List of tensor"""
        return [torch.tensor(aps_obv, dtype=torch.float32, device=self.args.device) for aps_obv in
                self.get_observation()]

    @staticmethod
    def pad_with_zeros(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    def get_observation(self):
        if gp.OBSERVATION_VERSION == 0:
            obs = self._get_observation_v0()
        elif gp.OBSERVATION_VERSION == 1:
            obs = self._get_observation_v1()
            # elif gp.OBSERVATION_VERSION == 2:
            #     obs = self._get_observation_v2()
        else:
            raise ValueError("Illegal observation version")

        pad_width = math.floor(1 + ((gp.ACCESS_POINTS_FIELD - 1) / 2) / gp.SQUARE_STEP)

        obs_decentral = []
        for index_obs in range(gp.OBSERVATION_DIMS):
            obs_decentral.append(np.pad(obs[index_obs], int(pad_width), self.pad_with_zeros, padder=0))
        obs_decentral = np.stack(obs_decentral, axis=0)

        aps_observation = []
        for ap_index, aps in enumerate(self.environment.ap_position):
            a = math.floor(aps[0] / gp.SQUARE_STEP) + pad_width - self.one_side_length
            b = math.floor(aps[0] / gp.SQUARE_STEP) + pad_width + self.one_side_length + 1
            c = math.floor(aps[1] / gp.SQUARE_STEP) + pad_width - self.one_side_length
            d = math.floor(aps[1] / gp.SQUARE_STEP) + pad_width + self.one_side_length + 1
            res_obs = obs_decentral[:, int(a):int(b), int(c):int(d)]

            aps_observation.append(res_obs)
        self.aps_observation = aps_observation
        # list ap: list uav: ndarray observation
        return self.aps_observation

    def _get_observation_v0(self):
        """
        ATTENTION: ASSIGN NEW POPULARITY BEFORE RUNNING THIS FUNTION!!!!!!!!!!

        :return Observation which is a x*x*3 matrix with position of uav, position of aps,
                ue position in largest cluster, total position with cluster number mark
        """

        observation = [np.zeros([np.floor(self.board_length_l / gp.SQUARE_STEP).astype(int),
                                 np.floor(self.board_length_w / gp.SQUARE_STEP).astype(int)], dtype=bool),
                       np.zeros([np.floor(self.board_length_l / gp.SQUARE_STEP).astype(int),
                                 np.floor(self.board_length_w / gp.SQUARE_STEP).astype(int)], dtype=bool)]

        ap_pos = np.floor(self.environment.ap_position / gp.SQUARE_STEP).astype(int)
        observation[0][ap_pos[:, 0], ap_pos[:, 1]] = True
        user_pos = np.floor(self.environment.user_position / gp.SQUARE_STEP).astype(int)
        observation[1][user_pos[:, 0], user_pos[:, 1]] = True

        self.observation = observation
        return self.observation

    def _get_observation_v1(self):
        """
        ATTENTION: ASSIGN NEW POPULARITY BEFORE RUNNING THIS FUNTION!!!!!!!!!!

        :return Observation which is a x*x*3 matrix with position of uav, position of aps,
                ue position in largest cluster, total position with cluster number mark
        """

        observation = [np.zeros([np.floor(self.board_length_l / gp.SQUARE_STEP).astype(int),
                                 np.floor(self.board_length_w / gp.SQUARE_STEP).astype(int)]),
                       np.zeros([np.floor(self.board_length_l / gp.SQUARE_STEP).astype(int),
                                 np.floor(self.board_length_w / gp.SQUARE_STEP).astype(int)])]

        ap_pos = np.floor(self.environment.ap_position / gp.SQUARE_STEP).astype(int)
        observation[0][ap_pos[:, 0], ap_pos[:, 1]] = True
        user_pos = np.floor(self.environment.user_position / gp.SQUARE_STEP).astype(int)
        for user_id, u_pos in enumerate(user_pos):
            observation[1][u_pos[0], u_pos[1]] += self.environment.user_qos[user_id, 0]
        observation[1] /= (gp.USER_QOS * 2)
        observation[1][observation[1] > 1] = 1
        # observation must bounded in 0-1

        self.observation = observation
        return self.observation

    @staticmethod
    def flip_avail(avail):
        if len(avail) == 6:
            act = np.where(avail == True)[0]
            act = act * 2 + 1
            act = ((6 - act % 6) + 6 * (act // 6) - 1) / 2
            new_avail = np.zeros(len(avail), dtype=bool)
            new_avail[act.astype(int)] = True
            return new_avail
        act = np.where(avail[0:12] == True)[0]
        act = ((6 - act % 6) + 6 * (act // 6)) % 12
        new_avail = np.zeros(len(avail), dtype=bool)
        new_avail[act.astype(int)] = True
        new_avail[-1] = True
        return new_avail

    @staticmethod
    def flip_action(act: int or np.ndarray):
        temp = ((6 - act % 6) + 6 * (act // 6)) % 12
        if type(act) is np.ndarray:
            temp[np.where(act == -1)[0]] = -1
        return temp

    @staticmethod
    def rot_action(act: int or np.ndarray):
        temp = (-6 + act + 12) % 12
        if type(act) is np.ndarray:
            temp[np.where(act == -1)[0]] = -1
        return temp

    @staticmethod
    def rot_avail(avail):
        if len(avail) == 6:
            act = np.where(avail == True)[0]
            act = act * 2 + 1
            act = ((-6 + act + 12) % 12 - 1) / 2
            new_avail = np.zeros(len(avail), dtype=bool)
            new_avail[act.astype(int)] = True
            return new_avail
        act = np.where(avail[0:12] == True)[0]
        act = (-6 + act + 12) % 12
        new_avail = np.zeros(len(avail), dtype=bool)
        new_avail[act.astype(int)] = True
        new_avail[-1] = True
        return new_avail

    def add_previous_action(self, ap_obs, ap_actual_action):
        hex_action_indices_map = [[3], [3, 5], [5], [4, 5], [4], [2, 4], [2], [0, 2], [0], [0, 1], [1], [1, 3], None]
        ap_obs = [ap_ob[-self.args.history_length::] for ap_ob in ap_obs]
        for ap_index, (ap_ob, aps, ap_act) in enumerate(zip(ap_obs, self.environment.ap_position, ap_actual_action)):
            neighbor_ind = np.where(ap_ob[0] == 1)
            neighbor_action = hex_action_indices_map[ap_act]
            if neighbor_action is None:
                self.state_buffer[ap_index][-1][0][neighbor_ind] = -1
                self.state_buffer[ap_index][-1][0][self.one_side_length, self.one_side_length] = 1
            else:
                self.state_buffer[ap_index][-1][0][neighbor_ind] = -1
                neighbor_enable = self.environment.coop_graph.neighbor_indices(ap_index, True)
                neighbor_enable_non = neighbor_enable[neighbor_enable != -1]
                neighbor_action = np.array(neighbor_action)
                neighbor_action[neighbor_action >= 3] += 1
                neighbor_action = np.insert(neighbor_action, 0, 3)
                matched_ind = [np.where(neighbor_enable_non == neighbor_enable[ind])[0] for ind in neighbor_action]
                for ind in matched_ind:
                    self.state_buffer[ap_index][-1][0][neighbor_ind[0][ind], neighbor_ind[1][ind]] = 1
                # for ind in matched_ind:
                #     self.state_buffer[ap_index][-1][0][neighbor_ind[0][ind], neighbor_ind[1][ind]] = \
                #         ap_actual_action[neighbor_enable_non[ind]].item(0) + 1
                # self.state_buffer[ap_index][-1][0] /= 12
        return [self.state_buffer[ind][-1] for ind in range(self.environment.ap_number)]

    @staticmethod
    def remove_previous_action(state):
        state[state != 0] = 1
        return state

    def end_game(self):
        if gp.ONE_EPISODE_RUN <= 0:
            return False
        if self.environment.time >= gp.ONE_EPISODE_RUN:
            return True
        return False

    def step(self, accesspoint=None, epsilon=0):
        """
            :parameter accesspoint: the models of access points
            :parameter accesspoint: the models of scheduler
            :parameter result_prob: output of network, with estimate weight of tiles for transmission
        """

        avil_action = self.environment.established()

        for index, tensor_obs in enumerate(self.get_observation_tensor()):
            self.state_buffer[index].append(tensor_obs)
        ap_state = [torch.cat(list(aps_obv), dim=0) for aps_obv in self.state_buffer]
        #  TODO: if state dims is two, change this to stack

        action = []
        action_logp = []
        if accesspoint is None:
            action = self.environment.random_action('randomnon12', avil_action)
            action_logp = [np.zeros(gp.ACTION_NUM) for _ in range(self.environment.ap_number)]
        else:
            # avil_action = [avil_action[ind][1::2] for ind in range(len(avil_action))]
            for index in range(self.environment.ap_number):
                action_ret = accesspoint[index].act_e_greedy(ap_state[index], avil_action[index],
                                                              epsilon, self.args.action_selection)
                if type(action_ret) is int:
                    action.append(action_ret)
                    action_logp.append(np.zeros(gp.ACTION_NUM))
                else:
                    action.append(action_ret[0])
                    action_logp.append(action_ret[1])
                # Choose an action greedily (with noisy weights)
        action = np.array(action)
        if gp.ACTION_NUM == 6:
            action_re = action * 2 + 1
        else:
            action_re = action

        actual_action = self.environment.set_action(action_re)
        reward = self.environment.decentralized_reward_exclude_central(self.environment.sinr_calculation(), actual_action)

        if self.args.previous_action_observable:
            ap_state = self.add_previous_action(ap_state, actual_action)

        if np.random.rand() < 0.5:
            print(reward, action_re, actual_action)
            myplt.plot_result_hexagon(self.environment.ap_position, action_re,
                                      self.environment.coop_graph.hand_shake_result,
                                      self.environment.user_position)

        return ap_state, action_re, action_logp, avil_action, \
               [torch.tensor(dec_rew).to(device=self.args.device) for dec_rew in reward], self.end_game()

    def step_p(self, accesspoint=None):
        """
            :parameter accesspoint: the models of access points
            :parameter accesspoint: the models of scheduler
            :parameter result_prob: output of network, with estimate weight of tiles for transmission
        """
        avil_action = self.environment.established()

        for index, tensor_obs in enumerate(self.get_observation_tensor()):
            self.state_buffer[index].append(tensor_obs)

        ap_state = [torch.cat(list(aps_obv), dim=0) for aps_obv in self.state_buffer]
        #  TODO: if state dims is two, change this to stack

        action = []
        action_logp = []
        if accesspoint is None:
            action = self.environment.random_action('randomnon12', avil_action)
            action_logp = [np.zeros(gp.ACTION_NUM) for _ in range(self.environment.ap_number)]
        else:
            # avil_action = [avil_action[ind][1::2] for ind in range(len(avil_action))]
            for index, pipe in enumerate(accesspoint):
                pipe.send((ap_state[index], avil_action[index]))
                action_ret = pipe.recv()
                if type(action_ret) is int:
                    action.append(action_ret)
                    action_logp.append(np.zeros(gp.ACTION_NUM))
                else:
                    action.append(action_ret[0])
                    action_logp.append(action_ret[1])
                # Choose an action greedily (with noisy weights)
        action = np.array(action)
        if gp.ACTION_NUM == 6:
            action_re = action * 2 + 1
        else:
            action_re = action

        actual_action = self.environment.set_action(action_re)
        reward = self.environment.decentralized_reward_exclude_central(self.environment.sinr_calculation(), actual_action)

        if self.args.previous_action_observable:
            ap_state = self.add_previous_action(ap_state, actual_action)

        return ap_state, action_re, action_logp, avil_action, \
               [torch.tensor(dec_rew).to(device=self.args.device) for dec_rew in reward], self.end_game()

    def close(self):
        del self.environment.coop_graph
        del self.environment
        del self
        return
