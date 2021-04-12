import numpy as np
import GLOBAL_PRARM as gp
import math
import env
import torch
import time
import copy as cp
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
        self.board_length = gp.LENGTH_OF_FIELD
        self.one_side_length = int(math.floor(gp.ACCESS_POINTS_FIELD - 1) / (2 * gp.SQUARE_STEP))
        self.environment = env.Channel(["square", gp.LENGTH_OF_FIELD, gp.LENGTH_OF_FIELD],
                                       ["PPP", gp.DENSE_OF_USERS],
                                       ["Hex", gp.NUM_OF_ACCESSPOINT, gp.ACCESSPOINT_SPACE],
                                       [gp.ACCESS_POINT_TRANSMISSION_EIRP, 0, gp.AP_TRANSMISSION_CENTER_FREUENCY],
                                       [gp.ACCESS_POINT_TRANSMISSION_EIRP, 0, gp.AP_TRANSMISSION_CENTER_FREUENCY],
                            ["alpha-exponential", "nakagami", False, gp.AP_UE_ALPHA, gp.NAKAGAMI_M, "zero_forcing"],
                            "Stronger First", gp.ACCESSPOINT_SPACE * 2 * np.sqrt(3) + 5)

        self.state_buffer = []
        self.scheduler_buffer = deque([], maxlen=self.args.history_length)
        self.history_buffer_length = args.history_length
        self.available_ap = np.zeros(self.environment.ap_number, dtype=bool)
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
        return 6

    def plot_grid_map(self, position_list):
        grid_map = np.zeros([int(self.board_length / gp.SQUARE_STEP), int(self.board_length / gp.SQUARE_STEP)],
                            dtype=bool)
        clusters_locations_norms = np.floor(position_list / gp.SQUARE_STEP).astype(int)
        for locations in clusters_locations_norms:
            grid_map[locations[0], locations[1]] = True
        return grid_map

    def get_observation_tensor(self):
        """:return List of tensor"""
        return [torch.tensor(aps_obv, dtype=torch.float32, device=self.args.device) for aps_obv in self.get_observation()]

    @staticmethod
    def pad_with_zeros(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value

    def get_observation(self):
        if gp.OBSERVATION_VERSION == 0:
            obs = self._get_observation_v0()
        # elif gp.OBSERVATION_VERSION == 2:
        #     obs = self._get_observation_v2()
        else:
            raise ValueError("Illegal observation version")

        if (gp.ACCESS_POINTS_FIELD - 1) % (2 * gp.SQUARE_STEP) != 0:
            raise ValueError("Access point field must be odd and diviable by step size")

        pad_width = math.floor(1 + ((gp.ACCESS_POINTS_FIELD - 1) / 2 - (gp.ACCESSPOINT_SPACE - 1)) / gp.SQUARE_STEP)

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

            if res_obs[-1].any():
                self.available_ap[ap_index] = True
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

        observation = [np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool),
                       np.zeros([np.floor(self.board_length / gp.SQUARE_STEP).astype(int),
                                  np.floor(self.board_length / gp.SQUARE_STEP).astype(int)], dtype=bool)]

        ap_pos = np.floor(self.environment.ap_position / gp.SQUARE_STEP).astype(int)
        observation[0][ap_pos[:, 0], ap_pos[:, 1]] = True
        user_pos = np.floor(self.environment.user_position / gp.SQUARE_STEP).astype(int)
        observation[1][user_pos[:, 0], user_pos[:, 1]] = True

        self.observation = observation
        return self.observation

    def step(self, accesspoint=None, epsilon=0):
        """
            :parameter accesspoint: the models of access points
            :parameter accesspoint: the models of scheduler
            :parameter result_prob: output of network, with estimate weight of tiles for transmission
        """

        self.environment.established()

        for index, tensor_obs in enumerate(self.get_observation_tensor()):
            self.state_buffer[index].append(tensor_obs)
        ap_state = [torch.cat(list(aps_obv), dim=0) for aps_obv in self.state_buffer]
        #  TODO: if state dims is two, change this to stack

        action = []
        if accesspoint is None:
            action = np.random.randint(6, size=self.environment.ap_number)
            action_re = action * 2 + 1
        else:
            for index in range(self.environment.ap_number):
                action.append(accesspoint[index].act_e_greedy(ap_state[index], self.available_ap[index],
                                                              epsilon, self.args.action_selection))
                # Choose an action greedily (with noisy weights)
            action_re = np.array(action) * 2 + 1

        self.environment.set_action(action_re)
        reward = self.decentralized_reward(self.environment.sinr_calculation())

        return ap_state, action, [torch.tensor(dec_rew).to(device=self.args.device) for dec_rew in reward], False

    def step_p(self, accesspoint=None):
        """
            :parameter accesspoint: the models of access points
            :parameter accesspoint: the models of scheduler
            :parameter result_prob: output of network, with estimate weight of tiles for transmission
        """
        self.environment.established()

        for index, tensor_obs in enumerate(self.get_observation_tensor()):
            self.state_buffer[index].append(tensor_obs)

        ap_state = [torch.cat(list(aps_obv), dim=0) for aps_obv in self.state_buffer]
        #  TODO: if state dims is two, change this to stack

        action = []
        if accesspoint is None:
            action = np.random.randint(6, size=self.environment.ap_number)
            action_re = action * 2 + 1
        else:
            for index, pipe in enumerate(accesspoint):
                pipe.send((ap_state[index], self.available_ap[index]))
                action.append(pipe.recv())
                # Choose an action greedily (with noisy weights)
            action_re = np.array(action) * 2 + 1

        self.environment.set_action(action_re)
        reward = self.decentralized_reward(self.environment.sinr_calculation())

        return ap_state, action, [torch.tensor(dec_rew).to(device=self.args.device) for dec_rew in reward], False

    def decentralized_reward(self, sinr):
        sinr_clip = sinr
        sinr_clip[sinr_clip > 8] = 8
        sinr_clip = (np.log10(sinr_clip / 8 + 1) * 2 - 0.35) * 10
        ap_observe_relation = np.stack([self.environment.user_position] * self.environment.ap_position.shape[0], axis=0) \
                              - np.stack([self.environment.ap_position] * self.environment.user_position.shape[0], axis=1)
        ap_observe_relation = np.all(np.absolute(ap_observe_relation) < int((gp.ACCESS_POINTS_FIELD - 1) / 2), axis=2)
        ap_distribute_reward = ap_observe_relation * sinr_clip
        normalized_factor = np.sum(ap_observe_relation, axis=1)
        normalized_factor[normalized_factor == 0] = 1
        ap_distribute_reward = np.sum(ap_distribute_reward, axis=1) / normalized_factor
        # normalization
        return ap_distribute_reward

    def decentralized_reward_directional(self, sinr, action):
        sinr_clip = sinr
        sinr_clip[sinr_clip > 8] = 8
        sinr_clip = (np.log10(sinr_clip / 8 + 1) * 2 - 0.35) * 10
        ap_observe_relation = np.stack([self.environment.user_position] * self.environment.ap_position.shape[0], axis=0) \
                              - np.stack([self.environment.ap_position] * self.environment.user_position.shape[0], axis=1)
        ap_observe_angle = np.arctan2(ap_observe_relation[:, :, 0], ap_observe_relation[:, :, 1]) * 180 / np.pi
        ap_observe_angle = ap_observe_angle - (150 - (np.ones([self.environment.ap_number, self.environment.user_number]).T * action).T * 30)
        ap_observe_angle = np.logical_and(ap_observe_angle < 0, ap_observe_angle > -120)
        ap_observe_relation = np.all(np.absolute(ap_observe_relation) < int((gp.ACCESS_POINTS_FIELD - 1) / 2), axis=2)
        ap_distribute_reward = np.logical_and(ap_observe_angle, ap_observe_relation) * sinr_clip
        normalized_factor = np.sum(ap_observe_relation, axis=1)
        normalized_factor[normalized_factor == 0] = 1
        ap_distribute_reward = np.sum(ap_distribute_reward, axis=1) / normalized_factor
        # normalization
        return ap_distribute_reward

    def close(self):
        del self
        return