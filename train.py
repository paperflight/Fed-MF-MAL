# -*- coding: utf-8 -*-
#from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import sys

sys.path.append('../..')
sys.path.append('./')

import pickle
import GLOBAL_PRARM as gp

import numpy as np
import math
import copy
import torch
from tqdm import trange
from collections import defaultdict, deque

import multiprocessing
import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# TODO: When running in server, uncomment this line if needed
import copy as cp

from maddpg_sp.agent import Agent
from game import Decentralized_Game as Env
from memory import ReplayMemory
from test import test, test_p

# from pympler.tracker import SummaryTracker
# tracker = SummaryTracker()

# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--T-max', type=int, default=int(50e6), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                    help='Max episode length in game frames (0 to disable)')
# TODO: Note that the change of UAV numbers should also change the history-length variable
parser.add_argument('--previous-action-observable', action='store_false', help='Observe previous action? (AP)')
parser.add_argument('--current-action-observable', action='store_true', help='Observe previous action? (AP)')
parser.add_argument('--history-length', type=int, default=2, metavar='T',
                    help='Total number of history state')
parser.add_argument('--architecture', type=str, default='canonical_61obv_16ap', metavar='ARCH', help='Network architecture')
# TODO: if select resnet8, obs v8 and dims 4 should be set in gp
parser.add_argument('--hidden-size', type=int, default=256, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.3, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-2, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=3, metavar='V', help='Maximum of value distribution support')
# TODO: Make sure the value located inside V_min and V_max
parser.add_argument('--epsilon-min', type=float, default=0.0, metavar='ep_d', help='Minimum of epsilon')
parser.add_argument('--epsilon-max', type=float, default=0.0, metavar='ep_u', help='Maximum of epsilon')
parser.add_argument('--epsilon-delta', type=float, default=0.0001, metavar='ep_d', help='Decreasing step of epsilon')
# TODO: Set the ep carefully
parser.add_argument('--action-selection', type=str, default='boltzmann', metavar='action_type',
                    choices=['greedy', 'boltzmann', 'no_limit'],
                    help='Type of action selection algorithm, 1: greedy, 2: boltzmann')
parser.add_argument('--model', type=str, default=None, metavar='PARAM', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(12e3), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=1, metavar='n',
                    help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=1, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(8000), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=5, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
parser.add_argument('--reward-update-rate', type=float, default=0.01, metavar='η',
                    help='Average value step rate (for non-episodic task)')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--better-indicator', type=float, default=1.0, metavar='b',
                    help='The new model should be b times of old performance to be recorded')
# TODO: Switch interval should not be large
parser.add_argument('--learn-start', type=int, default=int(2000), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--data-reinforce', action='store_true', help='DataReinforcement')
# TODO: Change this after debug
parser.add_argument('--evaluation-interval', type=int, default=2000, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=400, metavar='N',
                    help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
# TODO: Change this after debug
parser.add_argument('--evaluation-size', type=int, default=20, metavar='N',
                    help='Number of transitions to use for validating Q')
# TODO: This evaluation-size is used for Q value evaluation, can be small if Q is not important
parser.add_argument('--render', action='store_false', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=0,
                    help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_false',
                    help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
# TODO: Change federated round each time
parser.add_argument('--federated-round', type=int, default=20, metavar='F',
                    help='Rounds to perform global combination, set a negative number to disable federated aggregation')

# Setup
args = parser.parse_args()

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('./results', args.id)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
# if torch.cuda.is_available() and not args.disable_cuda:
#     args.device = torch.device('cuda')
#     torch.cuda.manual_seed(np.random.randint(1, 10000))
#     torch.backends.cudnn.enabled = args.enable_cudnn
# else:
#     args.device = torch.device('cpu')
args.device = torch.device('cpu')


# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def average_weights(list_of_weight):
    """aggregate all weights"""
    averga_w = copy.deepcopy(list_of_weight[0])
    for key in averga_w.keys():
        for ind in range(1, len(list_of_weight)):
            averga_w[key] += list_of_weight[ind][key]
        averga_w[key] = torch.div(averga_w[key], len(list_of_weight))
    return averga_w


def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip, index=-1):
    # save ap mem
    memory_path = memory_path[0:-4] + '_aps_' + str(index) + memory_path[-4:]
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


def run_game_once_parallel_random(new_game, train_history_aps_parallel, episode):
    train_examples_aps = []
    for _ in range(new_game.environment.ap_number):
        train_examples_aps.append([])
    eps, done = 0, True
    while eps < episode:
        if done:
            done = new_game.reset()
        state, action, avail, reward, done = new_game.step()  # Step
        for index_p, ele_p in enumerate(state):
            neighbor_indice = new_game.environment.coop_graph.neighbor_indices(index_p, True)
            action_patch = np.append(action, [-1])
            train_examples_aps[index_p].append((ele_p, action[index_p], action_patch[neighbor_indice],
                                                action, avail[index_p], reward[index_p], done))
        eps += 1
    train_history_aps_parallel.append(train_examples_aps)


# Environment
env = Env(args)
action_space = env.get_action_size()

# Agent
dqn = []
matric = []
for _ in range(env.environment.ap_number):
    # dqn.append(temp)
    dqn.append(Agent(args, env, _))
    matric.append(copy.deepcopy(metrics))

global_model = Agent(args, env, "Global_")

# If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
if args.model is not None and not args.evaluate:
    if not args.memory:
        raise ValueError('Cannot resume training without memory save path. Aborting...')
    elif not os.path.exists(args.memory):
        raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))

    mem_aps = []
    for index in range(env.environment.ap_number):
        path = os.path.join(args.memory, ('metrics_aps' + str(index) + '.pth'))
        mem_aps.append(load_memory(path, args.disable_bzip_memory))
else:
    mem_aps = []
    for _ in range(env.environment.ap_number):
        mem_aps.append(ReplayMemory(args, args.memory_capacity, env.remove_previous_action))

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

# Construct validation memory
val_mem_aps = []
for _ in range(env.environment.ap_number):
    val_mem_aps.append(ReplayMemory(args, args.evaluation_size, env.remove_previous_action))
if not gp.PARALLEL_EXICUSION:
    T, done = 0, True
    while T < args.evaluation_size:
        if done:
            done = env.reset()
        state, action, avail, reward, done = env.step()
        for index, ele in enumerate(state):
            neighbor_indice = env.environment.coop_graph.neighbor_indices(index, True)
            action_patch = np.append(action, [-1])
            val_mem_aps[index].append(ele, action[index], action_patch[neighbor_indice],
                                      action, avail[index], reward[index], done)
        T += 1
else:
    num_cores = min(multiprocessing.cpu_count(), gp.ALLOCATED_CORES) - 1
    num_eps = math.ceil(args.evaluation_size / num_cores)
    # make sure each subprocess can finish all the game (end with done)
    with multiprocessing.Manager() as manager:
        train_history_aps = manager.list()

        process_list = []
        for _ in range(num_cores):
            process = multiprocessing.Process(target=run_game_once_parallel_random,
                                              args=(cp.deepcopy(env), train_history_aps, num_eps))
            process_list.append(process)

        for pro in process_list:
            pro.start()
        for pro in process_list:
            pro.join()
            pro.terminate()

        for res in train_history_aps:
            for index, memerys in enumerate(res):
                for state, a, na, ga, av, rw, done in memerys:
                    val_mem_aps[index].append(state, a, na, ga, av, rw, done)

if args.evaluate:
    for index in range(env.environment.ap_number):
        dqn[index].eval()  # Set DQN (online network) to evaluation mode
    (avg_pack) = test(args, 0, dqn, val_mem_aps, matric, results_dir, evaluate=True)  # Test
    for index in range(env.environment.ap_number):
        print('Avg. reward for ap' + str(index) + ': ' + str(avg_pack[0][index]) + ' | Avg. Q: ' + str(avg_pack[1][index]))
else:
    # Training loop
    T, aps_state, epsilon, done = 0, None, args.epsilon_max, env.reset()
    reinforce_ap = []
    for i in range(env.environment.ap_number):
        temp = []
        for j in range(3):
            temp.append([])
        reinforce_ap.append(temp)

    for T in trange(1, args.T_max + 1):
        if done and T > 2:
            done = env.reset()
            if T > 1 and args.data_reinforce:
                for index, ap_rein in enumerate(reinforce_ap):
                    for ap_pair in ap_rein:
                        for ap_ele in ap_pair:
                            mem_aps[index].append(ap_ele[0], ap_ele[1], ap_ele[2], ap_ele[3], ap_ele[4], ap_ele[5], ap_ele[6])
            reinforce_ap = []
            for i in range(env.environment.ap_number):
                temp = []
                for j in range(3):
                    temp.append([])
                reinforce_ap.append(temp)

        # training loop
        if T % args.replay_frequency == 0:
            for _ in range(env.environment.ap_number):
                dqn[_].reset_noise()

        state, action, avail, reward, done = env.step(dqn)
        epsilon = epsilon - args.epsilon_delta
        epsilon = np.clip(epsilon, a_min=args.epsilon_min, a_max=args.epsilon_max)

        for _ in range(env.environment.ap_number):
            if args.reward_clip > 0:
                reward[_] = torch.clamp(reward[_], max=args.reward_clip, min=-args.reward_clip) # Clip rewards
            neighbor_indice = env.environment.coop_graph.neighbor_indices(_, True)
            action_patch = np.append(action, [-1])
            mem_aps[_].append(state[_], action[_], action_patch[neighbor_indice], action, avail[_], reward[_], done)
            # Append transition to memory
            if args.data_reinforce:
                # data reinforcement, not applicapable with infinite environment
                obs = state[_]
                obs = torch.rot90(obs, 2, [1, 2])
                if action[_] != 12 and not reward[_] == 0:
                    reinforce_ap[_][0].append((obs, env.rot_action(action[_]),
                                               env.rot_action(action_patch[neighbor_indice]),
                                               env.rot_action(action), env.rot_avail(avail[_]), reward[_], done))
                    reinforce_ap[_][1].append((torch.flip(obs, [1]), env.flip_action(env.rot_action(action))[_],
                                               env.flip_action(env.rot_action(action_patch[neighbor_indice])),
                                               env.flip_action(env.rot_action(action)),
                                               env.flip_avail(env.rot_avail(avail[_])), reward[_], done))
                    reinforce_ap[_][2].append((torch.flip(state[_], [1]), env.flip_action(action)[_],
                                               env.flip_action(action_patch[neighbor_indice]),
                                               env.flip_action(action), env.flip_avail(avail[_]), reward[_], done))
                    # append rotated observation for data reinforcement

        if T >= args.learn_start:
            # tracker.print_diff()
            for index in range(env.environment.ap_number):
                mem_aps[index].priority_weight = min(mem_aps[index].priority_weight + priority_weight_increase, 1)
                # Anneal importance sampling weight β to 1

            if T % args.replay_frequency == 0:
                for index in range(env.environment.ap_number):
                    dqn[index].learn(mem_aps[index])  # Train with n-step distributional double-Q learning

            if T % args.federated_round == 0 and 0 < args.federated_round:
                global_weight = average_weights([model.get_state_dict() for model in dqn])
                global_target = average_weights([model.get_target_dict() for model in dqn])
                global_model.set_state_dict(global_weight)
                global_model.set_target_dict(global_target)
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' Global averaging starts')
                global_model.save(results_dir, 'Global_')
                average_reward = np.array([model.average_reward for model in dqn])
                average_reward = np.mean(average_reward)
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' Averaged reward is: ' + str(float(average_reward)))
                for models in dqn:
                    models.set_state_dict(global_weight)
                    models.set_target_dict(global_target)
                    models.average_reward = average_reward

            # If memory path provided, save it
            for index in range(env.environment.ap_number):
                if args.memory is not None:
                    save_memory(mem_aps[index], args.memory, args.disable_bzip_memory, index)

            # Update target network
            # if T % args.target_update == 0:  # uncomment for hard update
            for index in range(env.environment.ap_number):
                dqn[index].soft_update_target_net(1/args.target_update)

            # Checkpoint the network
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                for index in range(env.environment.ap_number):
                    dqn[index].save(results_dir, 'checkpoint' + str(index) + '.pth')

        if T % args.evaluation_interval == 0 and T >= args.learn_start:
            for index in range(env.environment.ap_number):
                dqn[index].eval()  # Set DQN (online network) to evaluation mode

            if gp.PARALLEL_EXICUSION:
                aps_pack = test_p(args, T, dqn, val_mem_aps, matric, results_dir)  # Test
            else:
                aps_pack = test(args, T, dqn, val_mem_aps, matric, results_dir)  # Test

            if aps_pack[2]:
                log('T = ' + str(T) + ' / ' + str(args.T_max) + '   Better model, accepted.')
            else:
                log('T = ' + str(T) + ' / ' + str(args.T_max) + '   Worse model, reject.')
            for index in range(env.environment.ap_number):
                log('T = ' + str(T) + ' / ' + str(args.T_max) + '  For ap' + str(index) +
                    ' | Avg. reward: ' + str(aps_pack[0][index]) + ' | Avg. Q: ' + str(aps_pack[1][index])
                    + ' | Avg. R: ' + str(float(dqn[index].average_reward)))

            for index in range(env.environment.ap_number):
                dqn[index].train()  # Set DQN (online network) back to training mode

env.close()
