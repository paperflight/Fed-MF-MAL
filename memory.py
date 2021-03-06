# -*- coding: utf-8 -*-
from __future__ import division

import copy
from collections import namedtuple
import numpy as np
import torch
import GLOBAL_PRARM as gp

scale_factor = 255
Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.uint8, (2, 47, 47)),
                             ('action', np.int8), ('action_logp', np.float, (gp.ACTION_NUM)),
                             ('neighbor_action', np.int8, (6 + 1)),
                             ('global_action', np.int8, (gp.NUM_OF_ACCESSPOINT)),
                             ('avail', np.bool, (gp.ACTION_NUM)),
                             ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans_aps = (0, np.zeros((2, 47, 47), dtype=np.uint8), 0, np.zeros(gp.ACTION_NUM),
                   np.ones(7, dtype=np.int8),
                   np.ones(gp.NUM_OF_ACCESSPOINT, dtype=np.int8), np.ones(gp.ACTION_NUM, dtype=np.bool), 0.0, False)


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2 ** (size - 1).bit_length() - 1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.array([blank_trans_aps] * size, dtype=Transition_dtype)  # Build structured array
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
          self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1))  # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(
            np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[
            successor_choices, np.arange(indices.size)]  # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]


class ReplayMemory:
    def __init__(self, args, capacity, remove_function=None):
        self.device = args.device
        self.current_action_obs = args.current_action_observable
        self.previous_action_obs_ap = args.previous_action_observable
        self.remove_function = remove_function
        self.capacity = capacity
        self.memory_capacity = args.memory_capacity
        self.history = args.history_length
        self.n = args.multi_step
        self.discount = args.discount
        self.priority_weight = args.priority_weight
        # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.transitions = SegmentTree(capacity)
        self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32,
                                           device=self.device)  # Discount-scaling vector for n-step returns
        # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, action_logp, nei_action, glob_action, avail, reward, terminal):
        if gp.ACTION_NUM == 6:
            action = int((action - 1) / 2)
        state_clip = torch.clone(state)
        state_clip[0] = (state_clip[0] + 1) / 2
        state_clip = state_clip.mul(scale_factor).to(dtype=torch.uint8, device=torch.device('cpu'))
        # Only store last frame and discretise to save memory
        self.transitions.append((self.t, state_clip, action, action_logp, nei_action, glob_action,
                                 avail, reward, not terminal),
                                self.transitions.max)  # Store new transition with maximum priority
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0
        # Returns the transitions with blank states where appropriate

    def _get_transitions(self, idxs):
        transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
        transitions = self.transitions.get(transition_idxs)
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1],
                                             transitions_firsts[:, t + 1])  # True if future frame has timestep 0
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1],
                                             transitions_firsts[:, t])  # True if current or past frame has timestep 0
        transitions[blank_mask] = blank_trans_aps
        return transitions

    # Returns a valid sample from a segment
    def _get_sample_from_segment(self, batch_size, p_total, avg=0):
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        while not valid:
            samples = np.random.uniform(0.0, segment_length,
                                        [batch_size]) + segment_starts  # Uniformly sample from within all segments
            probs, idxs, tree_idxs = self.transitions.find(samples)
            # Retrieve samples from tree with un-normalised probability
            if np.all((self.transitions.index - idxs) % self.capacity > self.n) and np.all(
                    (idxs - self.transitions.index) % self.capacity >= self.history) and np.all(probs != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transitions = self._get_transitions(idxs)
        all_state = transitions['state']
        # Create un-discretised state and nth next state, if number-step is 1, don't need to add another dims
        state = torch.tensor(all_state[:, :self.history], device=self.device, dtype=torch.float32).div_(scale_factor)
        state = torch.reshape(state, (batch_size, -1, state.shape[-2], state.shape[-1]))
        state[:, 0::gp.OBSERVATION_DIMS] = torch.round((state[:, 0::gp.OBSERVATION_DIMS] - 0.5) * 2)
        #TODO: change to torch.stack if state is two dim
        next_state = torch.tensor(all_state[:, self.n : self.n + self.history],
                                  device=self.device, dtype=torch.float32).div_(scale_factor)
        next_state = torch.reshape(next_state, (batch_size, -1, next_state.shape[-2], next_state.shape[-1]))
        next_state[:, 0::gp.OBSERVATION_DIMS] = torch.round((next_state[:, 0::gp.OBSERVATION_DIMS] - 0.5) * 2)
        if not self.current_action_obs:
            state[:, gp.OBSERVATION_DIMS * (self.history - 1), :, :] = \
                self.remove_function(state[:, gp.OBSERVATION_DIMS * (self.history - 1), :, :])
            next_state[:, gp.OBSERVATION_DIMS * (self.history - 1), :, :] = \
                self.remove_function(next_state[:, gp.OBSERVATION_DIMS * (self.history - 1), :, :])
        # Discrete action to be used as index
        action = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64, device=self.device)
        action_logp = torch.tensor(np.copy(transitions['action_logp'][:, self.history - 1]), dtype=torch.float32,
                                   device=self.device)
        nei_action = torch.tensor(np.copy(transitions['neighbor_action'][:, self.history - 1]),
                                  dtype=torch.int64, device=self.device)
        glob_action = torch.tensor(np.copy(transitions['global_action'][:, self.history - 1]),
                                   dtype=torch.int64, device=self.device)
        avail = torch.tensor(np.copy(transitions['avail'][:, self.history - 1]), dtype=torch.bool, device=self.device)
        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        R = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), device=self.device, dtype=torch.float32)
        R = torch.matmul(R - avg, self.n_step_scaling)
        # Mask for non-terminal nth next states
        nonterminal = torch.tensor(
            np.expand_dims(transitions['nonterminal'][:, self.history + self.n - 1], axis=1),
            dtype=torch.float32, device=self.device)

        return probs, idxs, tree_idxs, state, action, action_logp, nei_action, \
               glob_action, avail, R, next_state, nonterminal

    def get_relate_sample(self, batch_size, idxs):
        transitions = self._get_transitions(idxs)
        all_state = transitions['state']
        state = torch.tensor(all_state[:, :self.history], device=self.device, dtype=torch.float32).div_(scale_factor)
        state = torch.reshape(state, (batch_size, -1, state.shape[-2], state.shape[-1]))
        state[:, 0::gp.OBSERVATION_DIMS] = torch.round((state[:, 0::gp.OBSERVATION_DIMS] - 0.5) * 2)
        next_state = torch.tensor(all_state[:, self.n : self.n + self.history],
                                  device=self.device, dtype=torch.float32).div_(scale_factor)
        next_state = torch.reshape(next_state, (batch_size, -1, next_state.shape[-2], next_state.shape[-1]))
        next_state[:, 0::gp.OBSERVATION_DIMS] = torch.round((next_state[:, 0::gp.OBSERVATION_DIMS] - 0.5) * 2)
        if not self.current_action_obs:
            next_state[:, gp.OBSERVATION_DIMS * (self.history - 1), :, :] = \
                self.remove_function(next_state[:, gp.OBSERVATION_DIMS * (self.history - 1), :, :])
        avail = torch.tensor(np.copy(transitions['avail'][:, self.history - 1]), dtype=torch.bool, device=self.device)
        return state, next_state, avail

    def sample(self, batch_size, avg=0):
        p_total = self.transitions.total()
        # Retrieve sum of all priorities (used to create a normalised probability distribution)
        probs, idxs, tree_idxs, states, actions, action_logp, nei_action, glob_action, \
        avail, returns, next_states, nonterminals = self._get_sample_from_segment(batch_size, p_total, avg)
        probs = probs / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32,
                               device=self.device)  # Normalise by max importance-sampling weight from batch
        return (tree_idxs, idxs), states, actions, action_logp, \
               nei_action, glob_action, avail, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

        # Return valid states for validation

    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in reversed(range(self.history - 1)):
            blank_mask[t] = np.logical_or(blank_mask[t + 1],
                                          transitions_firsts[t + 1])  # If future frame has timestep 0
        transitions[blank_mask] = blank_trans_aps
        state = torch.tensor(transitions['state'], dtype=torch.float32, device=self.device).div_(scale_factor)
        state = torch.reshape(state, (-1, state.shape[-2], state.shape[-1]))
        state[0::gp.OBSERVATION_DIMS] = torch.round((state[0::gp.OBSERVATION_DIMS] - 0.5) * 2)
        # Agent will turn into batch
        if not self.current_action_obs:
            state[gp.OBSERVATION_DIMS * (self.history - 1), :, :] = \
                self.remove_function(state[gp.OBSERVATION_DIMS * (self.history - 1), :, :])
        self.current_idx += 1
        return state

    next = __next__  # Alias __next__ for Python 2 compatibility
