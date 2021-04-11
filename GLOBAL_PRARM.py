import numpy as np

LENGTH_OF_FIELD = 200
REWARD_CAL_RANGE = 1  # reward calculation range for each accesspoint (range = RCR*ACCESS_FIELD)
NUM_OF_ACCESSPOINT = 16  # must be NxN
ACCESSPOINT_SPACE = 13  # the edge of each HEX is 2 unit
ACCESS_POINTS_FIELD = 4 * ACCESSPOINT_SPACE + 1  # must be odd

DENSE_OF_USERS = 250

MAX_USERS_MOBILITY = 1

AP_TRANSMISSION_CENTER_FREUENCY = 5e9

SPEED_OF_LIGHT = 3e8

DRONE_HEIGHT = 40
EXCESSIVE_NLOS_ATTENUATION = pow(10, 20 / 10)

ACCESS_POINT_TRANSMISSION_EIRP = pow(10, 46 / 10)  # 46 dBm
ACCESS_POINT_TRANSMISSION_BANDWIDTH = 50e6  # Hz
UAV_TRANSMISSION_EIRP = pow(10, 46 / 10)  # 46 dBm
UAV_TRANSMISSION_BANDWIDTH = 50e6  # Hz
NOISE_THETA = pow(10, -91 / 10)  # -91 dBm
AP_UE_ALPHA = -4
NAKAGAMI_M = 2
RAYLEIGH = 2
# https://arxiv.org/pdf/1704.02540.pdf

DEBUG = True
LOG_LEVEL = 0  # 0: nothing, 1: text_only, 2: rich text, 3: even detail+figure, 4: save figure
AP_COLOR = 'red'
UE_COLOR = 'green'
CS_COLOR = 'blue'

# training parameters
# observation square step
SQUARE_STEP = 2
# if step=2 : 0, 1, 2, 4, 8
USER_CLUSTER_INDICATOR_STEP = 2  # scale the length indicator to reduce the states num
# TODO: Observation version 1-3: 3, 4: 4, 5: 5, 6: 5, 7: 5, 8: 4
OBSERVATION_DIMS = 2  # each cluster has three observations: ap position, user position
REWARD_STAGE = [10, 15, 20]  # reward stage, correspoinding to -1, 0, 1, 1.5
# TODO: when selecting observation 4 and 5, change the observation dims too
OBSERVATION_VERSION = 0  # 1: observation v1, 2 observation v2
# for details look into game.get_observation_vx() function


# training and loading parameters
ENABLE_MODEL_RELOAD = False
ENABLE_MEMORY_RELOAD = False
ENABLE_EARLY_STOP = False
ENABLE_EARLY_STOP_THRESHOLD = 0.5
LOAD_MODE = False
PARALLEL_EXICUSION = True
ALLOCATED_CORES = 4
