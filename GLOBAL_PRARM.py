import numpy as np

LENGTH_OF_FIELD = 182
WIDTH_OF_FIELD = 162
REWARD_CAL_RANGE = 0.6  # reward calculation range for each accesspoint (range = RCR*ACCESS_FIELD)
NUM_OF_ACCESSPOINT = 20
ACCESSPOINT_SPACE = 13  # the edge of each HEX is 2 unit
ACCESS_POINTS_FIELD = np.floor(2 * np.sqrt(3) * ACCESSPOINT_SPACE * 2) + 3  # must be odd

DENSE_OF_USERS = 160
PCP_CLUSTER_NUM = 20
PCP_MAX_CLUSTER_SIZE = 50

MAX_USERS_MOBILITY = 1
USER_QOS = 10
USER_WAITING = 5
USER_ADDING = 5
ONE_EPISODE_RUN = 200
# 0 or negative for non-episodic run

AP_TRANSMISSION_CENTER_FREUENCY = 5  # GHz

SPEED_OF_LIGHT = 3e8

DRONE_HEIGHT = 40
EXCESSIVE_NLOS_ATTENUATION = pow(10, 20 / 10)

ACCESS_POINT_TRANSMISSION_EIRP = 43  # 43 dBm
ACCESS_POINT_TRANSMISSION_BANDWIDTH = 50e6  # Hz
UAV_TRANSMISSION_EIRP = 43  # 43 dBm
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
# TODO: Observation version 1-3: 3, 4: 4, 5: 5, 6: 5, 7: 5, 8: 4
OBSERVATION_DIMS = 2  # each cluster has two observations: ap position, user position
# TODO: when selecting observation 4 and 5, change the observation dims too
OBSERVATION_VERSION = 1  # 1: observation v1, 2 observation v2
# for details look into game.get_observation_vx() function
ACTION_NUM = 13


# training and loading parameters
ENABLE_MODEL_RELOAD = False
ENABLE_MEMORY_RELOAD = False
ENABLE_EARLY_STOP = False
ENABLE_EARLY_STOP_THRESHOLD = 0.5
LOAD_MODE = False
PARALLEL_EXICUSION = True
ALLOCATED_CORES = 4
