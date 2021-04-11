import numpy as np
from scipy.stats import nakagami, rayleigh
import scipy.spatial.distance as ssd
import typing
import copy as cp
from collections import defaultdict, deque
import GLOBAL_PRARM as gp
import mymatplotlib as myplt


class Connection_Graph:
    def __init__(self, ap_position, connect_threshold: int):
        dist_map = ssd.cdist(ap_position, ap_position)
        dist_map[np.where(dist_map < 1)] = 0
        if gp.DEBUG and connect_threshold <= 1 or dist_map.shape[0] != dist_map.shape[1]:
            raise ValueError("Too small connect threshold")
        self.ap_number = dist_map.shape[0]
        self.ap_side_number = int(np.sqrt(dist_map.shape[0]))
        self.connection_graph = np.ones(dist_map.shape)
        if gp.DEBUG and np.any(self.connection_graph < 0):
            raise ValueError("Exist Negative Distance")
        self.connection_graph[np.where(np.logical_or(dist_map > connect_threshold, dist_map < 1))] = 0
        if gp.DEBUG and np.any(np.sum(self.connection_graph, axis=0) > 6):
            raise ValueError("Graph Connection Error")
        self.decision = np.zeros(self.connection_graph.shape)
        self.hand_shake_result = np.zeros(self.connection_graph.shape)

    def neighbor_indices(self, target_indice):
        up_l_edge: int = target_indice % self.ap_side_number
        if (target_indice // self.ap_side_number) % 2 == 0:
            indi = np.array(
                [target_indice - self.ap_side_number - 1, target_indice - self.ap_side_number, target_indice - 1,
                 target_indice + 1, target_indice + self.ap_side_number - 1,
                 target_indice + self.ap_side_number])
            if up_l_edge == self.ap_side_number - 1:
                indi[3] = -1
            elif up_l_edge == 0:
                indi[[0, 2, 4]] = -1
        else:
            indi = np.array(
                [target_indice - self.ap_side_number, target_indice - self.ap_side_number + 1, target_indice - 1,
                 target_indice + 1, target_indice + self.ap_side_number,
                 target_indice + self.ap_side_number + 1])
            if up_l_edge == self.ap_side_number - 1:
                indi[[1, 3, 5]] = -1
            elif up_l_edge == 0:
                indi[2] = -1
        indi[np.where(np.logical_or(indi < 0, indi >= self.ap_number))] = -1
        return indi

    def hand_shake(self, ap_actions):
        hex_action_indices_map = [[3], [3, 5], [5], [5, 4], [4], [4, 2], [2], [2, 0], [0], [0, 1], [1], [1, 3], None]
        #     /11-0---1\
        #   10          2
        #  /             \
        # 9       12      3
        #  \             /
        #   8           4
        #    \7---6---5/
        # edge is 2 unit, ap-space present 1 unit
        # action map, the point point to joint present the action for both.
        #     3                type1 --- type2 --- type3
        # 1       5              |                   |
        #   self               type4     type0     type5
        # 0       4              |                   |
        #     2                type6 --- type7 --- type8
        # indices map, for six number in the connection graph
        self.decision = np.zeros(self.connection_graph.shape)
        for ap, ap_action in enumerate(ap_actions):
            hex_action = hex_action_indices_map[ap_action]
            if hex_action is None:
                self.decision[ap] = 0
            else:
                connected_ap = np.where(self.connection_graph[ap] == 1)[0]
                coop_indi = self.neighbor_indices(ap)[hex_action]
                if np.all(coop_indi == -1):
                    self.decision[ap] = 0
                else:
                    res_indi = coop_indi[coop_indi != -1]
                    if np.all(np.isin(res_indi, connected_ap)):
                        self.decision[ap][res_indi] = 1
                    else:
                        raise ValueError("Impossible selection, check neighbor indices function")

        hand_shake_bool = np.logical_and(self.decision, np.transpose(self.decision))
        self.hand_shake_result = self.decision * hand_shake_bool
        normalize_factor = np.sum(self.hand_shake_result, axis=1, keepdims=True)
        normalize_factor[np.where(normalize_factor == 0)] += 1
        self.hand_shake_result = (self.hand_shake_result / normalize_factor).round(decimals=1)
        # cut hand shake failures
        self.hand_shake_result = self.hand_shake_result + np.transpose(self.hand_shake_result)

        for ap, ap_action in enumerate(ap_actions):
            temp = np.where(self.hand_shake_result[ap] == 1)[0]
            if len(temp) == 2 and self.hand_shake_result[temp[0]][temp[1]] == 1:
                self.hand_shake_result[ap][temp] = 2
                # confirm the circles
                continue
            temp = np.where(self.hand_shake_result[ap] == 1.5)[0]
            if len(temp) == 2 and self.hand_shake_result[temp[0]][temp[1]] == 0:
                self.hand_shake_result[ap][temp] = 2
                self.hand_shake_result[temp[0]][[ap, temp[1]]] = 2
                self.hand_shake_result[temp[1]][[ap, temp[0]]] = 2
                # connect triangle connected aps
        self.hand_shake_result = np.floor(self.hand_shake_result / 1.5)


class Channel:
    def __init__(self, area, user_distribution, ap_distribution, user_parameters, ap_parameters, channel,
                 associate_type, connect_thre):
        self.user_distri_type, self.user_distri_para = user_distribution
        self.ap_distri_type, self.ap_number, self.ap_distri_space = ap_distribution
        # hex edge is 2 unit, ap-space present 1 unit
        self.user_number = 0
        self.user_trans_power, self.user_trans_gain, self.user_central_freq = user_parameters
        self.ap_trans_power, self.ap_trans_gain, self.ap_central_freq = ap_parameters
        self.large_scale_fading_type, self.small_scale_fading_type, self.non_los, self.large_scale_fading_parameter, \
        self.small_scale_fading_parameter, self.precoding = channel
        # example: ["alpha-exponential", "nakagami", False, gp.AP_UE_ALPHA, gp.NAKAGAMI_M, "zero_forcing"]
        self.area_shape, self.area_size_w, self.area_size_c = area

        # association
        self.associate_type = associate_type
        self.association_result = np.zeros([self.ap_number, self.user_number])

        # location matrixs
        self.ap_position = np.zeros([2, self.ap_number])
        self.user_position = np.zeros([2, self.ap_number])
        self.dist_matrix = np.zeros([self.ap_number, self.user_number])

        # fading matrixs
        self.power_gain = np.zeros(self.dist_matrix.shape, dtype=float)
        self.large_scale_fading = np.zeros(self.dist_matrix.shape, dtype=float)
        self.small_scale_fading = np.zeros(self.dist_matrix.shape, dtype=complex)
        self.line_of_sight = np.array([])
        self.channel = np.zeros(self.dist_matrix.shape, dtype=complex)

        # coop
        self.coop_graph = None
        self.connect_threshold = connect_thre
        self.coop_decision = np.zeros([self.ap_number, self.ap_number], dtype=int)

        # precoders
        self.zf_precoder_v = np.vectorize(self.zf_precoder)

    def number_init(self):
        if self.user_distri_type == "PPP":
            self.user_number = np.random.poisson(self.user_distri_para) + 1
        else:
            raise ValueError("Unknown User Distribution Type")
        if self.ap_distri_type == "Hex":
            calculated_ap = (int((gp.LENGTH_OF_FIELD - gp.ACCESSPOINT_SPACE) // (3 * gp.ACCESSPOINT_SPACE)) + 1) * \
                             (int((gp.LENGTH_OF_FIELD - np.sqrt(3) * gp.ACCESSPOINT_SPACE) //
                              (2 * np.sqrt(3) * gp.ACCESSPOINT_SPACE)) + 1)
            if calculated_ap != self.ap_number:
                raise ImportWarning("The actual ap number for Hex is "+str(calculated_ap) + ". Please Check input.")
            self.ap_number = calculated_ap
        else:
            raise ValueError("Unknown AP Distribution Type")
        self.dist_matrix = np.zeros([self.ap_number, self.user_number])
        self.association_result = np.zeros([self.ap_number, self.user_number])
        self.power_gain = np.zeros(self.dist_matrix.shape, dtype=float)
        self.large_scale_fading = np.zeros(self.dist_matrix.shape, dtype=float)
        self.small_scale_fading = np.zeros(self.dist_matrix.shape, dtype=complex)
        self.line_of_sight = np.array([])
        self.channel = np.zeros(self.dist_matrix.shape, dtype=complex)

    def location_init(self):
        if gp.DEBUG and self.user_number <= 0 or self.ap_number <= 0:
            raise ValueError("User/ap number invalid")
        self.user_position = \
            np.asarray([np.random.rand(2) * [self.area_size_w, self.area_size_c] for _ in range(self.user_number)])
        self.ap_position = \
            np.asarray([[x * 3 + 1, np.sqrt(3) * (y * 2 + 1 + x % 2)]
                        for x in range(int((gp.LENGTH_OF_FIELD - gp.ACCESSPOINT_SPACE) // (3 * gp.ACCESSPOINT_SPACE)) + 1)
                        for y in range(int((gp.LENGTH_OF_FIELD - np.sqrt(3) * gp.ACCESSPOINT_SPACE) //
                                       (2 * np.sqrt(3) * gp.ACCESSPOINT_SPACE)) + 1)]) \
            * self.ap_distri_space
        self.dist_matrix = ssd.cdist(self.ap_position, self.user_position)
        self.dist_matrix[np.where(self.dist_matrix < 1)] += 1
        self.coop_graph = Connection_Graph(self.ap_position, self.connect_threshold)

    def calculate_power_allocation(self):
        self.power_gain = np.ones(self.dist_matrix.shape) * (self.ap_trans_gain + self.ap_trans_power)

    def calculate_large_scale_fading(self):
        if self.large_scale_fading_type == "alpha-exponential":
            self.large_scale_fading = np.power(self.dist_matrix, self.large_scale_fading_parameter)
        elif self.large_scale_fading_type == "free-path-loss":
            self.large_scale_fading = np.power(4 * np.pi * self.dist_matrix *
                                               self.ap_central_freq / gp.SPEED_OF_LIGHT,
                                               self.large_scale_fading_parameter)
        elif self.large_scale_fading_type == "3GPP-InH-LOS":
            self.large_scale_fading = np.power(-(32.4 + 17.3 * np.log10(self.dist_matrix) + 20 *
                                                 np.log10(self.ap_central_freq) + np.random.normal(3)) / 10, 10)
        elif self.large_scale_fading_type == "3GPP-UMa-LOS":
            self.large_scale_fading = np.power(-(28 + 22 * np.log10(self.dist_matrix) + 20 *
                                                 np.log10(self.ap_central_freq) + np.random.normal(4)) / 10, 10)
        # Study on channel model for frequencies from 0.5 to 100 GHz

    def calculate_small_scale_fading(self):
        num_of_link = self.user_number * self.ap_number
        random_matrix = np.random.rand(num_of_link)
        if self.small_scale_fading_type == "nakagami":
            self.small_scale_fading = np.reshape(np.asarray(nakagami.rvs(self.small_scale_fading_parameter,
                                                                         size=num_of_link)) *
                                                 (np.cos(2 * np.pi * random_matrix) +
                                                  1j * np.sin(2 * np.pi * random_matrix)),
                                                 self.dist_matrix.shape)
        elif self.small_scale_fading_type == "rayleigh_indirect":
            self.small_scale_fading = np.reshape(np.asarray(rayleigh.rvs(size=num_of_link)) *
                                                 (np.cos(2 * np.pi * random_matrix) +
                                                  1j * np.sin(2 * np.pi * random_matrix)),
                                                 self.dist_matrix.shape)
        elif self.small_scale_fading_type == "rayleigh":
            self.small_scale_fading = np.reshape(np.asarray(np.random.normal(size=num_of_link) +
                                                            1j * np.random.normal(size=num_of_link)),
                                                 self.dist_matrix.shape)

    def calculate_association(self):
        self.association_result = self.association_result * 0
        if self.associate_type == "Stronger First":
            self.association_result[np.argmax(self.power_gain + self.large_scale_fading, axis=0),
                                    range(self.user_number)] = 1

    def map_association_with_coop_decision(self):
        self.coop_decision += np.eye(self.coop_decision.shape[0], dtype=int)
        association_coop_result = np.zeros(self.association_result.shape)
        for ap in range(self.association_result.shape[0]):
            association_coop_result[ap] = np.sum(self.association_result[np.where(self.coop_decision[ap] == 1)], axis=0)
        if gp.DEBUG and np.max(association_coop_result) > 1 and np.any(np.sum(association_coop_result, axis=0)):
            raise ValueError("Replicant Association or Unallocated User")
        self.coop_decision = association_coop_result

    @staticmethod
    def zf_precoder(a):
        if np.all(a == 0):
            return a
        if type(a) != np.ndarray:
            a = np.asarray([[a]], dtype=np.complex)
            return (np.linalg.inv(np.transpose(a.conj()) * a) * np.transpose(a.conj()))[0, 0]
        return np.linalg.inv(np.transpose(a.conj()) * a) * np.transpose(a.conj())

    def random_action(self):
        if not gp.DEBUG:
            raise TypeError("Function only called in Debug Mode")
        # self.coop_graph.hand_shake(np.random.randint(13, size=self.ap_number))
        # self.coop_graph.hand_shake(np.ones(self.ap_number, dtype=int) * 12)
        self.coop_graph.hand_shake(-np.power(-1, np.arange(self.ap_number, dtype=int)) * 3 + 3)
        # self.coop_graph.hand_shake(np.random.randint(6, size=self.ap_number) * 2 + 1)
        self.coop_decision = self.coop_graph.hand_shake_result

    def set_action(self, ap_action):
        if gp.DEBUG and len(ap_action) != self.ap_number:
            raise OverflowError("Unmatch action size")
        self.coop_graph.hand_shake(ap_action)
        self.coop_decision = self.coop_graph.hand_shake_result

    def precoder_ap_user(self):
        #  user_group: index of ap, users served by that ap within that ap
        precoder: np.ndarray = np.ones(self.channel.shape, dtype=complex)
        if self.precoding is None:
            return precoder
        precoder = self.coop_decision * self.small_scale_fading
        precoder = self.zf_precoder_v(precoder)
        if gp.LOG_LEVEL >= 2:
            myplt.table_print_color(precoder, "Precoder matrix for AP", gp.CS_COLOR)
        return precoder

    def sinr_ap_user(self):
        self.channel = np.power(self.power_gain / 10, 10) * self.large_scale_fading
        precoder = self.precoder_ap_user()
        mask: np.ndarray = self.coop_decision - \
                           0.5 * np.ones(self.coop_decision.shape) * np.sum(self.coop_decision, axis=1, keepdims=True,
                                                                            dtype=bool)
        mask = mask * 2  # convert to -1, 0, 1
        signal_mask, interference_mask = np.copy(mask), np.copy(mask)
        signal_mask[signal_mask < 0] = 0
        interference_mask[interference_mask > 0] = 0
        precoder *= signal_mask
        precoder[precoder == 0] = 1
        self.channel = self.channel * np.square(np.absolute(precoder * self.small_scale_fading))
        signal = self.channel * signal_mask
        interference = self.channel * interference_mask
        sinr = np.sum(signal, axis=0) / (-np.sum(interference, axis=0) +
                                         np.sum(gp.NOISE_THETA * np.square(np.absolute(precoder)) * signal_mask, axis=0)
                                         / np.sum(signal_mask, axis=0))
        if gp.LOG_LEVEL >= 2:
            myplt.table_print_color(sinr, "SINR for UE", gp.UE_COLOR)
        return sinr

    def sinr_ap_user_noncoop(self):
        self.channel = np.power(self.power_gain / 10, 10) * self.large_scale_fading
        precoder = self.precoder_ap_user()
        mask: np.ndarray = self.association_result - \
                           0.5 * np.ones(self.coop_decision.shape) * np.sum(self.association_result, axis=1, keepdims=True,
                                                                            dtype=bool)
        mask = mask * 2  # convert to -1, 0, 1
        signal_mask, interference_mask = np.copy(mask), np.copy(mask)
        signal_mask[signal_mask < 0] = 0
        interference_mask[interference_mask > 0] = 0
        precoder *= signal_mask
        precoder[precoder == 0] = 1
        self.channel = self.channel * np.square(np.absolute(precoder * self.small_scale_fading))
        signal = self.channel * signal_mask
        interference = self.channel * interference_mask
        sinr = np.sum(signal, axis=0) / (-np.sum(interference, axis=0) +
                                         np.sum(gp.NOISE_THETA * np.square(np.absolute(precoder)) * signal_mask, axis=0)
                                         / np.sum(signal_mask, axis=0))
        if gp.LOG_LEVEL >= 2:
            myplt.table_print_color(sinr, "SINR for UE", gp.UE_COLOR)
        return sinr

    def established(self):
        self.number_init()
        self.location_init()
        self.calculate_power_allocation()
        self.calculate_large_scale_fading()
        self.calculate_small_scale_fading()
        self.calculate_association()

    def sinr_calculation(self):
        self.map_association_with_coop_decision()
        return self.sinr_ap_user()

    def test_sinr(self):
        self.number_init()
        self.location_init()
        self.calculate_power_allocation()
        self.calculate_large_scale_fading()
        self.calculate_small_scale_fading()
        self.calculate_association()
        self.random_action()
        self.map_association_with_coop_decision()
        sinr = self.sinr_ap_user()
        if gp.LOG_LEVEL >= 2:
            associ_position = self.ap_position[np.argmax(self.association_result, axis=0)]
            myplt.table_print_color(np.stack((sinr,
                                              np.sqrt(np.sum(np.power(self.user_position - associ_position, 2),
                                                             axis=1))), axis=1), "SINR_DISTANCE", gp.UE_COLOR)
        return sinr

    def decentralized_reward(self, sinr):
        sinr_clip = sinr
        # sinr_clip[sinr_clip > 100] = 0
        sinr_clip[sinr_clip > 8] = 8
        sinr_clip = (np.log10(sinr_clip / 8 + 1) * 2 - 0.35) * 10
        # sinr_x = self.sinr_ap_user_noncoop()
        # sinr_x[np.where(sinr_clip == 0)] = 0
        # # sinr_x[sinr_x > 8] = 8
        # sinr_x = np.log10(sinr_x / 8 + 1) * 2
        # sinr_clip = sinr_clip - sinr_x
        ap_observe_relation = np.stack([self.user_position] * self.ap_position.shape[0], axis=0) \
                              - np.stack([self.ap_position] * self.user_position.shape[0], axis=1)
        ap_observe_relation = np.all(np.absolute(ap_observe_relation) < int((gp.ACCESS_POINTS_FIELD - 1) / 2), axis=2)
        ap_distribute_reward = ap_observe_relation * sinr_clip
        normalized_factor = np.sum(ap_observe_relation, axis=1)
        normalized_factor[normalized_factor == 0] = 1
        ap_distribute_reward = np.sum(ap_distribute_reward, axis=1) / normalized_factor
        # normalization
        return ap_distribute_reward


if __name__ == "__main__":
    x = Channel(["square", gp.LENGTH_OF_FIELD, gp.LENGTH_OF_FIELD], ["PPP", 250], ["Hex", 20, 13], [28, 15, 5e8], [28, 15, 5e8],
                ["alpha-exponential", "nakagami", False, gp.AP_UE_ALPHA, gp.NAKAGAMI_M, None], "Stronger First",
                13 * 2 * np.sqrt(3) + 5)
    # ["square", 150, 150], ["PPP", 250], ["Hex", 16, 13], [28, 15, 5e8], [28, 15, 5e8],
    #             ["alpha-exponential", "nakagami", False, gp.AP_UE_ALPHA, gp.NAKAGAMI_M, "zero_forcing"], "Stronger First",
    #             13 * 2 * np.sqrt(3) + 5
    res = x.decentralized_reward(x.test_sinr())
    print(np.mean(res))