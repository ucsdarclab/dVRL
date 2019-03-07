from gym import utils
from dVRL_simulator.PsmEnv_Position import PSMEnv_Position
import numpy as np

class PSMReachEnv(PSMEnv_Position):#, utils.EzPickle):
    def __init__(self, psm_num = 1, reward_type='sparse'):
        initial_pos = np.array([ 0,  0, -0.11])

        super(PSMReachEnv, self).__init__(psm_num = psm_num, n_substeps = 1, block_gripper = True,
                        has_object = False, target_in_the_air = True, height_offset = 0.01, 
                        target_offset = [0,0,0], obj_range = 0.05, target_range = 0.05,
                        distance_threshold = 0.003, initial_pos = initial_pos, reward_type = reward_type,
                        dynamics_enabled = False, two_dimension_only = False, 
                        randomize_initial_pos_obj = False, randomize_initial_pos_ee = False,
                        docker_container = "vrep_ee_reach")

        utils.EzPickle.__init__(self)