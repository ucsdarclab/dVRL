from gym import utils
from dVRL_simulator.PsmEnv_Position import PSMEnv_Position
import numpy as np

class PSMPickEnv(PSMEnv_Position):#, utils.EzPickle):
    def __init__(self, psm_num = 1, reward_type='sparse', randomize_obj = False, randomize_ee = False):
        initial_pos = np.array([ 0,  0, -0.10])

        super(PSMPickEnv, self).__init__(psm_num = psm_num, n_substeps = 1, block_gripper = False,
                        has_object = True, target_in_the_air = True, height_offset = 0.0001,
                        target_offset = [0,0,0.005], obj_range = 0.025, target_range = 0.025,
                        distance_threshold = 0.003, initial_pos = initial_pos, reward_type = reward_type,
                        dynamics_enabled = False, two_dimension_only = False, 
                        randomize_initial_pos_obj = randomize_obj, randomize_initial_pos_ee = randomize_ee,
                        docker_container = "vrep_ee_pick")

        utils.EzPickle.__init__(self)