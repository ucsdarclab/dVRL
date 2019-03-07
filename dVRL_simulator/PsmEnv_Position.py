import numpy as np

from dVRL_simulator.PsmEnv import PSMEnv
from dVRL_simulator.vrep.simObjects import table, obj, target

import transforms3d.euler as euler
import transforms3d.quaternions as quaternions


import time



def goal_distance(goal_a, goal_b):
	assert goal_a.shape == goal_b.shape
	return np.linalg.norm(goal_a - goal_b, axis=-1)


class PSMEnv_Position(PSMEnv):

	def __init__(self, psm_num, n_substeps, block_gripper,
				has_object, target_in_the_air, height_offset, target_offset, obj_range, target_range,
				distance_threshold, initial_pos, reward_type, dynamics_enabled, two_dimension_only,
				randomize_initial_pos_obj, randomize_initial_pos_ee, docker_container):

		"""Initializes a new signle PSM Position Controlled Environment
		Args:
			psm_num (int): which psm you are using (1 or 2)
			n_substeps (int): number of substeps the simulation runs on every call to step
			gripper_extra_height (float): additional height above the table when positioning the gripper
			block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
			has_object (boolean): whether or not the environment has an object
			target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
			height_offset (float): offset from the table for everything
			target_offset ( array with 3 elements): offset of the target, usually z is set to the height of the object
			obj_range (float): range of a uniform distribution for sampling initial object positions
			target_range (float): range of a uniform distribution for sampling a target Note: target_range must be set > obj_range
			distance_threshold (float): the threshold after which a goal is considered achieved
			initial_pos  (3x1 float array): The initial position for the PSM when reseting the environment. 
			reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
			dynamics_enabled (boolean): To enable dynamics or not
			two_dimension_only (boolean): To only do table top or not. target_in_the_air must be set off too.
			randomize_initial_pos_obj (boolean): If set true, it will randomize the initial position uniformly between 
										    [-target_range + initial_pos, target_range + initial_pos] for x and y
										    and [0+ initial_pos, initial_pos+ initial_pos] for z if target_in_air
			docker_container (string): name of the docke container that loads the v-rep

		"""
		#self.gripper_extra_height = gripper_extra_height
		self.block_gripper = block_gripper
		self.has_object = has_object
		self.target_in_the_air = target_in_the_air
		self.height_offset = height_offset
		self.target_offset = target_offset
		self.obj_range = obj_range
		self.target_range = target_range
		self.distance_threshold = distance_threshold
		self.initial_pos = initial_pos
		self.reward_type = reward_type
		self.dynamics_enabled = dynamics_enabled
		self.two_dimension_only = two_dimension_only
		self.randomize_initial_pos_obj = randomize_initial_pos_obj
		self.randomize_initial_pos_ee = randomize_initial_pos_ee




		if self.block_gripper:
			self.n_actions = 3 
			self.n_states  = 3 + self.has_object*3 
		else:
			self.n_actions = 4
			self.n_states  = 4 + self.has_object*3


		super(PSMEnv_Position, self).__init__(psm_num = psm_num, n_substeps=n_substeps, n_states = self.n_states, 
												n_goals = 3, n_actions=self.n_actions, camera_enabled = False,
												docker_container =docker_container)


		self.target = target(self.clientID, psm_num)
		if self.has_object:
			self.obj = obj(self.clientID)
		self.table = table(self.clientID)

		self.prev_ee_pos  = np.zeros((3,))
		self.prev_ee_rot  = np.zeros((3,))
		self.prev_obj_pos = np.zeros((3,))
		self.prev_obj_rot = np.zeros((3,))
		self.prev_jaw_pos = 0

		if(psm_num == 1):
			self.psm = self.psm1
		else:
			self.psm = self.psm2


		#Start the streaming from VREP for specific data:

		#PSM Arms:
		self.psm.getPoseAtEE(ignoreError = True, initialize = True)
		self.psm.getJawAngle(ignoreError = True, initialize = True)
		
		#Used for _sample_goal
		self.target.getPosition(self.psm.base_handle, ignoreError = True, initialize = True)

		#Used for _reset_sim
		self.table.getPose(self.psm.base_handle, ignoreError = True, initialize = True)
		if self.has_object:
			self.obj.getPose(self.psm.base_handle, ignoreError = True, initialize = True) #Also used in _get_obs

			#Used for _get_obs
			self.obj.isGrasped(ignoreError = True, initialize = True)


	# GoalEnv methods
	# ----------------------------

	def compute_reward(self, achieved_goal, goal, info):

		d = goal_distance(achieved_goal, goal)*self.target_range #Need to scale it back!

		if self.reward_type == 'sparse':
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			return -100*d

	# PsmEnv methods
	# ----------------------------

	def _set_action(self, action):
		assert action.shape == (self.n_actions,)
		action = action.copy()  # ensure that we don't change the action outside of this scope

		if self.block_gripper:
			pos_ctrl = action
		else:
			pos_ctrl, gripper_ctrl = action[:3], action[3]
			gripper_ctrl = (gripper_ctrl+1.0)/2.0

		pos_ee, quat_ee = self.psm.getPoseAtEE()
		pos_ee = pos_ee + pos_ctrl*0.001  # the maximum change in position is 0.1cm

		#Get table information to constrain orientation and position
		pos_table, q_table = self.table.getPose(self.psm.base_handle)

		#Make sure tool tip is not in the table by checking tt and which side of the table it is on

		#DH parameters to find tt position
		ct = np.cos(0)
		st = np.sin(0)

		ca = np.cos(-np.pi/2.0)
		sa = np.sin(-np.pi/2.0)

		T_x = np.array([[1,  0,  0, 0],
		               [0, ca, -sa, 0 ],
		               [0, sa,  ca, 0 ],
		               [0, 0, 0,    1 ]])
		T_z = np.array([[ct, -st, 0, 0],
		                [st,  ct, 0, 0],
		                [0,    0, 1, 0.0102],
		                [0,    0, 0, 1]])

		ee_T_tt = np.dot(T_x, T_z)

		pos_tt, quat_tt = self.psm.matrix2posquat(np.dot(self.psm.posquat2Matrix(pos_ee,quat_ee), ee_T_tt))

		pos_tt_on_table, distanceFromTable = self._project_point_on_table(pos_tt)

		#if the distance from the table is negative, then we need to project pos_tt onto the table top.
		#Or if two dim only are enabled
		if distanceFromTable < 0 or self.two_dimension_only:
			pos_ee, _ = self.psm.matrix2posquat(np.dot(self.psm.posquat2Matrix(pos_tt_on_table, quat_tt), np.linalg.inv(ee_T_tt)))


		#Get the constrained orientation of the ee
		temp_q =  quaternions.qmult([q_table[3], q_table[0], q_table[1], q_table[2]], [ 0.5, -0.5, -0.5,  0.5])
		rot_ctrl = np.array([temp_q[1], temp_q[2], temp_q[3], temp_q[0]])

		if self.block_gripper:
			gripper_ctrl = 0

		# #Make sure the new pos doesn't go out of bounds!!!
		upper_bound = self.initial_pos + self.target_range + 0.01
		lower_bound = self.initial_pos - self.target_range - 0.01

		pos_ee = np.clip(pos_ee, lower_bound, upper_bound)

		self.psm.setPoseAtEE(pos_ee, rot_ctrl, gripper_ctrl)

	def _get_obs(self):
		#Normalize ee_position:
		ee_pos,  _ = self.psm.getPoseAtEE()
		ee_pos = (ee_pos - self.initial_pos)/self.target_range

		jaw_pos = self.psm.getJawAngle()

		if self.has_object:
			#Normalize obj_pos:
			obj_pos,  _ = self.obj.getPose(self.psm.base_handle)
			obj_pos = (obj_pos - self.initial_pos)/self.target_range

			achieved_goal = np.squeeze(obj_pos)

			# if not self.obj.isGrasped():
			# 	obj_pos = np.zeros((3,))

			obs = np.concatenate((ee_pos, np.array([jaw_pos]), obj_pos)) 

		else:
			obj_pos = np.zeros((3,))
			achieved_goal = np.squeeze(ee_pos)
			if self.block_gripper:
				obs = ee_pos
			else:
				obs = np.concatenate((ee_pos, np.array([jaw_pos]))) 		

		self.prev_ee_pos  = ee_pos
		self.prev_ee_rot  = np.zeros((3,))
		self.prev_obj_pos = obj_pos
		self.prev_obj_rot = np.zeros((3,))
		self.prev_jaw_pos = jaw_pos

		return {
				'observation': obs.copy(),
				'achieved_goal': achieved_goal.copy(),
				'desired_goal' : self.goal.copy()
		}
		# return obs


	def _reset_sim(self):
		
		#Get the constrained orientation of the ee
		pos_table, q_table = self.table.getPose(self.psm.base_handle)
		b_T_table = self.psm.posquat2Matrix(pos_table, q_table)

		temp_q =  quaternions.qmult([q_table[3], q_table[0], q_table[1], q_table[2]], [ 0.5, -0.5, -0.5,  0.5])
		ee_quat_constrained = np.array([temp_q[1], temp_q[2], temp_q[3], temp_q[0]])

		#Put the EE in the correct orientation
		self.psm.setDynamicsMode(0, ignoreError = True)
		self._simulator_step

		if self.randomize_initial_pos_ee:

			if self.target_in_the_air:
				z = self.np_random.uniform(0, self.target_range) + 0.0102
			else:
				z = 0.0102

			#Add target_offset for goal. 
			deltaEEPos_b_homogeneous = np.append(self.np_random.uniform(-self.target_range, self.target_range, size=2),
									 [z, 0])
			deltaEEPos_b_homogeneous = np.dot(b_T_table, deltaEEPos_b_homogeneous)

			#Project EE on to the table and add the deltaEEPos to that
			pos_ee_projectedOnTable,_ = self._project_point_on_table(self.initial_pos)
			pos_ee = pos_ee_projectedOnTable + deltaEEPos_b_homogeneous[0:3]

		else:
			pos_ee = self.initial_pos

		self.psm.setPoseAtEE(pos_ee, ee_quat_constrained, 0, ignoreError = True)

		if self.has_object:
			self.obj.removeGrasped(ignoreError = True)
		self._simulator_step
		if self.dynamics_enabled:
			self.psm.setDynamicsMode(1, ignoreError = True)


		if self.has_object:

			#Get a random x,y vector to offset the object from the EE initial position.
			#	x,y plane is parallel to table top and is from -obj_range to obj_range
			#	z is perpindicular to the table top and is set to the height of the obj = 0.005

			z = 0.003
			
			dist_from_ee = 0
			while dist_from_ee < 0.005:

				if self.randomize_initial_pos_obj:
					x = self.np_random.uniform(-self.obj_range, self.obj_range)
					y = self.np_random.uniform(-self.obj_range, self.obj_range)
				else:
					x = 0
					y = 0

				deltaObject_b_homogeneous = np.dot(b_T_table,np.array([x,y,z,0]))

				#Project initial EE on to the table and add the deltaObject to that
				pos_ee_projectedOnTable,_ = self._project_point_on_table(self.initial_pos)
				obj_pos = pos_ee_projectedOnTable + deltaObject_b_homogeneous[0:3]

				if self.randomize_initial_pos_obj:
					dist_from_ee = np.linalg.norm(obj_pos - pos_ee)
				else:
					dist_from_ee = 1

			self.obj.setPose(obj_pos, q_table, self.psm.base_handle, ignoreError = True)

			self.prev_obj_pos = obj_pos
			self.prev_obj_rot = self.psm.quat2Euler(q_table)
		else:
			self.prev_obj_pos = self.prev_obj_rot = np.zeros((3,))

		self.prev_ee_pos  = pos_ee
		self.prev_ee_rot  = self.psm.quat2Euler(ee_quat_constrained)
		self.prev_jaw_pos = 0
		
		self._simulator_step()


		return True

	#Must be called immediately after _reset_sim since the goal is sampled around the position of the EE
	def _sample_goal(self):
		self._simulator_step()

		#Get a random x,y,z vector to offset from the EE from the goals initial position.
		#	x,y plane is parallel to table top and is from -target_range to target_range
		#	z is perpindicular to the table top and 
		#		if target_in_the_air from 0.0102 + self.height_offset to target_range + 0.0102 + self.heigh_offset 
		#		else 0.0102 + self.height_offset 
		pos_table, q_table = self.table.getPose(self.psm.base_handle)
		b_T_table = self.psm.posquat2Matrix(pos_table, q_table)

		if self.target_in_the_air:
			z = self.np_random.uniform(0, self.target_range)
		else:
			z = 0
		
		#Add target_offset for goal. 
		deltaGoal_b_homogeneous = np.append(self.np_random.uniform(-self.target_range, self.target_range, size=2),
								 [z, 0]) + np.append(self.target_offset, [0])
		deltaGoal_b_homogeneous = np.dot(b_T_table, deltaGoal_b_homogeneous)

		#Project EE on to the table and add the deltaGoal to that
		pos_ee_projectedOnTable,_ = self._project_point_on_table(self.initial_pos)

		goal = pos_ee_projectedOnTable + deltaGoal_b_homogeneous[0:3]

		self.target.setPosition(goal, self.psm.base_handle, ignoreError = True)
		self._simulator_step()

		goal = self.target.getPosition(self.psm.base_handle)
		goal = (goal - self.initial_pos)/self.target_range
 
		return goal.copy()

	def _is_success(self, achieved_goal, desired_goal):
		
		#if self.has_object:
		#	d = goal_distance(achieved_goal[3:], desired_goal[3:])
		#else:
		#	d = goal_distance(achieved_goal, desired_goal)
		d = goal_distance(achieved_goal, desired_goal)*self.target_range #Need to scale it back!

		return (d < self.distance_threshold).astype(np.float32)

	#Already accounts for height_offset!!!!
	def _project_point_on_table(self, point):
		pos_table, q_table = self.table.getPose(self.psm.base_handle)
		b_T_table = self.psm.posquat2Matrix(pos_table, q_table)

		normalVector_TableTop = b_T_table[0:3, 2]
		distanceFromTable = np.dot(normalVector_TableTop.transpose(), (point - ((self.height_offset)*normalVector_TableTop + pos_table)))
		point_projected_on_table = point - distanceFromTable*normalVector_TableTop

		return point_projected_on_table, distanceFromTable