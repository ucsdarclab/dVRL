try:
	import dVRL_simulator.vrep.vrep as vrep
except:
	print ('--------------------------------------------------------------')
	print ('"vrep.py" could not be imported. This means very probably that')
	print ('either "vrep.py" or the remoteApi library could not be found.')
	print ('Make sure both are in the same folder as this file,')
	print ('or appropriately adjust the file "vrep.py"')
	print ('--------------------------------------------------------------')
	print ('')


import copy
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import error, spaces
from gym.utils import seeding

from dVRL_simulator.vrep.ArmPSM import ArmPSM
from dVRL_simulator.vrep.simObjects import camera


import subprocess
import re
import docker
import os
import time

class PSMEnv(gym.GoalEnv):


	"""Initializes a new double PSM Environment
		Args:
			psm_num (int): which psm to enable, if not 1 or 2 both are enabled
			n_actions (int): the number of actions possible in the environment
			n_states (int): the state dimension
			n_goals (int): the goal dimension
			n_substeps (int): number of substeps between each "step" of the environment.
			camera_enabled (bool): if the cameras should be enabled. This slows down the environment a lot...
			docker_container (string): name of the docke container that loads the v-rep
		"""

	def __init__(self, psm_num, n_actions, n_states, n_goals, n_substeps, camera_enabled, docker_container):

		self.viewer  = None

		#Create docker of the environment
		client = docker.from_env()
		environment = {"DISPLAY":os.environ['DISPLAY'], "QT_X11_NO_MITSHM":1}
		volumes = {"/tmp/.X11-unix": 
		               {
		                'bind': "/tmp/.X11-unix",
		                'mode':"rw"
		               }
		          }

		kwargs = {
		            'environment': environment,
		            'volumes': volumes,
		            'runtime': "nvidia",
		         }

		self.container = client.containers.run(docker_container, detach = True, **kwargs)
		proc = subprocess.Popen(['docker','inspect','-f', "'{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'", self.container.id],
                      stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT)
		self.container_ip = '.'.join(re.findall('\d+', str(proc.stdout.read())))


		#Try to connnect to v-rep via the port
		res = -1
		i = 0

		while res != vrep.simx_return_ok: 
			time.sleep(1)
			self.clientID = vrep.simxStart(self.container_ip,19999,True,True,5000,5) # Connect to V-REP
			res = vrep.simxSynchronous(self.clientID , True)
			i = i + 1

			if i == 10:
				break

		if res != vrep.simx_return_ok:
			raise IOError('V-Rep failed to load!')

		vrep.simxStartSimulation(self.clientID , vrep.simx_opmode_oneshot)

		#Only initializes the psm asked for, otherwise initializes both
		self.psm_num = psm_num
		if self.psm_num == 1:
			self.psm1 = ArmPSM(self.clientID, 1)
			self.psm2 = None
		elif self.psm_num == 2:
			self.psm1 = None
			self.psm2 = ArmPSM(self.clientID, 2)
		else:
			self.psm1 = ArmPSM(self.clientID, 1)
			self.psm2 = ArmPSM(self.clientID, 2)

		self.n_substeps     = n_substeps

		#Time step is set to in V-REP
		self.sim_timestep   = 0.1

		self.viewer = None
		self.camera_enabled = camera_enabled
		if self.camera_enabled:
			self.metadata = {'render.modes': ['matplotlib', 'rgb', 'human']}
			self.camera = camera(self.clientID, rgb = True)
		else:
			self.metadata = {'render.modes': ['human']}


		self.seed()
		self._env_setup()

		self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
		self.observation_space = spaces.Dict(dict(
			desired_goal=spaces.Box(-np.inf, np.inf, shape=(n_goals,), dtype='float32'),
			achieved_goal=spaces.Box(-np.inf, np.inf, shape=(n_goals,), dtype='float32'),
			observation=spaces.Box(-np.inf, np.inf, shape=(n_states,), dtype='float32'),
			))

	def __del__(self):
		self.close()

	@property
	def dt(self):
		return self.sim_timestep * self.n_substeps

	# Env methods
	# ----------------------------

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		action = np.clip(action, self.action_space.low, self.action_space.high)
		self._set_action(action)

		self._simulator_step()
		self._step_callback()

		obs = self._get_obs()

		done = False
		info = {
			#'is_success': self._is_success(obs[-3:], self.goal),
			'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
			}
		#reward = self.compute_reward(obs[-3:], self.goal, info)
		reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
		return obs, reward, done, info

	def reset(self):
		self.psm1.setBooleanParameter(vrep.sim_boolparam_display_enabled, False, ignoreError = True)
		# Attempt to reset the simulator. Since we randomize initial conditions, it
		# is possible to get into a state with numerical issues (e.g. due to penetration or
		# Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
		# In this case, we just keep randomizing until we eventually achieve a valid initial
		# configuration.
		did_reset_sim = False
		while not did_reset_sim:
			did_reset_sim = self._reset_sim()
		self.goal = self._sample_goal().copy()
		obs = self._get_obs()
		return obs

	def close(self):
		if self.viewer is not None:
			plt.close(self.viewer.number)
			self.viewer = None
		vrep.simxFinish(self.clientID)
		self.container.kill()



	def render(self, mode = 'human'):
		if mode == 'human':
			self.psm1.setBooleanParameter(vrep.sim_boolparam_display_enabled, True, ignoreError = True)
		elif mode == 'matplotlib' and self.camera_enabled:
			if self.viewer is None:
				self.viewer = plt.figure()
			plt.figure(self.viewer.number)
			img = self.camera.getImage()
			plt.imshow(img, origin='lower')
		elif mode == 'rgb' and self.camera_enabled:
			return self.camera.getImage()

	def _get_viewer(self):
		""" no viewer has been made yet! 
		"""		
		raise NotImplementedError()


	# Extension methods
	# ----------------------------

	def _simulator_step(self):
		for i in range(0, self.n_substeps):
			vrep.simxSynchronousTrigger(self.clientID)
		vrep.simxGetPingTime(self.clientID)

	def _reset_sim(self):
		"""Resets a simulation and indicates whether or not it was successful.
		If a reset was unsuccessful (e.g. if a randomized state caused an error in the
		simulation), this method should indicate such a failure by returning False.
		In such a case, this method will be called again to attempt a the reset again.
		"""

		return True

	def _get_obs(self):
		"""Returns the observation.
		"""
		raise NotImplementedError()

	def _set_action(self, action):
		"""Applies the given action to the simulation.
		"""
		raise NotImplementedError()

	def _is_success(self, achieved_goal, desired_goal):
		"""Indicates whether or not the achieved goal successfully achieved the desired goal.
		"""
		raise NotImplementedError()

	def _sample_goal(self):
		"""Samples a new goal and returns it.
		"""
		raise NotImplementedError()

	def _env_setup(self):
		"""Initial configuration of the environment. Can be used to configure initial state
		and extract information from the simulation.
		"""
		pass

	def _viewer_setup(self):
		"""Initial configuration of the viewer. Can be used to set the camera position,
		for example.
		"""
		pass

	def _render_callback(self):
		"""A custom callback that is called before rendering. Can be used
		to implement custom visualizations.
		"""
		pass

	def _step_callback(self):
		"""A custom callback that is called after stepping the simulation. Can be used
		to enforce additional constraints on the simulation state.
		"""
		pass