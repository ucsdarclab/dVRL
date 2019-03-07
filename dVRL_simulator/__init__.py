from gym.envs.registration import registry, register, make, spec
from dVRL_simulator.environments.reach  import PSMReachEnv
from dVRL_simulator.environments.pick   import PSMPickEnv


register(
		id='dVRLReach-v0',
		entry_point='dVRL_simulator:PSMReachEnv',
		max_episode_steps=100,
)


register(
		id='dVRLPick-v0',
		entry_point='dVRL_simulator:PSMPickEnv',
		max_episode_steps=100,
	)
