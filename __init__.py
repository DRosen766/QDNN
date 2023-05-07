import numpy as np
from gym.envs.registration import register

register(
    id='qdnn',
    # entry_point='rfrl_gym.envs:RFRLGymEnv',
    max_episode_steps = np.inf
)
