import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import numpy as np

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

class ForwardView(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        print(self.observation_space)
        self.observation_space = gym.spaces.Box(shape=(194, 224, 1), low=0, high=255, dtype=np.uint8)
    
    def observation(self, obs):
        return obs[31:225, 32:, :]


def create_environment(world, stage, version, moves):
    env = gym.make(f"SuperMarioBros-{world}-{stage}-v{version}", apply_api_compatibility=True)
    env = JoypadSpace(env, moves)
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
    env = ForwardView(env)
    env.reset()
    return env
