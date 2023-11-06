import cv2
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# Hack to fix nes_py bug.
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

class ForwardView(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        print(self.observation_space)
        self.observation_space = gym.spaces.Box(shape=(194, 224, 1), low=0, high=255, dtype=np.uint8)
    
    def observation(self, obs):
        return obs[31:225, 32:, :]


class GrayScale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = np.expand_dims(observation, -1)
        return observation

def create_environment(world, stage, version, moves):
    env = gym.make(f"SuperMarioBros-{world}-{stage}-v{version}", apply_api_compatibility=True)
    env = JoypadSpace(env, moves)
    env = GrayScale(env)
    env = ForwardView(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order="last")
    env.reset()
    return env
