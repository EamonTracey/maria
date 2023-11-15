import cv2
import gym
from nes_py.wrappers import JoypadSpace
import numpy as np
import stable_baselines3.common.atari_wrappers
import stable_baselines3.common.vec_env

# Hack to fix nes_py bug.
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)


class GrayScaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.observation_space.shape[:2],
            dtype=np.uint8
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return obs


class PixelatedViewWrapper(gym.ObservationWrapper):
    def __init__(self, env, block):
        super().__init__(env)

        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[0] // block, shape[1] // block),
            dtype=np.uint8
        )

    def observation(self, obs):
        shape = self.observation_space.shape
        obs = cv2.resize(obs, shape[::-1], interpolation=cv2.INTER_NEAREST)
        return obs


class ImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(*self.observation_space.shape, 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        obs = np.expand_dims(obs, -1)
        return obs


class SkipWrapper(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

MariaJoypad = JoypadSpace
MariaGrayscale = GrayScaleWrapper
MariaPixelated = PixelatedViewWrapper
MariaImage = ImageWrapper

MariaSkip = SkipWrapper

MariaVector = stable_baselines3.common.vec_env.DummyVecEnv
MariaMonitor = stable_baselines3.common.vec_env.VecMonitor
