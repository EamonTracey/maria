from collections import deque

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

MAX_FRAMES_STUCK = 500

env = gym_super_mario_bros.make("SuperMarioBrosRandomStages-v0")
env = JoypadSpace(env, RIGHT_ONLY)

x_coordinates = deque(maxlen=MAX_FRAMES_STUCK)

done = True
while True:
    if done or (len(x_coordinates) == MAX_FRAMES_STUCK and len(set(x_coordinates)) == 1):
        env.reset()
    _, _, done, info = env.step(env.action_space.sample())

    # If Mario gets stuck, reset the environment.
    x_coordinates.append(info["x_pos"])

    env.render()

env.close()
