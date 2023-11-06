import sys

from stable_baselines3 import PPO

from environment import create_environment
from movement import Move

model = PPO.load(sys.argv[1])

env = create_environment(1, 1, 0, moves=[[Move.NOOP], [Move.RIGHT, Move.B], [Move.RIGHT, Move.B, Move.A]])
state = env.reset()

while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
