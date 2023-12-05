import cv2
import gym
import gym_super_mario_bros
import matplotlib.pyplot as plt
import numpy as np

from enums import World, Stage
from environment import create_environment
from moveset import MOVESET

def main():
    environment = gym.make(
        f"SuperMarioBros-1-1-v0",
        apply_api_compatibility=True
    )
    environment.reset()
    obs, *_ = environment.step(environment.action_space.sample())
    height, width = obs.shape[0], obs.shape[1]
    plt.imshow(obs)
    plt.imsave("images/1-1.svg", obs)
    plt.show()

    environment = create_environment(
        world=World.ONE,
        stage=Stage.ONE,
        moves=MOVESET,
    )
    obs, *_ = environment.step(environment.action_space.sample())
    obs = np.dstack((obs, obs, obs))
    obs = cv2.resize(obs, dsize=(height, width), interpolation=cv2.INTER_NEAREST)
    plt.imshow(obs, cmap="gray")
    plt.imsave("images/1-1_processed.svg", obs, cmap="gray")
    plt.show()

    return 0;


if __name__ == "__main__":
    exit(main())
