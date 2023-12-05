from concurrent.futures import ProcessPoolExecutor
import random
import warnings

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO

from environment import create_environment
from enums import Stage, World
from moveset import MOVESET

warnings.filterwarnings("ignore")

def best_seeds(seeds):
    model_file = "crc/outputs/2-1-super/models/best_model.zip"
    world = World.TWO
    stage = Stage.ONE

    # Create the environment.
    environment = create_environment(
        world=world,
        stage=stage,
        moves=MOVESET,
    )
    environment = DummyVecEnv([lambda: environment])
    environment = VecFrameStack(environment, 4)

    model = PPO.load(model_file)

    seed_rewards = {}
    best = 0
    for seed in seeds:
        # Load the pre-trained model and set seed.
        model.set_random_seed(seed)

        obs = environment.reset()

        # Run the model.
        accumulated_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = environment.step(action)
            accumulated_reward += reward[0]

        seed_rewards[seed] = accumulated_reward
        print(f"Seed {seed} -> {accumulated_reward} reward")

        if accumulated_reward > best:
            best = accumulated_reward
            print("New best!")

    print(seed_rewards)

def main():
    with ProcessPoolExecutor() as executor:
        executor.map(best_seeds, [range(n * 100, n * 100 + 100) for n in range(10)])

if __name__ == "__main__":
    exit(main())
