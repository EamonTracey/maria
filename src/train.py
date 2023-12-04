import argparse
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from environment import create_environment
from enums import World, Stage
from moveset import MOVESET

def main():
    parser = argparse.ArgumentParser(description="Train AI to play Super Mario Bros.")
    parser.add_argument("--world", type=int, required=True, help="world (1-8)")
    parser.add_argument("--stage", type=int, required=True, help="stage (1-4)")
    parser.add_argument("--vectors", type=int, default=4, help="number of vectors (processes)")
    parser.add_argument("--steps", type=int, default=100000, help="number of steps")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="learning rate")

    args = parser.parse_args()

    world = args.world
    stage = args.stage
    vectors = args.vectors
    steps = args.steps
    learning_rate = args.learning_rate

    if world not in World.__members__.values() or stage not in Stage.__members__.values():
        print("error: invalid world or stage.")
        return 1
    if vectors < 1:
        print("error: vectors must be greater than or equal to 1")
        return 1

    # Create environment.
    environments = [
        lambda: create_environment(
            world=world,
            stage=stage,
            moves=MOVESET,
            seed=s
        )
        for s in range(vectors)
    ]
    environment = SubprocVecEnv(environments, start_method="fork")
    environment = VecFrameStack(environment, 4)
    environment = VecMonitor(environment)

    # Create Proximal Policy Optimization (PPO) model.
    model = PPO(
        "CnnPolicy",
        environment,
        n_steps=2048,
        learning_rate=learning_rate,
        tensorboard_log="./board/"
    )

    # Create callback.
    callback = EvalCallback(
        environment,
        eval_freq=1000,
        best_model_save_path="./models/",
        log_path="./logs/",
        verbose=1
    )

    print(f"Training model!")
    print(f"World: {world}")
    print(f"Stage: {stage}")
    print(f"Vectors: {vectors}")
    print(f"Steps: {steps}")
    print(f"Learning rate: {learning_rate}")

    # Train PPO model.
    model.learn(steps, callback=callback, progress_bar=True)

    # Save PPO model.
    model.save(f"models/{world}-{stage}-{time.time()}.ppo")

    return 0

if __name__ == "__main__":
    exit(main())
