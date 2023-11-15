import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from environment import create_environment
from enums import Move, World, Stage, Version

def main():
    parser = argparse.ArgumentParser(description="Train AI to play Super Mario Bros.")
    parser.add_argument("--world", type=int, help="World (1-8)", required=True)
    parser.add_argument("--stage", type=int, help="Stage (1-4)", required=True)
    parser.add_argument("--steps", type=int, default=100000, help="Number of steps")
    parser.add_argument("--learning-rate", type=float, default=0.00003, help="Learning rate")

    args = parser.parse_args()

    world = args.world
    stage = args.stage
    steps = args.steps
    learning_rate = args.learning_rate

    if world not in World.__members__.values() or stage not in Stage.__members__.values():
        print("Invalid world or stage.")
        return

    # Create environment.
    environment = create_environment(
        world,
        stage,
        Version.RECTANGLE,
        moves=[
            [Move.RIGHT, Move.B],
            [Move.RIGHT, Move.B, Move.A]
        ]
    )

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

    print(f"Training on {world}-{stage}")
    print(f"Steps: {steps}")
    print(f"Learning rate: {learning_rate}")

    # Train PPO model.
    model.learn(steps, callback=callback, progress_bar=True)

    # Save PPO model.
    model.save(f"models/{world}-{stage}.ppo")

if __name__ == "__main__":
    main()
