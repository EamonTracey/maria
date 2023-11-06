import argparse

from stable_baselines3 import PPO
from environment import create_environment
from movement import Move
from stages import World, Stage, Version

def main():
    parser = argparse.ArgumentParser(description="Train AI to play Super Mario Bros.")
    parser.add_argument("--world", type=int, help="World (1-8)", required=True)
    parser.add_argument("--stage", type=int, help="Stage (1-4)", required=True)
    parser.add_argument("--version", type=int, help="Version (0-3)", required=True)

    args = parser.parse_args()

    world = args.world
    stage = args.stage
    version = args.version

    if world not in World.__members__.values() \
    or stage not in Stage.__members__.values() \
    or version not in Version.__members__.values():
        print("Invalid world, stage, or version.")
        return

    # Create environment.
    environment = create_environment(
        world,
        stage,
        version, 
        moves=[
            [Move.NOOP],
            [Move.RIGHT, Move.B],
            [Move.RIGHT, Move.B, Move.A]
        ]
    )

    # Create Proximal Policy Optimization (PPO) model.
    model = PPO("CnnPolicy", environment)

    # Train PPO model.
    model.learn(100000, progress_bar=True)

    # Save PPO model.
    model.save("model.ppo")

if __name__ == "__main__":
    main()
