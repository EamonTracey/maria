import argparse
from stable_baselines3 import PPO

#from callbacks import 
from environment import create_environment
from enums import Move, Stage, World, Version

def main():
    parser = argparse.ArgumentParser(description="Let artificial intelligence play Super Mario Bros.")
    parser.add_argument("--model", type=str, help="Path to the pre-trained model file", required=True)
    parser.add_argument("--world", type=int, help="World (1-8)", required=True)
    parser.add_argument("--stage", type=int, help="Stage (1-4)", required=True)

    args = parser.parse_args()
    model_file = args.model
    world = args.world
    stage = args.stage

    if world not in World.__members__.values() or stage not in Stage.__members__.values():
        print("Invalid world or stage.")
        return

    # Create the environment for evaluation.
    environment = create_environment(
        world,
        stage,
        Version.STANDARD,
        moves=[
            [Move.RIGHT, Move.B],
            [Move.RIGHT, Move.B, Move.A]
        ],
        render="human"
    )

    # Load the pre-trained model.
    model = PPO.load(model_file)

    while True:
        obs = environment.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = environment.step(action)
            environment.render()

if __name__ == "__main__":
    main()
