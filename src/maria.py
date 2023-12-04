import argparse

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO

from environment import create_environment
from enums import Stage, World
from moveset import MOVESET

def main():
    parser = argparse.ArgumentParser(description="Let artificial intelligence play Super Mario Bros.")
    parser.add_argument("--model", type=str, required=True, help="path to the pre-trained model file")
    parser.add_argument("--world", type=int, required=True, help="world (1-8)",)
    parser.add_argument("--stage", type=int, required=True, help="stage (1-4)")

    args = parser.parse_args()
    model_file = args.model
    world = args.world
    stage = args.stage

    if world not in World.__members__.values() or stage not in Stage.__members__.values():
        print("Invalid world or stage.")
        return 1

    # Create the environment for evaluation.
    environment = create_environment(
        world=world,
        stage=stage,
        moves=MOVESET,
        render="human"
    )
    environment = DummyVecEnv([lambda: environment])
    environment = VecFrameStack(environment, 4)

    # Load the pre-trained model.
    model = PPO.load(model_file)

    while True:
        obs = environment.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = environment.step(action)
            environment.render()

    return 0

if __name__ == "__main__":
    exit(main())
