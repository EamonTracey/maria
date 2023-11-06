import argparse
from stable_baselines3 import PPO
from environment import create_environment
from movement import Move

def main():
    parser = argparse.ArgumentParser(description="Let artificial intelligence play Super Mario Bros.")
    parser.add_argument("--model_file", type=str, help="Path to the pre-trained model file", required=True)
    parser.add_argument("--world", type=int, help="World (1-8)", required=True)
    parser.add_argument("--stage", type=int, help="Stage (1-4)", required=True)

    args = parser.parse_args()
    model_file = args.model_file
    world = args.world
    stage = args.stage

    # Create the environment for evaluation.
    environment = create_environment(
        world,
        stage,
        0,
        moves=[
            [Move.NOOP],
            [Move.RIGHT, Move.B],
            [Move.RIGHT, Move.B, Move.A]
        ]
    )

    # Load the pre-trained model.
    model = PPO.load(model_file)

    total_reward = 0
    num_episodes = 10

    for _ in range(num_episodes):
        obs = environment.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = environment.step(action)
            episode_reward += reward
            environment.render()

        total_reward += episode_reward

    average_reward = total_reward / num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")

if __name__ == "__main__":
    main()
