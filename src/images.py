import matplotlib.pyplot as plt

from enums import World, Stage
from environment import create_environment
from moveset import MOVESET

def main():
    environment = create_environment(
        world=World.ONE,
        stage=Stage.ONE,
        moves=MOVESET,
    )

    obs, *_ = environment.step(environment.action_space.sample())

    plt.imshow(obs, cmap="gray")
    plt.show()

    return 0;


if __name__ == "__main__":
    exit(main())
