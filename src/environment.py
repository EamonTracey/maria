import gym
import gym_super_mario_bros

import wrappers

def create_environment(world, stage, version, moves, render=None):
    # Create the baseline Super Mario Bros. environment.
    # This directly uses the original Super Mario Bros. ROM,
    # meaning the input/output is exactly that of an NES/emulator.
    env = gym.make(f"SuperMarioBros-{world}-{stage}-v{version}", apply_api_compatibility=True, render_mode=render)

    # Restrict the actions in the environment to a limited set of moves.
    # The NES controller has 8 buttons, meaning there are 2**8 = 256
    # different button press combinations. However, we typically only
    # care about a select few (e.g., walk right, run right, jump, etc.)
    env = wrappers.MariaJoypad(env, moves)

    # The NES renders 8-bit color, which add a lot of variation to the
    # output frames. The exact colors are not very important to reinforcement
    # learning; rather pixels must be differntiable. Therefore, it is
    # enough to output a grayscaled version of each frame.
    env = wrappers.MariaGrayscale(env)

    # The 256x240 frame output is quite large, and it is not very necessary
    # as the model should only be concerned with simple observations such
    # as the location of Mario, enemies, walls, pipes, gaps, etc. Therefore,
    # we can pixelate each frame by grouping blocks of 4 pixels together.
    env = wrappers.MariaPixelated(env, 4)

    # The prior wrappers removed the color dimension from the environment.
    # This simply re-adds the color-dimension to output the format that
    # the model expects. Note that this dimension has no variation due
    # to the grayscale.
    env = wrappers.MariaImage(env)

    # Our model does not need to review every frame as adjacent frames are
    # very similar to each other. It is reasonable to make unique decisions
    # every 4 frames.
    env = wrappers.MariaSkip(env, 4)

    # The final two wrappers vectorize the environment and place a monitor.
    # These are necessary for API compatibility and to view model results
    # over time.
    env = wrappers.MariaVector([lambda: env])
    env = wrappers.MariaMonitor(env)

    env.reset()
    return env
