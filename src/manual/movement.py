def right(env, frames):
    for _ in range(frames):
        env.step(3)
        env.render()

def right_jump(env, frames):
    for _ in range(frames):
        env.step(4)
        env.render()
