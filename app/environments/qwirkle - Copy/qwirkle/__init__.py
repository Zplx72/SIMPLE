from gym.envs.registration import register

register(
    # id is the name used to identify the environment when calling gym.make(). 
    id='Qwirkle-v0',
    # entry_point is the path of the environment class in the package.
    entry_point='qwirkle.envs.qwirkle:QwirkleEnv',
)
