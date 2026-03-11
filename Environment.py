import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
class MetaEnv:

    def __init__(self, env_list):

        self.env_list = env_list
        self.n_envs = len(env_list)
        self.env_idx = 0

    def step(self, action):
        return self.env_list[self.env_idx].step(action)

    def reset(self):
        return self.env_list[self.env_idx].reset()
    
    def reset_all(self):

        states, infos = [], []

        for env in self.env_list:

            state, info = env.reset()
            states.append(state)
            infos.append(info)
        
        return states, infos

    def switch_env(self):
        self.env_idx = (self.env_idx + 1) % self.n_envs

    def switch_to(self, env_idx):
        self.env_idx = env_idx
    
    def get_env(self):
        return self.env_idx, self.env_list[self.env_idx]

    def render(self):
        self.env_list[self.env_idx].render()
    
    def sample(self):
        return self.env_list[self.env_idx].action_space.sample()
    
    def close(self):

        for env in self.env_list:
            env.close()
    
    def __len__(self):
        return self.n_envs

# pong_env = gym.make('PongNoFrameskip-v4', full_action_space = True)
# pong_env = AtariPreprocessing(pong_env)
# pong_env = FrameStack(pong_env, num_stack = 4)

# breakout_env = gym.make('BreakoutNoFrameskip-v4', full_action_space = True)
# breakout_env = AtariPreprocessing(breakout_env)
# breakout_env = FrameStack(breakout_env, num_stack = 4)

# enduro_env = gym.make('EnduroNoFrameskip-v4', full_action_space = True)
# enduro_env = AtariPreprocessing(enduro_env)
# enduro_env = FrameStack(enduro_env, num_stack = 4)

# env = MetaEnv([pong_env, breakout_env, enduro_env])
# test_env = MetaEnv([pong_env, breakout_env, enduro_env])

boxing_env = gym.make('BoxingNoFrameskip-v4', full_action_space = True)
boxing_env = AtariPreprocessing(boxing_env)
boxing_env = FrameStack(boxing_env, num_stack = 4)

bowling_env = gym.make('BowlingNoFrameskip-v4', full_action_space = True)
bowling_env = AtariPreprocessing(bowling_env)
bowling_env = FrameStack(bowling_env, num_stack = 4)

skiing_env = gym.make('SkiingNoFrameskip-v4', full_action_space = True)
skiing_env = AtariPreprocessing(skiing_env)
skiing_env = FrameStack(skiing_env, num_stack = 4)

env = MetaEnv([boxing_env, bowling_env, skiing_env])
test_env = MetaEnv([boxing_env, bowling_env, skiing_env])