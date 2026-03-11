import random

import torch
import numpy as np

class ReplayBuffer:

    def __init__(self, capacity):

        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, env_idx):

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        self.buffer[self.position] = (state, action, reward, next_state, done, env_idx)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def collect(self, env, env_idx, n_transitions):

        state, _ = env.reset()

        for _ in range(n_transitions):

            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = torch.FloatTensor(np.array(state))
            next_state = torch.FloatTensor(np.array(next_state))
            
            self.push(state, action, reward, next_state, int(terminated or truncated), env_idx)

            if terminated or truncated:
                state, _ = env.reset()

            else:
                state = next_state


    def __len__(self):
        return len(self.buffer)
    
class MetaBuffer:

    def __init__(self, n_envs, capacity):
        self.buffers = [ReplayBuffer(capacity) for _ in range(n_envs)]
    
    def push(self, state, action, reward, next_state, done, env_idx):
        self.buffers[env_idx].push(state, action, reward, next_state, done, env_idx)
    
    def sample(self, batch_size, env_idx):
        return self.buffers[int(env_idx)].sample(batch_size)
    
    def collect(self, env, n_transitions):
        
        for env_idx in range(len(env)):
            self.buffers[env_idx].collect(env.env_list[env_idx], env_idx, n_transitions)