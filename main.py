import os
import numpy as np

import gymnasium as gym

import torch
import torch.nn as nn

import Model
import Replay
import Environment

BUFFER_SIZE = 2 ** 20
MIN_TRANSITIONS = 5_000
GAMMA = 0.99
LEARNING_RATE = 3e-4

EPSILON_START = 1.00
EPSILON_END = 0.10
EPSILON_DECAY = 1_000_000

UPDATE_FREQUENCY = 4
TARGET_UPDATE = 25_000
CHECKPOINT_DIR = './checkpoints/'
STATS_DIR = './stats/'

MAX_FRAMES = 3 * 4_000_000
TIMESTEPS = 4
TEST_INTERVAL = 3 * 50

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

if not os.path.exists(STATS_DIR):
    os.makedirs(STATS_DIR)

if torch.cuda.is_available():

    print("CUDA is available.")
    DEVICE = torch.device('cuda')

else:
    print("CUDA is not available.")
    DEVICE = torch.device('cpu')

def optimize_model(env_idx):

    transitions = memory.sample(BATCH_SIZE, env_idx)
    states, actions, rewards, next_states, dones, tasks = zip(*transitions)

    states = torch.stack(states, dim = 0).to(DEVICE)
    next_states = torch.stack(next_states, dim = 0).to(DEVICE)

    actions = torch.tensor(actions, dtype = torch.int64).unsqueeze(-1).to(DEVICE)
    rewards = torch.tensor(rewards, dtype = torch.float32).unsqueeze(-1).to(DEVICE)
    dones = torch.tensor(dones, dtype = torch.float32).unsqueeze(-1).to(DEVICE)

    tasks = torch.tensor(tasks, dtype = torch.int64).unsqueeze(-1).to(DEVICE)
    context_signals = nn.functional.one_hot(tasks, num_classes = len(env)).to(DEVICE).squeeze(1)

    with torch.no_grad():

        next_q_vals = target_network(next_states, context_signals)

        if USE_MULTI_GPU:
            next_actions = network.module.best_action(next_states, context_signals).unsqueeze(-1)
        
        else:
            next_actions = network.best_action(next_states, context_signals).unsqueeze(-1)

        chosen_q_vals = next_q_vals.gather(-1, next_actions.view(-1, 1))

        targets = (rewards + (1 - dones) * GAMMA * chosen_q_vals.view(-1, 1)).to(DEVICE)
    
    q_values = network(states, context_signals)
    predictions = (q_values.gather(-1, actions.view(-1, 1))).to(DEVICE)
    loss = nn.functional.smooth_l1_loss(predictions, targets)

    optimizer.zero_grad()
    loss.backward()

    for param in network.parameters():
        param.grad.data.clamp_(-1, 1)
    
    optimizer.step()

def test_agent(env, n_episodes, t):

    print(f"\nTesting the current model over {n_episodes} test episodes in each environment.")

    states, _ = env.reset_all()

    for i in range(len(env)):

        state = states[i]
        total_rewards = 0
        task = nn.functional.one_hot(torch.tensor(i), num_classes = len(env)).to(DEVICE)

        for _ in range(n_episodes):

            episodic_reward = 0

            while True:

                state = torch.FloatTensor(np.array(state)).to(DEVICE)

                if USE_MULTI_GPU:
                    action = network.module.best_action(state.unsqueeze(0), task)
                
                else:
                    action = network.best_action(state.unsqueeze(0), task)

                next_state, reward, terminated, truncated, _ = env.step(action)
                episodic_reward += reward

                if terminated or truncated:
                    
                    state, _ = env.reset()
                    total_rewards += episodic_reward
                    break
                
                else:
                    state = next_state
        
        print(f"Environment {i + 1}, Avg. Reward : {total_rewards/n_episodes:.3f}")

        with open(f'{STATS_DIR}/test_data_{i}.csv', 'a') as f:
            f.write(f"{t},{total_rewards/n_episodes}\n")
        
        env.switch_env()

    print()

class PerformanceTracker:
    
    def __init__(self, window_size=10, param_threshold=0.1, min_freq=50):
        
        self.window_size = window_size
        self.param_threshold = param_threshold
        self.min_freq = min_freq
        self.param_changes = []
        self.total_episodes = 0
        self.prev_param_dict = None
        
    def calculate_param_changes(self, network):
        """Calculate average relative parameter changes across all layers"""
        current_params = {}
        total_change = 0
        param_count = 0
        
        # Get current parameters
        for name, param in network.named_parameters():
            if param.requires_grad:
                current_params[name] = param.data.detach().float()
        
        # First run - just store parameters
        if self.prev_param_dict is None:
            self.prev_param_dict = current_params
            return float('inf')
            
        # Calculate changes for each parameter
        for name, curr_param in current_params.items():
            prev_param = self.prev_param_dict[name]
            param_norm = torch.norm(curr_param)
            
            if param_norm > 0:
                # Calculate relative change using L2 norm
                rel_change = torch.norm(curr_param - prev_param) / param_norm
                total_change += rel_change.item()
                param_count += 1
                
        # Update stored parameters
        self.prev_param_dict = current_params
        
        avg_change = total_change / param_count
        self.param_changes.append(avg_change)
        
        # Keep only recent history
        if len(self.param_changes) > self.window_size:
            self.param_changes.pop(0)
            
        return avg_change
    
    def should_switch_env(self, network):
        """Check if we should switch environments based on parameter convergence"""
        if self.total_episodes > self.min_freq:
            current_change = self.calculate_param_changes(network)
            
            # Need enough history to check for convergence
            if len(self.param_changes) >= self.window_size:
                avg_change = sum(self.param_changes) / len(self.param_changes)
                return avg_change < self.param_threshold
                
        return False
        
    def reset(self):
        """Reset tracker state for new environment"""
        self.param_changes = []
        self.total_episodes = 0
        self.prev_param_dict = None

def train_agent(env, test_env, frame = 0):
    
    print('Training the agent.')
    
    states, _ = env.reset_all()
    episodic_rewards = [0 for _ in range(len(env))]
    episodic_counts = [0 for _ in range(len(env))]
    
    performance_trackers = [
        PerformanceTracker(
            window_size = 5,
            param_threshold = 0.1,
            min_freq = 25
        ) for _ in range(len(env))
    ]
    
    total_episodes = 0
    
    for t in range(frame, MAX_FRAMES + 1):
        
        epsilon = np.interp(t, (0, EPSILON_DECAY), (EPSILON_START, EPSILON_END))
        current_env_idx, _ = env.get_env()
        
        state = states[current_env_idx]
        state = torch.FloatTensor(np.array(state))
        context = nn.functional.one_hot(torch.tensor(current_env_idx), num_classes=len(env)).to(DEVICE)
        
        if USE_MULTI_GPU:
            action = network.module.act(state.unsqueeze(0).to(DEVICE), context, epsilon)
        else:
            action = network.act(state.unsqueeze(0).to(DEVICE), context, epsilon)
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = torch.FloatTensor(np.array(next_state))
        
        memory.push(state, action, reward, next_state, int(terminated or truncated), current_env_idx)
        episodic_rewards[current_env_idx] += reward
        
        if terminated or truncated:
            
            states[current_env_idx], _ = env.reset()
            episodic_counts[current_env_idx] += 1
        
            tracker = performance_trackers[current_env_idx]
            tracker.total_episodes += 1
            
            if tracker.should_switch_env(network):
                if current_env_idx + 1 == len(env):
                    test_agent(test_env, 5, t)
                    checkpoint = {
                        'network': network.state_dict(),
                        'target_network': target_network.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'frames': t
                    }
                            
                    torch.save(checkpoint, f"{CHECKPOINT_DIR}/checkpoint_{t}.pth")
                    
                env.switch_env()
                tracker.reset()
                print(f"\nSwitching environment due to performance plateau after {episodic_counts[current_env_idx]} episodes\n")
                
            print(f"Step {t}, Environment {current_env_idx + 1}, Episode {episodic_counts[current_env_idx]}, "
                  f"Reward: {episodic_rewards[current_env_idx]}")
            
            with open(f'{STATS_DIR}/train_data_{current_env_idx}.csv', 'a') as f:
                f.write(f"{t},{episodic_counts[current_env_idx]},{episodic_rewards[current_env_idx]}\n")
                
            episodic_rewards[current_env_idx] = 0
        
        else:
            states[current_env_idx] = next_state
        
        if t % UPDATE_FREQUENCY == 0:
            
            for i in range(len(env)):
                optimize_model(i)
                
        if t % TARGET_UPDATE == 0:
            target_network.load_state_dict(network.state_dict())
            
env = Environment.env
test_env = Environment.test_env

SGN_DIMS = len(env)
N_DENDRITES = len(env)

memory = Replay.MetaBuffer(len(env), capacity = BUFFER_SIZE)
memory.collect(env, n_transitions = MIN_TRANSITIONS)

sgn_dims = len(env)
n_dendrites = len(env)

network = Model.DuelingDQN(env, SGN_DIMS, N_DENDRITES, TIMESTEPS)
target_network = Model.DuelingDQN(env, SGN_DIMS, N_DENDRITES, TIMESTEPS)

if torch.cuda.device_count() > 1:

    USE_MULTI_GPU = True
    network = nn.DataParallel(network)
    target_network = nn.DataParallel(target_network)

BATCH_SIZE = 1024 if not USE_MULTI_GPU else 1024 * torch.cuda.device_count()
network = network.to(DEVICE)
target_network = target_network.to(DEVICE)
target_network.load_state_dict(network.state_dict())

optimizer = torch.optim.Adam(network.parameters(), lr = LEARNING_RATE)
latest_checkpoint = None

USE_MULTI_GPU = True
CONTINUE_TRAINING = False

if len(os.listdir(CHECKPOINT_DIR)) > 0 and CONTINUE_TRAINING:

    checkpoints = [f'{CHECKPOINT_DIR}{checkpoint}' for checkpoint in os.listdir(CHECKPOINT_DIR)]
    latest_checkpoint = max(checkpoints, key = lambda x : int(x.split('_')[-1].split('.')[0]))
    checkpoint = torch.load(latest_checkpoint)

    if not USE_MULTI_GPU and 'module' in list(checkpoint['network'].keys())[0]:

        checkpoint['network'] = {k[7:]: v for k, v in checkpoint['network'].items()}
        checkpoint['target_network'] = {k[7:]: v for k, v in checkpoint['target_network'].items()}
    
    if USE_MULTI_GPU and 'module' not in list(checkpoint['network'].keys())[0]:
        
        checkpoint['network'] = {f"module.{k}": v for k, v in checkpoint['network'].items()}
        checkpoint['target_network'] = {f"module.{k}": v for k, v in checkpoint['target_network'].items()}

    network.load_state_dict(checkpoint['network'])
    target_network.load_state_dict(checkpoint['target_network'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(f"Loaded checkpoint {latest_checkpoint} with {checkpoint['frames']} frames.")

train_agent(env, test_env, frame = checkpoint['frames'] if latest_checkpoint is not None else 0)