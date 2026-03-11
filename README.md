# SwitchMT: Scalable Multi-Task Learning through Spiking Neural Networks with Adaptive Task-Switching Policy for Intelligent Autonomous Agents

## Overview
This repository contains the implementation of SwitchMT methodology, which contains the following files:
- `Environment.py`: Defines the environment wrapper for Atari games
- `main.py`: The main training script
- `Model.py`: Neural network model architecture
- `Replay.py`: Experience replay buffer implementation
- `Neuron.py`: Custom neuron models

### Key features:
- Uses the SpikingJelly library for SNN components
- Implements spiking neurons (IF nodes) for more biologically plausible computation
- Processes information across multiple timesteps

### Active Dendrite Models
Versions with the "Active" suffix implement active dendrites, which are a biologically-inspired mechanism that allows neurons to perform more complex computations through dendritic processing. These models:
- Use context signals to modulate neuron behavior
- Implement custom active spiking neurons
- Feature enhanced context-dependent processing

### Dueling Architecture
It implements the dueling DQN-based architecture, which separates the estimation of state value and action advantages. This approach splits the network into value and advantage streams, combines them to produce Q-values, and offers better performance for states where actions don't significantly affect the outcome.

## Requirements
- Python 3.8+
- PyTorch 1.10+
- Gymnasium
- SpikingJelly
- NumPy

## Key Hyperparameters
- Batch Size: 256
- Buffer Size: 2^20
- Discount Factor (Gamma): 0.99
- Learning Rate: 1e-4
- Epsilon: Starts at 1.0, decays to 0.1 over 3M frames
- Target Network Update: Every 10,000 steps
- Training Duration: 12M frames (3M × 4)

## Usage
To train a model, navigate to the desired implementation directory and run:

```bash
python main.py
```

Each implementation saves model checkpoints and statistics during training in the respective directories:
- `./checkpoints/`: For model weights
- `./stats/`: For training statistics

## Meta-Environment
The project uses a meta-environment approach to handle multiple environments:
- Allows training on multiple Atari games in sequence
- Implements environment switching at specified intervals
- Facilitates continual learning experiments

## Citation
If you use and/or cite SwitchMT in your research or find it useful, kindly cite the following [article](https://arxiv.org/abs/2504.13541):
```
@article{Ref_SwitchMT,
  title={Scalable Multi-Task Learning through Spiking Neural Networks with Adaptive Task-Switching Policy for Intelligent Autonomous Agents},
  author={Putra, Rachmad Vidya Wicaksana and Devkota, Avaneesh and Shafique, Muhammad},
  journal={arXiv preprint arXiv:2504.13541},
  year={2025}
}
```
