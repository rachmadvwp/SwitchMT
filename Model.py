import random

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, neuron, layer
import Neuron

class DuelingDQN(nn.Module):
    def __init__(self, env, sgn_dims, n_dendrites, timesteps = 4):
        super(DuelingDQN, self).__init__()

        self.env = env
        in_channels = 4
        n_actions = 18

        self.features = nn.Sequential(

            layer.Conv2d(in_channels, 32, kernel_size = 8, stride = 4),
            layer.BatchNorm2d(32),
            neuron.LIFNode(),

            layer.Conv2d(32, 64, kernel_size = 4, stride = 2),
            layer.BatchNorm2d(64),
            neuron.LIFNode(),

            layer.Conv2d(64, 64, kernel_size = 3, stride = 1),
            layer.BatchNorm2d(64),
            neuron.LIFNode(),

            layer.Flatten(),
        )

        self.a_fc1 = layer.Linear(7 * 7 * 64, 512)
        self.a_san = Neuron.ActiveIFNode(sgn_dims, n_dendrites)
        self.a_fc2 = layer.Linear(512, n_actions)
        self.a_nsif = Neuron.NonSpikingIFNode()

        self.v_fc1 = layer.Linear(7 * 7 * 64, 512)
        self.v_san = Neuron.ActiveIFNode(sgn_dims, n_dendrites)
        self.v_fc2 = layer.Linear(512, 1)
        self.v_nsif = Neuron.NonSpikingIFNode()

        self.timesteps = timesteps
    
    def forward(self, x, context_signal):

        output = 0
        context_signal = context_signal.float()

        for _ in range(self.timesteps):

            features = self.features(x)

            adv = self.a_fc1(features)
            adv = self.a_san(adv, context_signal)
            adv = self.a_fc2(adv)
            adv = self.a_nsif(adv)

            adv = self.a_nsif.v

            val = self.v_fc1(features)
            val = self.v_san(val, context_signal)
            val = self.v_fc2(val)
            val = self.v_nsif(val)

            val = self.v_nsif.v
            
            output += val + adv - adv.mean(dim = -1, keepdim = True)

        self.reset_net()
        return output/self.timesteps
    
    def act(self, state, context_signal, epsilon):

        if random.random() <= epsilon:
            return self.env.sample()
        
        with torch.no_grad():

            q_value = self.forward(state, context_signal)
            action = torch.argmax(q_value, dim = -1).item()
            return action
        
    def best_action(self, state, context_signal):

        with torch.no_grad():

            q_value = self.forward(state, context_signal)
            action = torch.argmax(q_value, dim = -1)
            return action
    
    def reset_net(self):

        functional.reset_net(self)