import torch
import torch.nn as nn
from spikingjelly.activation_based import base, neuron
from abc import abstractmethod

class ActiveIFNode(neuron.IFNode):

    def __init__(self, sgn_dims, n_dendrites):
        super().__init__()

        self.sgn_dims = sgn_dims
        self.active_dendrites = nn.Parameter(torch.randn(sgn_dims, n_dendrites))
    
    def forward(self, x, context):

        dendritic_activations = torch.matmul(context, self.active_dendrites)
        maximal_dendritic_activation = torch.max(dendritic_activations, dim = -1).values

        x_shape = x.shape
        new_shape = x_shape[:1] + (1,) * (len(x_shape) - 1)
        maximal_dendritic_activation = maximal_dendritic_activation.view(new_shape)

        return super().forward(x * nn.functional.sigmoid(maximal_dendritic_activation))

class ActiveLIFNode(neuron.LIFNode):

    def __init__(self, sgn_dims, n_dendrites):
        super().__init__()

        self.sgn_dims = sgn_dims
        self.active_dendrites = nn.Parameter(torch.randn(sgn_dims, n_dendrites))
    
    def forward(self, x, context):

        dendritic_activations = torch.matmul(context, self.active_dendrites)
        maximal_dendritic_activation = torch.max(dendritic_activations, dim = -1).values

        x_shape = x.shape
        new_shape = x_shape[:1] + (1,) * (len(x_shape) - 1)
        maximal_dendritic_activation = maximal_dendritic_activation.view(new_shape)

        return super().forward(x) * nn.functional.sigmoid(maximal_dendritic_activation)

class NonSpikingBaseNode(nn.Module, base.MultiStepModule):

    def __init__(self):

        super().__init__()
    
    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):

        raise NotImplementedError

    def forward(self, x_seq: torch.Tensor):

        self.v = torch.full_like(x_seq.data, fill_value=0.0)
        v_seq = []
        self.neuronal_charge(x_seq)
        v_seq.append(self.v)
        
        return v_seq[-1]

class NonSpikingIFNode(NonSpikingBaseNode):

    def __init__(self):
        super().__init__()

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x