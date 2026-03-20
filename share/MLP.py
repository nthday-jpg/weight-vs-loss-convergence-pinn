import torch.nn as nn

class MLP(nn.Module):
    """Multi-layer perceptron for PINN."""
    
    def __init__(self, layers, activation=nn.Tanh(), final_activation=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = activation
        self.final_activation = final_activation
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x
