# A modular PyTorch Model used to test GenerateMerged. 

# 11/12/2022
# Wesley Cooke
# Augusta University 

import torch.nn as nn


class OurModel(nn.Module):

    def __init__(self, input_dimension, output_dimension, num_hidden_neurons, num_hidden_layers):

        super(OurModel, self).__init__()

        self.layers = nn.Sequential()
        
        self.layers.append(nn.Linear(input_dimension, num_hidden_neurons))
        self.layers.append(nn.ReLU())

        for x in range(num_hidden_layers):
            self.layers.append(nn.Linear(num_hidden_neurons, num_hidden_neurons))
            self.layers.append(nn.ReLU())
        
        self.layers.append(nn.Linear(num_hidden_neurons, output_dimension))
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            module.bias.data.normal_(mean=0.0, std=1.0)

    def forward(self, x):
        x = self.layers(x)
        return x