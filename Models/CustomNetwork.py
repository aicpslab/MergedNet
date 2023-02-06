import torch.nn as nn

class PytorchNetwork(nn.Module):
    """ 
        This class is used to generate a pytorch network
        with a specific number of input dimenions and 
        specific number of output dimensions 
    """

    def __init__(self, input_neurons, output_neurons, hidden_neurons, num_hidden_layers):
        """
            input_neurons: The dimensions of the inputs to the network.
            output_neurons: The dimensions of the outputs of the network.
            hidden_neurons: The number of neurons in the hidden layers.
            num_hidden_layers: How many hidden layers for the network?
        """

        super(PytorchNetwork, self).__init__()
        
        self.layers = nn.Sequential()

        # Input layer
        self.layers.append(nn.Linear(input_neurons , hidden_neurons))
        self.layers.append(nn.ReLU())
        
        # Hidden Layers
        for x in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            self.layers.append(nn.ReLU())

        # Output Layer
        self.layers.append(nn.Linear(hidden_neurons, output_neurons))

        #self.apply(self._init_weights)

    def _init_weights(self, module):
        """ This method will normalize the wieghts around 0. """
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            module.bias.data.normal_(mean=0.0, std=1.0)

    def forward(self, x):
        x = self.layers(x)
        return x
