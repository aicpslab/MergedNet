import numpy as np
import torch

def generate_n_dimensional_inputs(num_values, input_dim):

    """ This Function creates a numpy list with length num_values. 
        Each value is in the range 0-1. 

        Args:
            num_values: int - The number of values required. 
            input_dim: int - The dimension of one value. 
    """
    rng = np.random.default_rng()
    values = rng.uniform(0, 1, (num_values, input_dim))

    return values

def get_seq_outputs(model, inputs):
    """ 
        This function will run each input through a sequential pytroch model
        and collect the results into a list. 
    """
    outputs = []
    model.float()

    for input in inputs:
        input = torch.tensor(input).float()
        output = model(input)
        outputs.append(output.detach().numpy())

    return outputs

def get_max_norm(set):
    """
        This method will compute the max norm given a set of verticies. 
        
        Args:
            Set:list - A list of verticies
    """
    max_dist = None
    
    for x in set:
        dist = np.linalg.norm(x)

        if max_dist is None:
            max_dist = dist
        elif dist > max_dist:
            max_dist = dist

    return max_dist
