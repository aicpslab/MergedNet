# A quick test to verify that GenreateMerged is working properly. 

# 11/12/2022
# Wesley Cooke
# Augusta University 

import torch
import numpy as np

from GenerateMerged import GenerateMergedNet
from Model import OurModel


def generate_n_dimensional_inputs(num_inputs, input_dim):
    """ Generates num_inputs samples of data with a shape of input_dim """

    rng = np.random.default_rng()
    values = rng.uniform(0, 1, (num_inputs, input_dim))

    return values

def generate_outputs(model, inputs):
    """ Run infereces on a Pytorch Model and collect the results. """

    outputs = []
    model.double()
    for input in inputs:
        input = torch.tensor(input).double()
        output = model(input)
        outputs.append(output.detach().numpy())
    
    return np.array(outputs)

def test_merged_net(num_samples, max_input_dim, max_output_dim, print_samples):
    """
    Test the merged net on 1 dimensional input up to max_input_dim, and 
    1 ouput up to max_output_dimensions. Set print_samples to false to only 
    see the results of np.allclose(). 
    """

    for y in range(1, max_input_dim+1):
        
        test_inputs = generate_n_dimensional_inputs(num_samples, y,) 

        for x in range(1, max_output_dim+1): # Check y input vs x output dimensions. 
            
            print(f"\n{y} input, {x} output:")

            LargerModel = OurModel(y, x, 10, 8) # y in, x out, 10 hidden neurons, 8 hidden layers
            SmallerModel = OurModel(y, x, 5, 3) # y in, x out, 5 hidden neurons, 3 hidden layers

            MergedDictionary = GenerateMergedNet.from_PyTorch_Models(LargerModel, SmallerModel)
            MergedModel = GenerateMergedNet.generate_merged_sequential(MergedDictionary)

            LargeOutputs = generate_outputs(LargerModel, test_inputs)
            SmallOutputs = generate_outputs(SmallerModel, test_inputs)
            MergedOutputs = generate_outputs(MergedModel, test_inputs)
            
            ExcpectedValues = LargeOutputs-SmallOutputs
            
            if print_samples:
                print(f"\nExpected Vs Merged Outputs")
                for a,b in zip(ExcpectedValues, MergedOutputs):
                    print(f"{a}\t{b}")

            print("\nResults of np.allclose(Expected, Merged): ") 
            print(np.allclose(ExcpectedValues, MergedOutputs))
            
        print("="*79)

if __name__ == '__main__': 

    np.random.seed(2022)
    torch.manual_seed(2022)

    num_samples_to_test = 3
    max_input_dim = 5
    max_output_dim = 5
    print_samples = True

    print("="*79)
    test_merged_net(num_samples_to_test, max_input_dim, max_output_dim, print_samples)
