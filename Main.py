# This file will show an example of how to use the 
# merged neural network code to compute the distance 
# between two randomly generated networks. 

# Wesley Cooke 2/3/2023

# Python Packages
import matplotlib.pyplot as plt
import numpy as np
import torch 
import multiprocessing as mp

# Custom Classes 
from Models.CustomNetwork import PytorchNetwork
from GenerateMerged import GenerateMergedNet
from DataUtils import generate_n_dimensional_inputs, get_seq_outputs, get_max_norm

# Veritex imports 
from veritex.utils.plot_poly import plot_polytope2d
from veritex.utils.sfproperty import Property
from veritex.networks.ffnn import FFNN   # -> This file has been modified from original 
from veritex.methods.shared import SharedState
from veritex.methods.worker import Worker

def get_exact_output_sets(SequentialModel, lbs, ubs):
    """
        This method utilizes veritex's parallel processing to get the 
        exact reachable set of a network. 

        Args: 
            SequentialModel: A PyTorch sequential model. 
            lbs: A list of the lower bounds of the inputs to the network
            ubs: A list of the upper bounds of the inputs to the network

        lbs and ubs example for an n dimensional network: 
        lbs = [lb_x, lb_y, ..., lb_n]
        ubs = [ub_x, ub_y, ..., ub_n]
    """

    # Code to get the exact output sets from Veritex. Can be seen in their examples.

    # Veritex only uses the Sequential part of the model.
    # NOTE FFNN has been modified for use with the merged model
    dnn = FFNN(SequentialModel, exact_outputd=True)

    # Define the input set using the lower and upper bounds
    # We don't specify an unsafe input domain.
    property_1 = Property([lbs, ubs], [], set_type='FVIM') # Veritex Property
    dnn.set_property(property_1)

    # Setting up the Parallel Processing Framework
    processes = []
    num_processors = mp.cpu_count()
    shared_state = SharedState(property_1, num_processors)
    one_worker = Worker(dnn)

    # Starting the Parallel Processing
    for index in range(num_processors):
        p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
        processes.append(p)
        p.start()

    # Waiting for Process to finish
    for p in processes:
        p.join()

    # Gathering Output Sets
    outputs = []
    while not shared_state.outputs.empty():
        outputs.append(shared_state.outputs.get())

    # Extract vertices of output reachable sets
    exact_output_sets = [np.dot(item.vertices, item.M.T) + item.b.T for item in outputs]

    return exact_output_sets

def get_max_norm_from_exact_output_sets(exact_output_sets):
    """
        This method will compute the max norm of a set of verticies from the 
        veritex tool. 

        Args: 
            exact_output_sets: 
    """

    max_dist = None

    for a_set in exact_output_sets:
        for a_vertici in a_set: 

            new_dist = np.linalg.norm(a_vertici)
            if max_dist is None:
                max_dist = new_dist
            elif new_dist > max_dist: 
                max_dist = new_dist
    
    return max_dist


if __name__ == "__main__":

    # Settings to play with 
    num_trials = 5              # How many times to run the program 
    input_dim = 2               # Number of input dimensions 
                                # If output dimension is 2, it will plot. 
    output_dim = 2              # Number of output dimensions

    hidden_neurons_large = 10   # The number of neurons in the hidden layers
    hidden_neurons_small = 5

    num_hidden_layers_large = 5 # The number of hidden layers
    num_hidden_layers_small = 1

    num_inputs = 5_000          # Number of inferences to compare the   
                                # exact output set with.  

    for x in range(num_trials):
        
        # Set a seed for reproducibility 
        torch.manual_seed(2002+x)
        np.random.seed(2002+x)

        # Define the lower and upper bounds of the input to the network. 
        # In the case of the 2 dimensional input network, we are limiting 
        # our input to a square of [0,0] -> [1,1].
        # A more general notation: [lower_bound_x, lower_bound_y], [upper_bound_x, upper_bound_y]
        # This can extend to an n dimensional network. [lb_x, lb_y, ..., lb_n], [ub_x, ub_y, ..., ub_n]
        lbs = [0 for x in range(input_dim)]
        ubs = [1 for x in range(input_dim)]
        
        # Create the "larger model" and "smaller model". 
        LargeModel = PytorchNetwork(input_dim, output_dim, hidden_neurons=hidden_neurons_large, num_hidden_layers=num_hidden_layers_large)
        SmallModel = PytorchNetwork(input_dim, output_dim, hidden_neurons=hidden_neurons_small, num_hidden_layers=num_hidden_layers_small)

        # Create the merged network by 
        # combining the large model and the small model
        MergedNetDictionary = GenerateMergedNet.from_PyTorch_Models(LargeModel, SmallModel) 
        
        # Get the sequential part of the merged model
        # (The sequential model is what veritex is expecting)
        SeqMerged = GenerateMergedNet.generate_merged_sequential(MergedNetDictionary)
        inputs = generate_n_dimensional_inputs(num_inputs, input_dim)
        
        # Use Veritex to get the exact output sets. 
        try:
            exact_output_sets = get_exact_output_sets(SeqMerged, lbs, ubs)
        except KeyboardInterrupt: # Ctrl-c when "Hyperplane intersects with the verticies"
            print('Continuing...') 
            continue

        # Compute the difference in two ways: 

        # 1. using inferences. -> Less prefered method
        LargeOutputs = get_seq_outputs(LargeModel.layers, inputs)
        SmallOutputs = get_seq_outputs(SmallModel.layers, inputs)

        bf_max_norm = get_max_norm(np.array(LargeOutputs) - np.array(SmallOutputs))
        print("Max of Large Outputs - Small Outputs: ", bf_max_norm)

        # Using the exact output set -> Method we are researching 
        max_distance = get_max_norm_from_exact_output_sets(exact_output_sets)

        print("Max Norm of Exact Output Set Verticies: ", max_distance)
        print()

        # If the output is 2 diemsional, plot the output set. 
        if output_dim == 2:
            
            # Run some sample inferences to plot ontop of the exact output set
            SeqOutputs = get_seq_outputs(SeqMerged, inputs)

            # Set up plotting
            fig = plt.figure(0)
            ax = fig.add_subplot(111)
            dim0, dim1 = -1, 1
            
            # Use Veritex's plotting function to plot the exact output sets. 
            for vs in exact_output_sets:
                plot_polytope2d(vs, ax, color='g', alpha=1.0, edgecolor='k', linewidth=0.0,zorder=2)
            
            # Plot the trial outputs over the exact output sets to verify that they are actually in the exact set. 
            for vertici in SeqOutputs:
                plt.plot(vertici[0], vertici[1], marker="o", markersize=2, markeredgecolor="cyan", markerfacecolor="cyan")

            plt.show()
        