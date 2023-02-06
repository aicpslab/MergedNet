# This file contains a class to help generate a merged neural network
# when given a larger network and a smalelr network.
# The output of the merged network is the difference in the
# outputs of the larger and smaller network.

# https://github.com/aicpslab/MergedNet

# 8/19/2022
# Wesley Cooke

# Dependencies
import torch
import torch.nn as nn
import copy as cp
import numpy as np


class GenerateMergedNet:
    """
    Class that contains methods used to generate a merged network. Based on the equations 12 - 27 of https://arxiv.org/pdf/2202.01214.pdf. 
    """

    @staticmethod
    def from_PyTorch_Path(largePath:str, smallPath:str):
        """ Method to get the Merged Network given two .pt or .pth paths."""
        
        if (largePath.endswith(".pt") or largePath.endswith(".pth")) and (smallPath.endswith(".pt") or smallPath.endswith(".pth")): 

            LargeModel = torch.load(largePath)
            SmallModel = torch.load(smallPath)

            LargeModelDict = GenerateMergedNet._extract_pytorch_wandb(LargeModel)
            SmallModelDict = GenerateMergedNet._extract_pytorch_wandb(SmallModel)

            return GenerateMergedNet.from_WandB(LargeModelDict, SmallModelDict)

    @staticmethod 
    def from_PyTorch_Models(largeModel, smallModel):
        """ Method to get the Merged Network given two PyTorch Models """

        LargeModelDict = GenerateMergedNet._extract_pytorch_wandb(largeModel)
        SmallModelDict = GenerateMergedNet._extract_pytorch_wandb(smallModel)

        return GenerateMergedNet.from_WandB(LargeModelDict, SmallModelDict)

    @staticmethod
    def _extract_pytorch_wandb(model):

        weights = []
        biases = []
        weights_shape = []
        biases_shape = []
        actFuncs = []

        # For each parameter in the model
        for name, param in model.named_parameters():
            
            # Extract the param if it is a weight or bias. 
            if name.endswith(".weight"):
                weights.append(cp.deepcopy(param.data.numpy()))
                weights_shape.append(tuple(param.data.shape))
            elif name.endswith(".bias"):
                temp = np.expand_dims(cp.deepcopy(param.data.numpy()), axis=1)
                biases.append(temp)
                biases_shape.append(tuple(biases[-1].shape))

        # Grab the Activation Functions 
        for param in model.modules():
            if isinstance(param, nn.ReLU):
                actFuncs.append("ReLU")
            elif isinstance(param, nn.Sigmoid):
                actFuncs.append("sigmoid")
            elif isinstance(param, nn.Tanh):
                actFuncs.append("tanh")

        model_dict = {
            "w": weights,
            "b": biases,
            "w_shapes": weights_shape,
            "b_shapes": biases_shape,
            "acts": actFuncs
        }

        return model_dict

    @staticmethod 
    def from_TensorFlow(largePath, smallPath):
        # :TODO: Implement generating a merged network
        # from TensorFlow files.

        raise NotImplementedError

    @staticmethod
    def from_onnx(largePath, smallPath):
        # :TODO: Implement generating a merged network
        # from Onnx files.

        raise NotImplementedError

    @staticmethod
    def from_WandB(aLargeModelDict, aSmallModelDict):
        """
        Based on Equations 12 - 27 of https://arxiv.org/pdf/2202.01214.pdf
        """

        # Dictionary must contain weights, biases, activation functions,
        # the shape of the weights, and the shape of the biases.

        # Unpack Dictionaries 
        wLar = aLargeModelDict['w']
        bLar = aLargeModelDict['b']
        wLarShape = aLargeModelDict['w_shapes']
        bLarShape = aLargeModelDict['b_shapes']
        aLar = aLargeModelDict['acts']

        wSmall = aSmallModelDict['w']
        bSmall = aSmallModelDict['b']
        wSmallShape = aSmallModelDict['w_shapes']
        bSmallShape = aSmallModelDict['b_shapes']
        aSmall = aSmallModelDict['acts']

        numLayersLar = len(wLar)
        numLayersSmall = len(wSmall)

        # Merged Model Variables
        wMerged = []
        bMerged = []

        # This version assumes that all the 
        # inner Activation Functions are the same. -> Don't know how to merge two networks that use different activations between layers. 
        # For example: ReLU Functions. 

        aMerged = aLar #Keep the activations of the Large network. 

        # Case 1
        # Layer One, the input layer.
        wMerged.append(np.vstack((wLar[0], wSmall[0]))) # Equation 13
        bMerged.append(np.concatenate((bLar[0], bSmall[0]))) # Equation 14

        # Case 2 
        # Layer 2 - Hidden Layers up to and including the last hidden layer of the smaller network.
        for i in range(1, numLayersSmall-1): # second argument is exclusive. 

            tempWL = np.hstack((wLar[i], np.zeros((wLarShape[i][0], wSmallShape[i][0]))))
            tempWS = np.hstack((np.zeros((bSmallShape[i][0], bLarShape[i][0])), wSmall[i]))

            wMerged.append(np.concatenate((tempWL, tempWS)))
            bMerged.append(np.concatenate((bLar[i], bSmall[i])))

        # Case 3
        # Expanded Layers for the small model
        # If the Small Model and Large Model have the same layers, this will be skipped

        for i in range(numLayersSmall-1, numLayersLar-1):

            tempWL = np.hstack((wLar[i], np.zeros((wLarShape[i][0], wSmallShape[numLayersSmall-2][0]))))
            tempWS = np.hstack((np.zeros((wSmallShape[numLayersSmall-2][0], wLarShape[i][0])), np.eye(wSmallShape[numLayersSmall-2][0]))) # eye is identity matrix

            wMerged.append(np.concatenate((tempWL, tempWS)))
            bMerged.append(np.concatenate((bLar[i], np.expand_dims(np.zeros(wSmallShape[numLayersSmall-2][0]), axis=1))))

        # Case 4
        # Parallel output layer for the Small and Large Model. 

        tempWL = np.hstack((wLar[numLayersLar-1], np.zeros((wLarShape[numLayersLar-1][0], wSmallShape[-1][1]))))
        tempWS = np.hstack((np.zeros((wSmallShape[-1][0], wLarShape[-1][1])), wSmall[numLayersSmall-1]))

        wMerged.append(np.concatenate((tempWL, tempWS)))
        bMerged.append(np.concatenate((bLar[numLayersLar-1], bSmall[numLayersSmall-1])))

        # Case 5
        # Final output layer that will compute the difference of the two outputs
        wMerged.append(np.hstack((np.eye(wLarShape[-1][0]), -np.eye(wLarShape[-1][0]))))

        # This bias Needs to be the shape of the output that we expect
        bMerged.append(np.zeros((wLarShape[-1][0], 1)))

        for index, bias in enumerate(bMerged):
            bMerged[index] = np.squeeze(bias)
        
        if bMerged[-1].shape == ():
            bMerged[-1] = np.array([bMerged[-1]]) # This is needed for a weird interaction between numpy and pytorch. 

        mergedDict = {
            "w": wMerged,
            "b": bMerged,
            "acts": aMerged
        }

        return mergedDict

    @staticmethod
    def truncate_params(ModelDict, num):
        """
        Take the weights and biases and truncate them to num decimal places.

        Returns a new Model dictionary
        """

        q_w = []
        for weight in ModelDict['w']:
            q_w.append(np.around(weight, num).astype(np.float16))

        q_b = []
        for bias in ModelDict['b']:
            q_b.append(np.around(bias, num).astype(np.float16))

        QuantDict = {
            "w": q_w,
            "b": q_b,
            "w_shapes": ModelDict['w_shapes'],
            "b_shapes": ModelDict["b_shapes"],
            "acts": ModelDict["acts"]
        }

        return QuantDict

    @staticmethod
    def generated_sequential_from_dictionary(ModelDict):
        """
            Method to Generate a sequential pytorch model from a model dictionary.
            Works for models that don't use an activation on the last linear layer. 
        """

        w = ModelDict['w']
        w_shapes = ModelDict['w_shapes']
        b = ModelDict['b']
        b_shapes = ModelDict['b_shapes']
        activations = ModelDict['acts']

        seq = nn.Sequential()

        # count the number of linear layers we add.
        num_linear = 0
        index = 0

        # Add each liner that also has an activation function. 
        for layer in activations:

            in_features, out_features = tuple(reversed(w_shapes[index]))

            seq.append(nn.Linear(in_features, out_features))
            num_linear += 1

            if layer == "ReLU":
                seq.append(nn.ReLU())
            else:
                return NotImplementedError
            index += 1

        # Last Layer
        in_features, out_features = tuple(reversed(w_shapes[-1]))
        seq.append(nn.Linear(in_features, out_features)) # Output layers of the two separate networks
        index += 1
        num_linear += 1

        # Has to be true or the model will be wrong
        assert num_linear == len(w)

        # Init weights and biases from merged net
        index = 0
        for layer in seq:
            if isinstance(layer, nn.Linear):
                layer.weight.data = torch.nn.parameter.Parameter(torch.tensor(w[index]))
                layer.bias.data = torch.nn.parameter.Parameter(torch.tensor(b[index]))
                index += 1

        # Return the Sequential model
        return seq

    @staticmethod
    def generate_merged_sequential(ModelDict):
        """ Given a Merged Net Dictionary, return a sequential model pytorch representation. """

        weights = ModelDict['w']
        biases = ModelDict['b']
        activations = ModelDict['acts']

        seq = nn.Sequential()

        # count the number of linear layers we add.
        num_linear = 0
        index = 0

        # Add each layer that has a coresponding activation function. 
        for layer in activations:
            seq.append(nn.Linear(weights[index].shape[-1], weights[index].shape[0]))
            num_linear += 1

            if layer == "ReLU":
                seq.append(nn.ReLU())
            else:
                return NotImplementedError
            index += 1

        # Last two linear layers for mergeed network
        seq.append(nn.Linear(weights[index].shape[-1], weights[index].shape[0])) # Output layers of the two separate networks
        index += 1
        num_linear += 1

        seq.append(nn.Linear(weights[index].shape[-1], weights[index].shape[0])) # Output of the merged network: IE The difference in the two separate networks
        num_linear += 1

        # Has to be true or the model will be wrong
        assert num_linear == len(weights)

        # Init weights and biases from merged net
        index = 0
        for layer in seq:
            if isinstance(layer, nn.Linear):
                layer.weight.data = torch.nn.parameter.Parameter(torch.tensor(weights[index]))
                layer.bias.data = torch.nn.parameter.Parameter(torch.tensor(biases[index]))
                #print(biases[index])
                index += 1

        # Return the Sequential model
        return seq

    @staticmethod
    def simulate_merged_relu_network(ModelDict, inputs):
        """
            Generate Output data for a merged model dictionary.
            The last two layers are just linear.

            The activation function used inbetween layers aside from the 
            last two linear layers is relu. 
        """
        outputs = []

        for input in inputs:
            #input = [input]
            #print(input)

            # Concurent Layers
            for weight, bias in zip(ModelDict["w"][:-2], ModelDict['b'][:-2]):

                input = np.matmul(weight, input) + bias
                input = np.maximum(0, input)

            # Output Layer of individal networks
            weight = ModelDict['w'][-2]
            input = np.matmul(weight, input) + ModelDict['b'][-2]

            # Output Layer of Merged Network
            weight = ModelDict['w'][-1]
            output = np.matmul(weight, input) + ModelDict['b'][-1]
            outputs.append(output)

        return outputs
    
    @staticmethod 
    def print_shapes(ModelDict):
        """ Method to check the shapes of the Merged Network Model Dictionary """

        weights = ModelDict['w']
        biases = ModelDict['b']

        print("Weights Shape:")
        for weight in weights:
            print(weight.shape)
        
        print("Bias Shape:")
        for bias in biases:
            print(bias.shape)
    
    @staticmethod
    def print_params(ModelDict):
        """ Method to check the weights and biases of the Merged Network Model Dictonary """

        weights = ModelDict['w']
        biases = ModelDict['b']

        print("Weights:")
        for weight in weights:
            print(weights)
        
        print("Biases")
        for bias in biases:
            print(bias)
