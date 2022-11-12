# This File will contains a class to help generate a merged neural network
# when given a larger network and a smalelr network, or two networks with 
# the same number of layers. 
# The output layer of the merged network is the difference in the
# outputs of the two networks it is composed of. 

# 8/19/2022
# Wesley Cooke
# Augusta University 

# Dependencies
import torch
import torch.nn as nn
import numpy as np
import copy as cp


class GenerateMergedNet:
    """
    Class that contains methods used to generate a merged network. 
    Based on the equations 12 - 27 of https://arxiv.org/pdf/2202.01214.pdf. 
    """

    @staticmethod
    def from_PyTorch_Path(largePath:str, smallPath:str):
        """ 
        Method to get a dictionary containing the weights, biases, 
        and activation functions of the merged network.

        Typical Usage:
            MergedDictionary = GenerateMergedNet.from_PyTorch_Path(path1, path2)
        """
        
        if (largePath.endswith(".pt") or largePath.endswith(".pth") and 
            smallPath.endswith(".pt") or smallPath.endswith(".pth")): 

            LargeModel = torch.load(largePath)
            SmallModel = torch.load(smallPath)

            LargeModelDict = GenerateMergedNet._extract_pytorch_wandb(LargeModel)
            SmallModelDict = GenerateMergedNet._extract_pytorch_wandb(SmallModel)

            return GenerateMergedNet.from_WandB(LargeModelDict, SmallModelDict)

    @staticmethod 
    def from_PyTorch_Models(largeModel, smallModel):
        """
        Method to get the Merged Network given two PyTorch Models 
        This function can be used in place of "from_Pytorch_Path" if you 
        have already loaded your models in the main program. 

        Typical Usage:
            MergedDictionary = GenerateMergedNet.from_PyTorch_Models(LargeModel, SmallModel)
        """

        LargeModelDict = GenerateMergedNet._extract_pytorch_wandb(largeModel)
        SmallModelDict = GenerateMergedNet._extract_pytorch_wandb(smallModel)

        return GenerateMergedNet.from_WandB(LargeModelDict, SmallModelDict)

    @staticmethod
    def _extract_pytorch_wandb(model):
        """
        Function to extract the weights and biases of a pytorch network. 
        Used internally in "from_PyTorch_Models" and "from_PyTorch_Path".

        Typical Usage:
            aModelDictionary = GenerateMergedNet._extract_pytorch_wandb(aModel)
        
        The returned dicitonary contains list of the following items:
            "w": weights,
            "b": biases, 
            "w_shapes": weights_shape,
            "b_shapes": biases_shape,
            "acts": actFuncs, 
        """

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

        # Basic idea is to convert Tensorflow to pytorch
        # and work with pytorch representation. 

        raise NotImplementedError

    @staticmethod
    def from_onnx(largePath, smallPath):
        # :TODO: Implement generating a merged network
        # from Onnx files.

        # Basic idea is to convert onnx to pytorch
        # and work with pytorch representation

        raise NotImplementedError

    @staticmethod
    def from_WandB(aLargeModelDict, aSmallModelDict):
        """
        Based on Equations 12 - 27 of https://arxiv.org/pdf/2202.01214.pdf

        Returns a dictionary containg a list of the following items: 
        mergedDictionary = {
            "w": wMerged,
            "b": bMerged,
            "acts": aMerged
        }

        Typical Usage:
            LargeModelDict = GenerateMergedNet._extract_pytorch_wandb(LargeModel)
            SmallModelDict = GenerateMergedNet._extract_pytorch_wandb(SmallModel)
            MergedModelDict = GenreateMergedNet.from_WandB(LargeModelDict, SmallModelDict)
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
        # inner Activation Functions are the same. -> Don't know how to merge 
        # two networks that use different activations between layers. 

        aMerged = aLar #Keep the activations of the Large network. 

        # Case 1
        # Layer One, the input layer.
        wMerged.append(np.vstack((wLar[0], wSmall[0]))) # Equation 13
        bMerged.append(np.concatenate((bLar[0], bSmall[0]))) # Equation 14

        # Case 2 
        # Layer 2 - Hidden Layers up to and including the last hidden layer of the smaller network.
        for i in range(1, numLayersSmall-1): # second argument is exclusive. 

            tempWL = np.hstack((wLar[i], 
                                np.zeros((wLarShape[i][0], 
                                          wSmallShape[i][0]))))
                                          
            tempWS = np.hstack((np.zeros((bSmallShape[i][0], 
                                          bLarShape[i][0])), 
                                wSmall[i]))

            wMerged.append(np.concatenate((tempWL, tempWS)))
            bMerged.append(np.concatenate((bLar[i], bSmall[i])))

        # Case 3
        # Expanded Layers for the small model
        # If the Small Model and Large Model have the same layers, this will be skipped

        for i in range(numLayersSmall-1, numLayersLar-1):

            tempWL = np.hstack((wLar[i], 
                                np.zeros((wLarShape[i][0], 
                                          wSmallShape[numLayersSmall-2][0]))))

            tempWS = np.hstack((np.zeros((wSmallShape[numLayersSmall-2][0], 
                                          wLarShape[i][0])), 
                                np.eye(wSmallShape[numLayersSmall-2][0]))) # eye is identity matrix

            wMerged.append(np.concatenate((tempWL, tempWS)))
            bMerged.append(np.concatenate((bLar[i], 
                           np.expand_dims(np.zeros(wSmallShape[numLayersSmall-2][0]),
                                          axis=1))))

        # Case 4
        # Parallel output layer for the Small and Large Model. 

        tempWL = np.hstack((wLar[numLayersLar-1], 
                            np.zeros((wLarShape[numLayersLar-1][0], 
                                      wSmallShape[-1][1]))))

        tempWS = np.hstack((np.zeros((wSmallShape[-1][0], 
                                      wLarShape[-1][1])), 
                                      wSmall[numLayersSmall-1]))

        wMerged.append(np.concatenate((tempWL, tempWS)))
        bMerged.append(np.concatenate((bLar[numLayersLar-1], bSmall[numLayersSmall-1])))

        # Case 5
        # Final output layer that will compute the difference of the two outputs
        wMerged.append(np.hstack((np.eye(wLarShape[-1][0]), -np.eye(wLarShape[-1][0]))))

        # This bias Needs to be the shape of the output that we expect
        bMerged.append(np.zeros((wLarShape[-1][0], 1)))

        for index, bias in enumerate(bMerged):
            bMerged[index] = np.squeeze(bias)
        
        # This is needed if both the networks had 1 dimensional outputs. 
        if bMerged[-1].shape == (): 
            bMerged[-1] = np.array([bMerged[-1]])  

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

        Typical Usage: 
            aModelDict = GenerateMergedNet._extract_pytorch_wandb(aModel)
            aQuantDict = GenerateMergedNet.truncate_params(aModelDict, 4) # truncate to 4 decimal places. 
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
            "w_shapes": ModelDict["w_shapes"],
            "b_shapes": ModelDict["b_shapes"],
            "acts": ModelDict["acts"]
        }

        return QuantDict

    @staticmethod
    def generate_merged_sequential(ModelDict):
        """ 
        Given a Merged Net Dictionary, 
        return a sequential pytorch model representation. 

        Typical Usage: 
            LargeModelDict = GenerateMergedNet._extract_pytorch_wandb(LargeModel)
            SmallModelDict = GenerateMergedNet._extract_pytorch_wandb(SmallModel)
            MergedModelDict = GenreateMergedNet.from_WandB(LargeModelDict, SmallModelDict)
            SeqPytorchModel = GenerateMergedNet.generate_merged_sequetnail(MergedModelDict)
        """
        
        # Unpack the Merged Model Dictionary
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
            elif layer == "tanh":
                seq.append(nn.Tanh())
            elif layer == "sigmoid":
                seq.append(nn.Sigmoid())
            else:
                raise Exception(f"The activation function '{layer}'  has not been implemented"
                    "in the creation of a sequential merged model.")
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

            Note this method was used extensively for debugging rather 
            than practical application. 
            
            It is easier to just call "generate_merged_sequential" and 
            run inferences on the model using pytorch instead of doing this
            numpy implementation. 
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
        """ 
        Method to check the shapes of the Merged Network Model Dictionary 
        Used for debugging. 

        Typical Usage:
            ModelDictionary = GenerateMergedNet._extract_pytorch_wandb(aModel)
            GenerateMergedNet.print_shapes(ModelDictionary)

            or 

            LargeModelDict = GenerateMergedNet._extract_pytorch_wandb(LargeModel)
            SmallModelDict = GenerateMergedNet._extract_pytorch_wandb(SmallModel)
            MergedModelDict = GenreateMergedNet.from_WandB(LargeModelDict, SmallModelDict)
            GenerateMergedNet.print_shapes(MergedModelDict)
        """

        weights = ModelDict['w']
        biases = ModelDict['b']

        print("\nWeights Shape:")
        for weight in weights:
            print(weight.shape)
        
        print("Bias Shape:")
        for bias in biases:
            print(bias.shape)
    
    @staticmethod
    def print_params(ModelDict):
        """ 
        Method to check the weights and biases of the Merged Network Model Dictonary
        Used for debugging.  

        Typical Usage:
            ModelDictionary = GenerateMergedNet._extract_pytorch_wandb(aModel)
            GenerateMergedNet.print_params(ModelDictionary)

            or 

            LargeModelDict = GenerateMergedNet._extract_pytorch_wandb(LargeModel)
            SmallModelDict = GenerateMergedNet._extract_pytorch_wandb(SmallModel)
            MergedModelDict = GenreateMergedNet.from_WandB(LargeModelDict, SmallModelDict)
            GenerateMergedNet.print_params(MergedModelDict)
        """

        weights = ModelDict['w']
        biases = ModelDict['b']

        print("Weights:")
        for weight in weights:
            print(weights)
        
        print("Biases")
        for bias in biases:
            print(bias)
