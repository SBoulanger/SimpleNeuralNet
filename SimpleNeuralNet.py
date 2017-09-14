
import numpy as np
import scipy.special
"""
Very simple three layer neural network 
"""
class SimpleNeuralNet:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningrate):
        self.inodes  = inputNodes
        self.hnodes = hiddenNodes
        self.onNodes = outputNodes

        self.ihW = np.random.normal(0.0,pow(inputNodes,-0.5),(hiddenNodes,inputNodes))
        self.hoW = np.random.normal(0.0,pow(hiddenNodes,-0.5),(outputNodes,hiddenNodes))

        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list,targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets= np.array(targets_list, ndmin=2).T

        # calculate signals to hidden layer
        input_hidden = np.dot(self.ihW, inputs)
        # run hidden values through activation function
        hidden_outputs = self.activation_function(input_hidden)
        # calculate signals to output layer
        input_final = np.dot(self.hoW, hidden_outputs)
        # run final layer values through activation function
        final_outputs = self.activation_function(input_final)

        #calcute error
        output_errors = targets - final_outputs
        #calculate hidden layer error ( layer n+1 error split by weights and recombined)
        hidden_errors = np.dot(self.hoW.T, output_errors)

        #update weights for links between hidden layer and output layer
        self.hoW += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        #udpate weights for links between input layer and hidden layer
        self.ihW += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))


        pass

    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals to hidden layer
        input_hidden = np.dot(self.ihW, inputs)
        # run hidden values through activation function
        hidden_outputs = self.activation_function(input_hidden)
        # calculate signals to output layer
        input_final = np.dot(self.hoW, hidden_outputs)
        # run final layer values through activation function
        final_outputs = self.activation_function(input_final)
        return final_outputs

if __name__ == '__main__':
    input_nodes  = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    snn = SimpleNeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print(snn.query([1.0, 0.5, -1.5]))