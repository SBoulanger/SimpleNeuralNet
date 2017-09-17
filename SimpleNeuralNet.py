
import numpy as np
import scipy.special
import matplotlib.pyplot
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
    input_nodes  = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.2
    epochs = 2

    #create neural network
    snn = SimpleNeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

    #get data
    training_data_file = open('../MNIST/mnist_train.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    #do training
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            snn.train(inputs, targets)

    #get test data
    test_data_file = open('../MNIST/mnist_test.csv', 'r')
    test_data_list = test_data_file.readlines()
    training_data_file.close()

    scorecard = []
    #do testing
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print("Correct label: " + all_values[0])
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = snn.query(inputs)
        label = np.argmax(outputs)
        print("SNN Answer: " + str(label))
        scorecard.append(int(label == correct_label))

    print("Percent Correct: ", float(np.asarray(scorecard).sum()/float(len(scorecard))))


    #image_array = np.asfarray(all_values[1:]).reshape((28,28))

    #matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation='None')
    #matplotlib.pyplot.show()

    #print(snn.query((np.asfarray(all_values[1:]) / 255.0 * .99) + 0.01 ))
