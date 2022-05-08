import numpy as np
from copy import deepcopy
import gc

class NeuralNetwork:

    # expect list of numbers following sizes of each layer
    def __init__(self, sizes_of_layers):
        self.sizes_of_layers = sizes_of_layers
        self.number_of_layers = len(sizes_of_layers)
        self.biases = [np.random.rand(y, 1)*2-1 for y in sizes_of_layers[1:]]
        self.weights = [np.random.rand(y, x)*2-1 for x, y in zip(sizes_of_layers[:-1], sizes_of_layers[1:])]
        self.learning_rate = 1
        self.deltas_weights = []
        self.deltas_biases = []
        self.activation_values = []
        self.errors = []

    # feeds input forward, returns guess output and save activations
    def feed_forward(self, input_vector):
        self.activation_values = []
        if type(input_vector) != np.ndarray:
            input_vector = np.matrix(input_vector)
        input_vector = input_vector.reshape(self.sizes_of_layers[0], 1)
        for i in range(len(self.weights)):
            self.activation_values.append(deepcopy(input_vector))
            input_vector = sigmoid(np.dot(self.weights[i], input_vector) + self.biases[i])
        input_vector = np.power(input_vector, 3)
        #input_vector /= sum(input_vector)
        self.activation_values.append(deepcopy(input_vector))

        return input_vector

    # used for validating non-labeled input
    def guess(self, input_vector):
        output = self.feed_forward(input_vector)
        return output

    def calculate_errors(self, input_vector, target_output):
        target_output = np.matrix(target_output).reshape(self.sizes_of_layers[-1], 1)
        guess_output = self.feed_forward(input_vector)
        self.errors.append(target_output - guess_output)
        transposed_weights = [np.transpose(matrix) for matrix in self.weights]
        for weight in reversed(transposed_weights[1:]):
            self.errors.append(np.dot(weight, self.errors[-1]))
        self.errors.reverse()

    def calculate_deltas_weights(self, input_vector, target_output):
        self.calculate_errors(input_vector, target_output)
        while len(self.errors) > 0:
            gradient = dsigmoid(self.activation_values[-1])
            gradient = np.multiply(gradient, self.errors[-1])
            gradient *= self.learning_rate
            self.deltas_biases.append(gradient)
            delta = np.transpose(self.activation_values[-2])
            delta = np.dot(gradient, delta)
            self.deltas_weights.append(delta)
            self.activation_values.pop(-1)
            self.errors.pop(-1)
        self.deltas_biases.reverse()
        self.deltas_weights.reverse()

    def train(self, inputs, targets):
        for input, target in zip(inputs, targets):
            self.deltas_weights = []
            self.deltas_biases = []
            self.errors = []
            self.activation_values = []
            self.calculate_deltas_weights(input, target)
            for j in range(self.number_of_layers-1):
                self.weights[j] += self.deltas_weights[j]
                self.biases[j] += self.deltas_biases[j]


def sigmoid(number):
    for x in number:
        if x[0] > 705:
            x[0] = 705
        if x[0] < -705:
            x[0] = -705
    return 1.0 / (1.0 + np.exp(-number))


def dsigmoid(number):
    return np.multiply(number, (1-number))
