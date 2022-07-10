import numpy as np
from math import sqrt

class Neuron:
    def __init__(self, input_size, activation_fun):
        self.in_size = input_size
        self.weights = self.initialize_weights()
        self.bias = self.initialize_bias()
        self.activation_function = activation_fun
        self.output_before_activation = 0
        self.output = 0

    def initialize_weights(self):
        """
        draw random weights from uniform distribution
        """
        min_value = -1/sqrt(self.in_size)
        max_value = 1/sqrt(self.in_size)

        return np.random.uniform(min_value, max_value, self.in_size)
    
    def initialize_bias(self):
        """
        draw random bias from uniform distribution
        """
        min_value = -1/sqrt(self.in_size)
        max_value = 1/sqrt(self.in_size)

        return np.random.uniform(min_value, max_value)
    
    def calculate_output(self, inputs):
        """
        calculate output of a neuron for a given vector of input values
        """
        if len(inputs) != self.in_size:
            print("Wrong input size for a given neuron!\n")

        weighted_input = np.multiply(inputs, self.weights)
        sum_of_weighted = np.sum(weighted_input)
        sum_of_weighted_bias = sum_of_weighted + self.bias
        
        variables = sorted(self.activation_function().free_symbols, key = lambda symbol: symbol.name)
        out = self.activation_function().evalf(subs = dict(zip(variables,np.array([sum_of_weighted_bias]))))

        self.output_before_activation = sum_of_weighted_bias
        self.output = out

        return out

    def calculate_sum_of_inputs(self, inputs):
        """
        calculate sum of inputs, without activation function
        """
        weighted_input = np.multiply(inputs, self.weights)
        sum_of_weighted = np.sum(weighted_input)

        return sum_of_weighted

    def update_weights(self, weight_diff, beta):
        """
        change weights of a neuron according input vector of differences and beta parameter
        """
        if len(weight_diff) != self.in_size:
            print("Wrong input size for a given neuron!\n")

        gradient = np.multiply(weight_diff, -beta)
        new_weights = np.add(self.weights, gradient)
        self.weights = new_weights

    def update_bias(self, bias_diff, beta):
        """
        change bias of a neuron according input difference value and beta parameter
        """
        gradient = -bias_diff*beta
        self.bias = self.bias + gradient

