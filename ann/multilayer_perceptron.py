from numpy import number
import numbers
import random
import numpy as np
from neuron import Neuron
from sgd import calc_derivative, get_derivative
from time import time
from csv import writer


class MultilayerPerceptron:
    def __init__(self, input_size:int, output_size:int, number_of_layers:int, number_of_neurons_in_layers:list, activ_funs:list):
        self.in_size = input_size
        self.out_size = output_size
        self.n_layers = number_of_layers
        self.network = self.initialize_perceptron(number_of_neurons_in_layers, activ_funs)

    def initialize_perceptron(self, number_of_neurons_in_layers, acvivations):
        """
        function that initializes all of the layers and all of the neurons
        """
        network = []
        num_neurons = number_of_neurons_in_layers + [self.out_size]
        n_inputs_to_neuron = self.in_size
        for i in range(0, self.n_layers):
            layer = []
            for j in range(0, num_neurons[i]):
                neuron = Neuron(n_inputs_to_neuron, acvivations[i])
                layer.append(neuron)
            n_inputs_to_neuron = num_neurons[i]
            network.append(layer)

        return network
    
    def get_output_of_layer(self, input_to_layer, layer_number):
        """
        calculate output of a single layer, given its input.
        """
        outputs = []
        for neuron in self.network[layer_number]:
            neuron_out = neuron.calculate_output(input_to_layer)
            outputs.append(neuron_out)

        return np.array(outputs)

    def get_prediction(self, sample):
        """
        get single prediction from a sample
        """
        input_to_layer = sample
        for i in range(0, len(self.network)):
            input_to_layer = self.get_output_of_layer(input_to_layer, i)
            
        return input_to_layer

    def get_multiple_predictions(self, sample_list):
        """
        Returns predictions for a list of input samples
        """
        predictions_list = []
        for sample in sample_list:
            prediction = self.get_prediction(sample)
            predictions_list.append(prediction)
        return np.array(predictions_list)

    def get_multiple_predictions_numeric(self, sample_list):
        """
        calculates predictions in a form of numeric values instead of one-hot encoded vector
        """
        predictions_list = []
        for sample in sample_list:
            prediction = self.get_prediction(sample)
            predictions_list.append(np.argmax(prediction))
        return np.array(predictions_list)

    def train_network(self, X_data, y_data, n_epochs, n_batches, beta):
        """
        Basic verssion of train funcion.
        """
        training_data_size = len(X_data)
        batch_size = int(np.floor(training_data_size/n_batches))

        for i in range(0, n_epochs):
            print("Epoch %d of %d " % (i+1, n_epochs))
            temp = list(zip(X_data, y_data))
            random.shuffle(temp)
            X_data, y_data = zip(*temp)

            for j in range(0, n_batches):
                print("#", end = '')
                if j == n_batches-1:
                    X_batch = X_data[j*batch_size:training_data_size]
                    y_batch = y_data[j*batch_size:training_data_size]
                else:
                    X_batch = X_data[j*batch_size:(j+1)*batch_size]
                    y_batch = y_data[j*batch_size:(j+1)*batch_size]
                self.backpropagate_error(X_batch, y_batch, beta)
            print("")

    def train_network_get_learning_curve(self, X_data, y_data, X_valid, y_valid, n_epochs, n_batches, beta):
        """
        Version of training function used to gain extra insights into learning process.
        Returns data that can be later used to plot learning curve of training process.
        """
        training_data_size = len(X_data)
        batch_size = int(np.floor(training_data_size/n_batches))

        epoch_data_train = []
        epoch_data_test = []

        train_quality = self.calculate_loss_function(y_data, np.reshape(self.get_multiple_predictions(X_data), np.shape(y_data)))
        validation_quality = self.calculate_loss_function(y_valid, np.reshape(self.get_multiple_predictions(X_valid), np.shape(y_valid)))
        epoch_data_train.append(np.nanmean(train_quality, dtype='float32'))
        epoch_data_test.append(np.nanmean(validation_quality, dtype='float32'))

        for i in range(0, n_epochs):
            time_start = time()
            print("Epoch %d of %d " % (i+1, n_epochs))
            temp = list(zip(X_data, y_data))
            random.shuffle(temp)
            X_data, y_data = zip(*temp)
            
            for j in range(0, n_batches):
                print("#", end = '')
                if j == n_batches-1:
                    X_batch = X_data[j*batch_size:training_data_size]
                    y_batch = y_data[j*batch_size:training_data_size]
                else:
                    X_batch = X_data[j*batch_size:(j+1)*batch_size]
                    y_batch = y_data[j*batch_size:(j+1)*batch_size]
                current_batch_size = len(X_batch)
                self.backpropagate_error(X_batch, y_batch, beta)
            print("    time taken: ", time()-time_start)

            train_quality = self.calculate_loss_function(y_data, np.reshape(self.get_multiple_predictions(X_data), np.shape(y_data)))
            validation_quality = self.calculate_loss_function(y_valid, np.reshape(self.get_multiple_predictions(X_valid), np.shape(y_valid)))
            epoch_data_train.append(np.nanmean(train_quality, dtype='float32'))
            epoch_data_test.append(np.nanmean(validation_quality, dtype='float32'))
            print("")
    
        return epoch_data_train, epoch_data_test
            

    def backpropagate_error(self, X, y, beta):
        """
        For input batch (or entire dataset) calculate a weights and bias updates for each layer. 
        Calculated differences are then applied with an input beta factor to entire network.
        """
        y_pred = self.get_multiple_predictions(X)
        error_vec = self.calculate_loss_function_differential(y, y_pred)
        weight_diffs = []
        bias_diffs = []

        for k in range(0, len(X)):
            weight_diffs.append([])
            bias_diffs.append([])
            x = X[k]
            error = error_vec[k]

            curr_layer = self.network[len(self.network)-1]
            input_layer = self.network[len(self.network)-2]
            delta, b_diff, w_diff = self.get_weight_diff(input_layer, curr_layer, deltas=None, error_calc=error, is_output=True)
            weight_diffs[k].append(w_diff)
            bias_diffs[k].append(b_diff)

            for i in range(len(self.network)-2, 0, -1):
                curr_layer = self.network[i]
                input_layer = self.network[i-1]
                delta, b_diff, w_diff = self.get_weight_diff(input_layer, curr_layer, deltas=delta, error_calc=error, is_output=False)
                weight_diffs[k].append(w_diff)
                bias_diffs[k].append(b_diff)

            delta, b_diff, w_diff = self.get_weight_diff(x, self.network[0], deltas=delta, error_calc=error, is_output=False)
            weight_diffs[k].append(w_diff)
            bias_diffs[k].append(b_diff)
        for layer_id in range(0, len(weight_diffs[0])):
            for neuron_id in range(0, len(weight_diffs[0][layer_id])):
                bias_for_neuron = [bias_diffs[sample_id][layer_id][neuron_id] for sample_id in range(0, len(bias_diffs))]
                weights_for_neuron = [weight_diffs[sample_id][layer_id][neuron_id] for sample_id in range(0, len(weight_diffs))]
                weights_for_neuron_trans = np.matrix.transpose(np.array(weights_for_neuron))
                bias_diff_for_neuron = np.nanmean(bias_for_neuron, dtype='float32')
                weights_diff_for_neuron = [np.nanmean(weights_for_neuron_trans[p], dtype='float32') for p in range(0, len(weights_for_neuron_trans))] 
                self.network[self.n_layers - 1 - layer_id][neuron_id].update_bias(bias_diff_for_neuron, beta)
                self.network[self.n_layers - 1 - layer_id][neuron_id].update_weights(weights_diff_for_neuron, beta)
            

    def get_weight_diff(self, input_layer, curr_layer, deltas=None, error_calc=None, is_output=False):
        """
        Get weight shift for the whole layer of ANN, including output layer
        """
        weight_diff = []
        new_deltas = []
        old_deltas = []

        if not isinstance(input_layer[0], numbers.Number):
            if is_output:
                for k in range(0, len(curr_layer)): 
                    single_delta = calc_derivative(curr_layer[k].activation_function, curr_layer[k].output_before_activation)*error_calc[k]
                    weight_diff.append([single_delta*input_layer[i].output for i in range(0, len(input_layer))])
                    old_deltas.append(single_delta)
            else:
                old_deltas = deltas
                for k in range(0, len(curr_layer)): 
                    weight_diff.append([old_deltas[k]*input_layer[i].output for i in range(0, len(input_layer))])

            for j in range(0, len(input_layer)): 
                single_derivative = calc_derivative(input_layer[j].activation_function, input_layer[j].output_before_activation)
                sum_of_deltas = 0
                for k in range(0, len(curr_layer)): 
                    sum_of_deltas+=old_deltas[k]*curr_layer[k].weights[j]
                new_deltas.append(single_derivative*sum_of_deltas)
        else: 
            old_deltas = deltas
            for k in range(0, len(curr_layer)): 
                weight_diff.append([old_deltas[k]*input_layer[i] for i in range(0, len(input_layer))])
        
        return new_deltas, old_deltas, weight_diff
            
    def calculate_loss_function(self, correct_output, predicted_output):
        """
        loss = (y_pred - y_correct)^2
        """
        return np.power(np.add(np.array(predicted_output), -np.array(correct_output)), 2)
        
    def calculate_loss_function_differential(self, correct_output, predicted_output):
        """
        diff_loss = 2 * (y_pred - y_correct)
        """
        p_np = np.array(predicted_output)
        c_np = np.array(correct_output)
        return np.add(p_np, -c_np)

    def save_network(self, file_name):
        """
        Save contents of network in a CSV file. Weights for each layer are saved in a single row.
        """
        for layer in self.network:
            layer_data = []
            for neuron in layer:
                w = neuron.weights.tolist()
                layer_data = layer_data + w
            append_list_as_row(file_name, layer_data)

        
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


def flatten_list(t):
    return [item for sublist in t for item in sublist]