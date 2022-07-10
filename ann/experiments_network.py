from get_data import read_mnist_data, get_train_test_validation, get_house_prediction_data
import numpy as np
from multilayer_perceptron import MultilayerPerceptron
from activation_functions import *
from plot_data import plot_learning_data, plot_confusion_matrix


def experiment_and():
    #-------------------------------INITIALIZE NETWORK PARAMETERS----------------------------------------------------
    input_size = 2
    output_size = 1
    number_of_layers = 2
    neuron_quantities = [3]
    activation_funs = [tanh_act, sigmoid_act]

    number_of_epochs = 40
    number_of_batches = 4
    beta = 0.9
    # --------------------------------GET DATA-----------------------------------------------------------------------
    y = [0,1,0,0] 
    X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    #-------------------------------INITIALIZE AND TRAIN NETWORK---------------------------------------------------
    net_mlp = MultilayerPerceptron(input_size, output_size, number_of_layers, neuron_quantities, activation_funs)
    net_mlp.train_network(X, y, number_of_epochs, number_of_batches, beta)
    #--------------------------------------DISPLAY RESULTS-----------------------------------------------------------
    print(net_mlp.get_multiple_predictions(X))
    print(y)


def experiment_or():
    #-------------------------------INITIALIZE NETWORK PARAMETERS----------------------------------------------------
    input_size = 2
    output_size = 1
    number_of_layers = 2
    neuron_quantities = [3]
    activation_funs = [tanh_act, sigmoid_act]

    number_of_epochs = 40
    number_of_batches = 4
    beta = 0.9
    # --------------------------------GET DATA-----------------------------------------------------------------------
    y = [0,1,1,1]
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    #-------------------------------INITIALIZE AND TRAIN NETWORK---------------------------------------------------
    net_mlp = MultilayerPerceptron(input_size, output_size, number_of_layers, neuron_quantities, activation_funs)
    net_mlp.train_network(X, y, number_of_epochs, number_of_batches, beta)
    #--------------------------------------DISPLAY RESULTS-----------------------------------------------------------
    print(net_mlp.get_multiple_predictions(X))
    print(y)


def experiment_house_price_prediction():
    #-------------------------------INITIALIZE NETWORK PARAMETERS----------------------------------------------------
    input_size = 10
    output_size = 1
    number_of_layers = 3
    neuron_quantities = [32, 32]
    activation_funs = [tanh_act, relu_act, sigmoid_act]

    number_of_epochs = 80
    number_of_batches = 32
    beta = 0.4
    # --------------------------------GET DATA-----------------------------------------------------------------------
    x, y = get_house_prediction_data()
    train_y = y[0:32]
    train_x = x[0:32]

    valid_y = y[33:39]
    valid_x = x[33:39]
    #-------------------------------INITIALIZE AND TRAIN NETWORK---------------------------------------------------
    net_mlp = MultilayerPerceptron(input_size, output_size, number_of_layers, neuron_quantities, activation_funs)
    net_mlp.train_network(train_x, train_y, number_of_epochs, number_of_batches, beta)
    #--------------------------------------DISPLAY RESULTS-----------------------------------------------------------
    print(net_mlp.get_multiple_predictions(valid_x))
    print(valid_y)


def experiment_mnist():
    #-------------------------------INITIALIZE NETWORK PARAMETERS----------------------------------------------------
    input_size = 64
    output_size = 10
    number_of_layers = 3
    neuron_quantities = [10, 10]
    activation_funs = [tanh_act, relu_act, sigmoid_act]

    number_of_epochs = 2
    number_of_batches = 40
    beta = 0.4
    # --------------------------------GET DATA-----------------------------------------------------------------------
    train_X, train_y, test_X, test_y, validation_X, validation_y = get_train_test_validation(0.2, 0.2)

    print("Shape of train data X {}, train data y {}".format(np.shape(train_X), np.shape(train_y)))
    print("Shape of validation data X {}, validation data y {}".format(np.shape(validation_X), np.shape(validation_y)))
    print("Shape of test data X {}, test data y {}".format(np.shape(test_X), np.shape(test_y)))

    #-------------------------------INITIALIZE AND TRAIN NETWORK---------------------------------------------------
    net_mlp = MultilayerPerceptron(input_size, output_size, number_of_layers, neuron_quantities, activation_funs)
    train, val = net_mlp.train_network_get_learning_curve(train_X, train_y, validation_X, validation_y, number_of_epochs, number_of_batches, beta) 

    #--------------------------------------DISPLAY RESULTS-----------------------------------------------------------
    plot_learning_data(train, val, "Funkcja straty - MNIST")

    preds = net_mlp.get_multiple_predictions_numeric(train_X)
    correct = []
    for element in train_y:
        correct.append(np.argmax(element))
    plot_confusion_matrix(preds, correct, "Macierz pomyłek zbioru treningowego MNIST")

    preds = net_mlp.get_multiple_predictions_numeric(validation_X)
    correct = []
    for element in validation_y:
        correct.append(np.argmax(element))
    plot_confusion_matrix(preds, correct, "Macierz pomyłek zbioru walidcyjnego MNIST")

    preds = net_mlp.get_multiple_predictions_numeric(test_X)
    correct = []
    for element in test_y:
        correct.append(np.argmax(element))
    plot_confusion_matrix(preds, correct, "Macierz pomyłek zbioru testowego MNIST")

    #-------------------------------SAVE MODEL------------------------------------------------------------------------
    net_mlp.save_network("models/test2_model.csv")