from bayes_classifier import BayesClassifier
from helpers import confusion_matrix_plot, get_accuracy, plot, plot_two_sets, visualize
import numpy as np
import pandas as pd

def k_cross_validation_test():
    df = pd.read_csv("iris_data.csv")
    iterations = 50
    partitions = [3, 5, 7, 9, 11, 13, 15]

    classifier = BayesClassifier()

    accuracy_acc = []
    for partition in partitions:
        accuracies = []
        for _ in range(iterations):
            data = df.sample(frac=1).reset_index(drop=True)
            X, y = data.iloc[:, :-1], data.iloc[:, -1]
            _, accuracy = classifier.k_cross_validation_training(partition, X, y)
            accuracies.append(np.mean(accuracy))
        accuracy_acc.append(accuracies)
    plot(partitions, accuracy_acc, "liczba k w k-krotnej walidacji krzyowej")

def fracs_test():
    df = pd.read_csv("iris_data.csv")
    iterations = 50
    fracs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    classifier = BayesClassifier()

    data = df.sample(frac=1, random_state=1).reset_index(drop=True)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    accuracy_acc_valid = []
    accuracy_acc_test = []

    for frac in fracs:
        accuracies_valid = []
        accuracies_test = []
        
        for _ in range(iterations):
            data = df.sample(frac=1).reset_index(drop=True)
            X, y = data.iloc[:, :-1], data.iloc[:, -1]
            len = X.shape[0]

            X_train, X_test, y_train, y_test = X[:int(frac*len)], X[int(frac*len):], y[:int(frac*len)], y[int(frac*len):]

            predictions_valid = classifier.simple_trainig(X_train, y_train, X_test)
            predictions_test = classifier.simple_trainig(X_train, y_train, X_train)

            accuracy_valid = get_accuracy(y_test, predictions_valid)
            accuracy_test = get_accuracy(y_train, predictions_test)

            accuracies_valid.append(accuracy_valid)
            accuracies_test.append(accuracy_test)

        accuracy_acc_valid.append(accuracies_valid)
        accuracy_acc_test.append(accuracies_test)

    plot_two_sets(fracs, accuracy_acc_valid, accuracy_acc_test, "point of sets separating")

def single_frac_test():
    df = pd.read_csv("iris_data.csv")
    frac = 0.7
    classifier = BayesClassifier()

    data = df.sample(frac=1, random_state=1).reset_index(drop=True)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    data = df.sample(frac=1).reset_index(drop=True)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    len = X.shape[0]

    X_train, X_test, y_train, y_test = X[:int(frac*len)], X[int(frac*len):], y[:int(frac*len)], y[int(frac*len):]

    predictions_test = classifier.simple_trainig(X_train, y_train, X_test)
    predictions_train = classifier.simple_trainig(X_train, y_train, X_train)

    visualize(y_test, predictions_test, frac, 'Species')
    confusion_matrix_plot(y_test, predictions_test, classifier.classes)
    confusion_matrix_plot(y_train, predictions_train, classifier.classes)