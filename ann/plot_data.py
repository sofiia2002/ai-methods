from matplotlib import legend
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_confusion_matrix as plot_CM
from sklearn.metrics import confusion_matrix


def plot_learning_data(train_data, validation_data, plot_title):
    x = np.linspace(0, len(train_data), len(train_data))
    fig, ax = plt.subplots()
    ax.set_title(plot_title)
    ax.set_xlabel('numer epoki')
    ax.set_ylabel('wartość')
    ax.plot(x, train_data, color="firebrick", label="zestaw treningowy", linewidth=2)
    ax.plot(x, validation_data, color="turquoise", label="zestaw walidacyjny", linewidth=2)
    ax.legend(loc="best")
    plt.savefig("out_plots/" + plot_title + ".png")
    plt.show()
    plt.clf()

def plot_confusion_matrix(predicted_values, correct_values, plot_title):
    conf_matrix = confusion_matrix(correct_values, predicted_values)
    fig, ax = plot_CM(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Przewidywania', fontsize=11)
    plt.ylabel('Prawdziwe', fontsize=11)
    plt.title(plot_title, fontsize=18)
    plt.savefig("out_plots/" + plot_title + ".png")
    plt.show()
    plt.clf()
