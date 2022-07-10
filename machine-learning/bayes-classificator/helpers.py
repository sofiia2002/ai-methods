from matplotlib.transforms import Affine2D
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
    

def get_accuracy(testY, predY):
    accuracy = np.sum(testY == predY) / len(testY)
    return accuracy

def visualize(y_true, y_pred, alpha, target):
    true = pd.DataFrame(data=y_true, columns=[target])
    predicted = pd.DataFrame(data=y_pred, columns=[target])
    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(15,6))
        
    sns.countplot(x=target, data=true, ax=ax[0], palette='viridis', alpha=alpha, hue=target, dodge=False)
    sns.countplot(x=target, data=predicted, ax=ax[1], palette='viridis', alpha=alpha, hue=target, dodge=False)

    ax[0].tick_params(labelsize=12)
    ax[1].tick_params(labelsize=12)
    ax[0].set_title("True classes")
    ax[1].set_title("Predicted classes")
    plt.show()

def confusion_matrix_plot(y_true, y_pred, target):
    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(cf_matrix, annot=True, xticklabels=target, yticklabels=target)
    plt.xlabel("True class")
    plt.ylabel("Predicted class") 
    plt.show()

def plot(x_data, y_data, x_label):
    _, ax = plt.subplots()
    y = [np.mean(y_data[i]) for i in range(len(y_data))]
    yerr = [np.var(y_data[i]) for i in range(len(y_data))]
    _ = ax.errorbar(x_data, y, yerr=yerr, marker="o",  elinewidth=1, capsize=5) 
    plt.xlabel(x_label)
    plt.ylabel("Accuracy") 
    plt.show()

def plot_two_sets(x_data, y_data1, y_data2, x_label):
    _, ax = plt.subplots()
    y1 = [np.mean(y_data1[i]) for i in range(len(y_data1))]
    yerr1 = [np.var(y_data1[i]) for i in range(len(y_data1))]
    y2 = [np.mean(y_data2[i]) for i in range(len(y_data2))]
    yerr2 = [np.var(y_data2[i]) for i in range(len(y_data2))]
    trans1 = Affine2D().translate(-0.02, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.02, 0.0) + ax.transData
    _ = ax.errorbar(x_data, y1, yerr=yerr1, marker="o", transform=trans1, elinewidth=1, capsize=5, label="validation set") 
    _ = ax.errorbar(x_data, y2, yerr=yerr2, marker="o",  transform=trans2, elinewidth=1, capsize=5, label="test set") 
    ax.legend()
    plt.xlabel(x_label)
    plt.ylabel("Accuracy") 
    plt.show()