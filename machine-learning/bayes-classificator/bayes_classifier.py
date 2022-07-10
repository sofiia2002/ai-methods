import numpy as np
from helpers import get_accuracy

class BayesClassifier:
    def __init__(self):
        self.mean = 0
        self.var = 0
        self.classes = []

    def get_statistics(self, features, target):
        self.mean = features.groupby(target).apply(np.mean).to_numpy()
        self.var = features.groupby(target).apply(np.var).to_numpy()
    
    def density(self, class_id, x):
        mean = self.mean[class_id]
        var = self.var[class_id]
        prob = np.exp((-1/2)*((x-mean)**2)/(2*var))/np.sqrt(2*np.pi*var)
        return prob
    
    def get_prior(self, features, target_class):
        num = lambda x: len(x)
        self.prior = (features.groupby(target_class).apply(num)/self.rows).to_numpy()
        return self.prior
    
    def get_posterior(self, x):
        posteriors = []
        for i in range(self.count):
            prior = np.log(self.prior[i]) 
            conditional = np.sum(np.log(self.density(i, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def simple_trainig(self, trainX, trainY, testX):
        self.classes = np.unique(trainY)
        self.count = len(self.classes)
        self.feature_nums = trainX.shape[1]
        self.rows = trainX.shape[0]    
        self.get_statistics(trainX, trainY)
        self.get_prior(trainX, trainY)
        results = [self.get_posterior(row) for row in testX.to_numpy()]
        return results

    def partition_sets(self, k, i, setX, setY):
        set_len = len(setX)
        if i == 1:
            validX = setX[:int(set_len/k)]
            validY = setY[:int(set_len/k)]
            trainX = setX[int(set_len/k):] 
            trainY = setY[int(set_len/k):] 
        elif i == k:
            validX = setX[int(set_len - set_len/k):] 
            validY = setY[int(set_len - set_len/k):] 
            trainX = setX[:int(set_len - set_len/k)] 
            trainY = setY[:int(set_len - set_len/k)] 
        else:
            validX = setX[int((i-1)*(set_len/k + 1)):int(i*(set_len/k + 1) - 1)] 
            validY = setY[int((i-1)*(set_len/k + 1)):int(i*(set_len/k + 1) - 1)] 
            trainY = setY[:int((i-1)*(set_len/k + 1))].append(setY[int(i*(set_len/k + 1) - 1):], ignore_index=True)
            trainX = setX[:int((i-1)*(set_len/k + 1))].append(setX[int(i*(set_len/k + 1) - 1):], ignore_index=True)
        return trainX, validX, validY, trainY


    def k_cross_validation_training(self, k, setX, setY):
        results = []
        accuracy = []
        for i in range(1, k+1):
            trainX, validX, validY, trainY = self.partition_sets(k, i, setX, setY)
            self.classes = np.unique(trainY)
            self.count = len(self.classes)
            self.feature_nums = trainX.shape[1]
            self.rows = trainX.shape[0]    
            self.get_statistics(trainX, trainY)
            self.get_prior(trainX, trainY)
            result = [self.get_posterior(row) for row in validX.to_numpy()]
            results.append(result)
            accuracy.append(get_accuracy(validY, result))
        return results, accuracy