# Under progress

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import math

#_________________________________________________________#

class GaussianNaiveBayes:
  
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_prior = np.zeros(len(self.classes))
        self.mean, self.var = {}, {}
 
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_prior[i] = len(X_c) / len(X)
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
    
    def class_conditionsal(self, x, c):
        mean = self.mean[c]
        var = self.var[c]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def priors(self, c):
        return self.class_prior[c]
    
    def posterior(self, X, c):
        prior = self.priors(c)
        likelihood = np.prod(self.class_conditionals(X, c), axis=1)
        return prior * likelihood
    
    def predict(self, X):
        preds = []
        for x in X:
            class_scores = []
            for i, c in enumerate(self.classes):
                class_score = np.sum(np.log(self.pdf(x, c))) + np.log(self.class_prior[i])
                class_scores.append(class_score)
            
            pred = self.classes[np.argmax(class_scores)]
            preds.append(pred)
        return preds
      
#___________________________________________________________#

