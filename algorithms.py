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


def Bagging(model,X,Y):
  
    all_acc = []                                            # List to store the accuracies of the 10 classifiers
    all_predictions = []                                # this list will store all the predictions of all 10 classifier

    for i in range(1,11):
        start = random.randint(0,len(X)/2)        # random starting point
        end = random.randint(len(X)/2,len(X))    # random ending point
        train_X = X.iloc[start:end,:]                 # training data for this iteration 'i'
        train_Y = Y.iloc[start:end,:]
        model.fit(train_X,train_Y)
        y_pred = model.predict(X_test)
        indiv_acc = accuracy_score(y_pred,Y_test)           # accuracy obtained for one of the 10 prediction model
        all_acc.append(indiv_acc)                                     # storing all the 10 accuracies
        all_predictions.append(y_pred)

    n = len(all_predictions[0])
    m = len(all_predictions)
    modes = []                                # the list will store the most occuring prediction out of the 10 predictions for each datapoint

    for i in range(n):
        x = []
        for j in range(m):
            x.append(all_predictions[j][i])
        max = mode(x)
        modes.append(max)
    acc = accuracy_score(modes,Y_test)              # comparing the Y_test and the predicted output
    
    return all_acc,acc
  
#___________________________________________________________#


class KMeans:

    def __init__(self, k, max_iter):
        self.k = k                                                 
        self.max_iter = max_iter
        self.labels = None
        self.class_wise_centroids = None          # Cluster centroids for all k clusters
        self.clusters = np.empty((k,), dtype = np.ndarray)
                                    
          
    def train(self, dataset, initial_centroids):

      if initial_centroids == None:
            random_init_class_centroids = []

            for i in range(self.k):
                indx = np.random.randint(0,len(dataset))
                random_init_class_centroids.append(dataset[i])

            self.class_wise_centroids = random_init_class_centroids
            class_centroids = self.class_wise_centroids

      else:
          self.class_wise_centroids = initial_centroids
          class_centroids = initial_centroids       
      
      for iter in range(self.max_iter):

            predictions = []                                                      # this will store all the datapoints that have got classified in a particular class at that index - if a point gets classified as class 0, it will be stored as a part of a list at predictions[0]
            for i in range(self.k):
                predictions.append([])

            for p in dataset:
                dist_from_centroids = []                                

                for c in range(len(class_centroids)):
                    sum = 0
                    centroid = class_centroids[c]
                    for j in range(len(p)):
                        diff = p[j] - centroid[j]
                        sum+= (diff*diff)
                    d = sum**0.5
                    dist_from_centroids.append(d)

                if(len(dist_from_centroids)!=0):
                    closest_centroid = dist_from_centroids.index(min(dist_from_centroids))
                    predictions[closest_centroid].append(p)

            new_centroids = []
            for i in range(self.k):

                centroid = []
                class_predictions = predictions[i]

                for col in range(len(class_predictions[0])):
                    avg_for_col = 0
                    sum_for_col = 0
                    for row in range(len(class_predictions)):
                        sum_for_col += class_predictions[row][col]
                    avg_for_col = sum_for_col/len(class_predictions)
                    centroid.append(avg_for_col)

                new_centroids.append(centroid)

            class_centroids = new_centroids
            self.class_wise_centroids = new_centroids       

      return class_centroids,predictions

  #________________________________________________________________#
  
  
def PCA(X,N):

      cov_mat = covariance_matrix(X)

      eigen_vectors = np.linalg.eigh(cov_mat)[1]
      evec = eigen_vectors
      eigen_values = np.linalg.eigh(cov_mat)[0]
      ev = eigen_values

      sorted_index = np.argsort(ev)[::-1]
      sorted_eigenvalue = ev[sorted_index]
      sorted_eigenvectors = evec[:,sorted_index]
      sorted_titles = X.columns.values[sorted_index]

      eigenvector_subset = sorted_eigenvectors[:,0:N]
      titles_subset = sorted_titles[0:N]

      X_reduced = np.dot(eigenvector_subset.transpose() , X.transpose()).transpose()
      X_reduced = pd.DataFrame(X_reduced, columns = titles_subset)

      return X_reduced,evec,ev

    
  def PCA_Gaussian(X,N):

      mat = covariance_matrix(X)
      cov_mat = mat.copy()
      for i in range(len(cov_mat)):
          for j in range(len(cov_mat[0])):
              if i!=j:
                  cov_mat[i][j] = 0

      eigen_vectors = np.linalg.eigh(cov_mat)[1]
      evec = eigen_vectors
      eigen_values = np.linalg.eigh(cov_mat)[0]
      ev = eigen_values

      sorted_index = np.argsort(ev)[::-1]
      sorted_eigenvalue = ev[sorted_index]
      sorted_eigenvectors = evec[:,sorted_index]
      sorted_titles = X.columns.values[sorted_index]

      eigenvector_subset = sorted_eigenvectors[:,0:N]
      titles_subset = sorted_titles[0:N]

      X_reduced = np.dot(eigenvector_subset.transpose() , X.transpose()).transpose()
      X_reduced = pd.DataFrame(X_reduced, columns = titles_subset)

      return X_reduced,evec,ev

    #_____________________________________#
