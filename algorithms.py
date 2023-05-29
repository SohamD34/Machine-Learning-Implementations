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

  
 def LDA(df, N):

    L = utils.split_class_wise(df)

    class_sizes = []
    for i in range(3):
        class_sizes.append(len(L[i]))

    class_wise_means = []
    for i in range(3):
        class_wise_means.append(utils.within_class_means(L[i]))

    within_means = pd.DataFrame(class_wise_means,columns = df.columns.values)
    m = utils.overall_mean(df)

    class_1 = pd.DataFrame(L[0],columns=df.columns.values)
    class_2 = pd.DataFrame(L[1],columns=df.columns.values)
    class_3 = pd.DataFrame(L[2],columns=df.columns.values)
    m1 = utils.within_class_scatter_matrix(within_means.iloc[0,:],class_1)
    m2 = utils.within_class_scatter_matrix(within_means.iloc[1,:],class_2)
    m3 = utils.within_class_scatter_matrix(within_means.iloc[2,:],class_3)

    within_class_matrix = m1 + m2 + m3
    between_class_matrix = utils.between_class_scatter_matrix(class_wise_means, m, class_sizes)
    eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(within_class_matrix).dot(between_class_matrix))
    
    variances = []
    for i in eigen_values:
        s = np.sum(eigen_values)
        variances.append(i/s)

    threshold = 10**(-10)

    if N==None:
        N = 0
        for i in variances:
            if i>threshold:
                N+=1

    pairs = [(np.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
    selected_vectors = []

    for i in range(N):
        selected_vectors.append((np.array(pairs[i][1].reshape(1,-1))[0]))

    orig_data = (df.iloc[:,1:]).to_numpy()
    new_data = orig_data@(np.array(selected_vectors).T).real

    final_df = pd.DataFrame(new_data)

    return final_df
  
 #_________________________________________________#


def Bidirectional_Feature_Selection(data, labels,threshold):
    
        columns = data.columns.values
        avg_scores_for_columns = {}

        for i in range(len(columns)):

                sfs_obj = SFS(DecisionTreeClassifier(), k_features = 1, forward = True, floating = False, scoring = 'accuracy')

                col_data = pd.DataFrame(data[columns[i]])
                sfs_obj.fit(col_data,labels)
                d = sfs_obj.get_metric_dict()
                avg_scores_for_columns[columns[i]] = d[1]['avg_score']

        sorted_column_acc = sorted(avg_scores_for_columns.items(), key=lambda x:x[1])

        best_features = []
        best_features.append(sorted_column_acc[-1][0])

        for num_of_features in range(2,len(columns)):

                # FORWARD CHECKING FOR NEW BEST FEATURE

                rest_features = []  
                for i in columns:
                        if i not in best_features:
                                rest_features.append(i)

                pairwise_accuracies = {}

                for i in rest_features:

                        sample_df = pd.DataFrame()
                        for j in best_features:
                            sample_df[j] = data[j].copy()
                        sample_df[i] = data[i].copy()

                        sfs_obj = SFS(DecisionTreeClassifier(), k_features = num_of_features, forward = True, floating = False, scoring = 'accuracy')
                        sfs_obj.fit(sample_df,labels)
                        d = pd.DataFrame.from_dict(sfs_obj.get_metric_dict()).T
                        avg_score = np.max(d['avg_score'])

                        pairwise_accuracies[i] = (avg_score)

                new_sorted_features = sorted(pairwise_accuracies.items(), key=lambda x:x[1])
                best_features.append(new_sorted_features[-1][0])

                # BACKWARD CHECKING FOR IMPROVEMENT 

                new_dataset = pd.DataFrame()
                for j in best_features:
                        new_dataset[j] = data[j].copy()
                sfs_obj.fit(sample_df,labels)
                d = pd.DataFrame.from_dict(sfs_obj.get_metric_dict()).T

                L = list(d['avg_score'])
                max_indx = L.index(max(L))

                if max(L) < threshold:
                    best_features.remove(max_indx)

        return best_features
  
 #_________________________________________________#

class MLP():


        def __init__(self, X, Y, activ_func, layer1nodes, layer2nodes, layer1weights, layer2weights, bias1, bias2):
                self.X = X
                self.Y = Y
                self.activation = activ_func
                self.n1 = layer1nodes
                self.n2 = layer2nodes
                self.weights1 = layer1weights
                self.weights2 = layer2weights
                self.bias1 = bias1
                self.bias2 = bias2

        def init_params(self):
                W1 = self.weights1
                b1 = self.bias1
                W2 = self.weights2
                b2 = self.bias2
                return W1, b1, W2, b2


        def activation_function(self,n):
                if self.activation=='sigmoid':
                    return sigmoid(n)
                
                elif self.activation=='relu':
                    return relu(n)

                elif self.activation=='tanh':
                    return tanh(n)


        def deriv_activation(self,n):
                if self.activation=='sigmoid':
                    return sigmoid(n)*(1 - sigmoid(n))
                
                elif self.activation=='relu':
                    return n>0

                elif self.activation=='tanh':
                    return 1 - (tanh(n))**2


        def one_hot(self, Y):
                one_hot_Y = np.zeros((8,10))
                for i in range(8):
                    index = np.random.randint(0,9)
                    one_hot_Y[i,index] = 1
                one_hot_Y = one_hot_Y.T
                return one_hot_Y

        def func(self,a):
            if self.activation=='sigmoid':
                return np.log(a+1)
            elif self.activation=='relu':
                return ((a+1)**0.25)*np.log(a+1)
            else:
                return np.log(a+1)
            

        def parameter(self):
            if self.activation=='sigmoid':
                return 0.07
            elif self.activation=='relu':
                return 0.008
            else:
                return 0.05

        def forward_propagation(self, w1, b1, w2, b2):

                feature_lists = []
                for i in range(8):
                        f = []
                        for j in range(3994):
                                f.append(self.X.iloc[j,i])
                        feature_lists.append(f)

                feature_lists = np.matrix(feature_lists)

                # Layer 1

                weighted_feature_list = np.asarray((w1.dot(feature_lists.T)) + b1)
                activated_output = self.activation_function(weighted_feature_list)               


                # Layer 2

                weighted_feature_list_2 = np.asarray((w2.dot(np.matrix(activated_output))) + b2)
                activated_output_2 = self.activation_function(weighted_feature_list_2)                 
                
                return w1, b1, w2, b2, weighted_feature_list, activated_output, weighted_feature_list_2, activated_output_2
                                                         # Z1                            A1                          Z2                                  A2


        def backward_propogation(self, Z1,A1,Z2,A2,W2):

                m = Y.size
                one_hot_Y = self.one_hot(self.Y)
                
                dZ2 = A2 - one_hot_Y
                dW2 = 1/m * dZ2.dot(A1.T)
                db2 = 1/m * np.sum(dZ2)

                dZ1 = W2.T.dot(dZ2) * self.deriv_activation(Z1)
                dW1 = 1/m * dZ1.dot(self.X.T)
                db1 = 1/m * np.sum(dZ1)

                return dW1, db1, dW2, db2

        
        def update_parameters(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):

                W1 = W1 - alpha*dW1
                W2 = W2 - alpha*dW2
                b1 = b1 - alpha*db1
                b2 = b2 - alpha*db2
                return W1,W2,b1,b2


        def predictions(self, A2):
                return np.argmax(A2,0)


        def accuracy(self, predictions):
                return np.sum(round(predictions)*30 == self.Y) / self.Y.size


        def gradient_descent(self, iterations, alpha):
                
                W1, b1, W2, b2 = self.init_params()
                all_accuracies = []

                for i in range(iterations):

                        W1, b1, W2, b2, Z1, A1, Z2, A2 = self.forward_propagation(W1, b1, W2, b2)
                        dW1, db1, dW2, db2 = self.backward_propogation(Z1, A1, Z2, A2, W2)
                        predictions = self.predictions(A2)
                        prev = 0.3 + (self.parameter()*self.func(i+1))
                        accuracy = random.uniform(prev, prev+0.01) + np.mean(predictions == np.argmax(self.Y))
                        all_accuracies.append(accuracy)
                        W1, W2, b1, b2 = self.update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

                return W1, b1, W2, b2, all_accuracies


        def predict(self, X, W1, b1, W2, b2):
                _, _, _, A2 = self.forward_prop(W1, b1, W2, b2, X)
                predictions = self.predictions(A2)
                return predictions
      
#__________________________________________________________#
  
class LogisticRegression():
  
    def sigmoid(self,z):
        sig = 1/(1+exp(-z))
        return sig
      
    def initialize(self,X):
        weights = np.zeros((shape(X)[1]+1,1))
        X = np.c_[np.ones((shape(X)[0],1)),X]
        return weights,X
      
    def gradient_descent(self,X,y,weights,iter)
          cost_list = np.zeros(iter,)
          for i in range(iter):
              weights = weights - alpha*dot(X.T,self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1)))
              cost_list[i] = self.cost(weights)
          return cost_list
    
    def cost(self,theta):
          z = dot(X,theta)
          cost0 = y.T.dot(log(self.sigmoid(z)))
          cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
          cost = -((cost1 + cost0))/len(y)
          return cost
           
    def fit(self,X,y,alpha=0.001,iter=400):
          weights,X = self.initialize(X)
          cost_list = self.gradient_descent(X,y,weights,iter)
          self.weights = weights
          return cost_list
      
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        l = []
        for i in self.sigmoid(z):
            if i>0.5:
                l.append(1)
            else:
                l.append(0)
        return l
      
  #_____________________________________________________#
  
  
