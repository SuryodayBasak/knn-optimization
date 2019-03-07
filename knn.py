import numpy as np

class KNNRegressor:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k
    
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            nbrs_dist = []

            for i in range(len(self.X)):
                nbrs_dist.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            y = 0.0
            for j in k_idx:
                y += self.y[j]

            y = y/(self.k)
            y_pred.append(y)
        return y_pred

class DKNNRegressor:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            nbrs_dist = []

            for i in range(len(self.X)):
                nbrs_dist.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            y = 0.0
            w = 0.0
            for j in k_idx:
                y += (1/nbrs_dist[j])*self.y[j]
                w += 1/nbrs_dist[j]
            y = y/w
            y_pred.append(y)
        return y_pred

class ExpKNNRegressor:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            nbrs_dist = []

            for i in range(len(self.X)):
                nbrs_dist.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            y = 0.0
            w = 0.0
            for j in k_idx:
                y += np.e**(nbrs_dist[j])*self.y[j]
                w += np.e**(nbrs_dist[j])
            y = y/w
            y_pred.append(y)
        return y_pred

class WeightedExpKNNRegressor:
    def __init__(self, X, y, weights, k):
        self.X = X
        self.y = y
        self.k = k
        self.weights = weights

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            nbrs_dist = []

            for i in range(len(self.X)):
                nbrs_dist.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            y = 0.0
            w = 0.0
            for j in k_idx:
                y += np.e**(-abs(self.weights[j])*nbrs_dist[j])*self.y[j]
                w += np.e**(-abs(self.weights[j])*nbrs_dist[j])
            y = y/w
            y_pred.append(y)
        return y_pred
