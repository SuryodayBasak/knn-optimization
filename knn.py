import numpy as np

class DirectionBalancedKNN:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k
    
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            x_l = []
            y_l = []
            x_r = []
            y_r = []
            d_l = []
            d_r = []

            for i in range(len(self.X)):
                if (self.X[i] < x):
                    x_l.append(self.X[i])
                    y_l.append(self.y[i])
                    d_l.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist

                else:
                    x_l.append(self.X[i])
                    y_l.append(self.y[i])
                    d_l.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist

            sorted_l_idx = np.argsort(d_l)
            sorted_r_idx = np.argsort(d_r)

            l_idx = sorted_l_idx[:self.k]
            r_idx = sorted_r_idx[:self.k]

            y = 0.0
            for j in l_idx:
                y += y_l[j]

            for j in r_idx:
                y += y_r[j]

            y = y/self.k
            y_pred.append(y)
        return y_pred
