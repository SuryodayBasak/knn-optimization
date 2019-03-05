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
                    x_r.append(self.X[i])
                    y_r.append(self.y[i])
                    d_r.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist

            sorted_l_idx = np.argsort(d_l)
            sorted_r_idx = np.argsort(d_r)

            l_idx = sorted_l_idx[:self.k]
            r_idx = sorted_r_idx[:self.k]

            y = 0.0
            for j in l_idx:
                y += y_l[j]

            for j in r_idx:
                y += y_r[j]

            y = y/(self.k*2)
            y_pred.append(y)
        return y_pred

class InvDistDirectionBalancedKNN:
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
                    x_r.append(self.X[i])
                    y_r.append(self.y[i])
                    d_r.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist

            sorted_l_idx = np.argsort(d_l)
            sorted_r_idx = np.argsort(d_r)

            l_idx = sorted_l_idx[:self.k]
            r_idx = sorted_r_idx[:self.k]

            y = 0.0
            w_sum = 0.0

            for j in l_idx:
                y += (1/d_l[j])*y_l[j]
                w_sum += (1/d_l[j])

            for j in r_idx:
                y += (1/d_r[j])*y_r[j]
                w_sum += (1/d_r[j])

            y = y/(w_sum)
            y_pred.append(y)
        return y_pred

class WeightOptimizedKNN:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def train(self):
        for i in range(len(self.X)):
            print(self.X[i])
