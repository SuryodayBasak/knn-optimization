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
    def __init__(self, X, y, k, alpha=0.01):
        self.X = X
        self.y = y
        self.k = k

        self.n_samples = np.size(self.X, 0)
        self.neighbor_mat = np.zeros((self.n_samples, self.n_samples))
        self.neighbor_mat_l = np.zeros((self.n_samples, self.n_samples))
        self.neighbor_mat_r = np.zeros((self.n_samples, self.n_samples))

        for j in range(len(self.X)):
            x = self.X[j]
            x_l = []
            y_l = []
            x_r = []
            y_r = []
            d_l = []
            d_r = []
            idx_l = []
            idx_r = []

            for i in range(len(self.X)):
                if (i != j):
                    if (self.X[i] < x):
                        x_l.append(self.X[i])
                        y_l.append(self.y[i])
                        d_l.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist
                        idx_l.append(i)

                    else:
                        x_r.append(self.X[i])
                        y_r.append(self.y[i])
                        d_r.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist
                        idx_r.append(i)

            sorted_l_idx = np.argsort(d_l)
            sorted_r_idx = np.argsort(d_r)

            l_idx = sorted_l_idx[:self.k]
            r_idx = sorted_r_idx[:self.k]

            y = 0.0
            w_sum = 0.0

            for i in l_idx:
                self.neighbor_mat[j, idx_l[i]] = 1.0
                self.neighbor_mat_l[j, idx_l[i]] = 1.0

            for i in r_idx:
                self.neighbor_mat[j, idx_r[i]] = 1.0
                self.neighbor_mat_r[j, idx_r[i]] = 1.0

            #lam = np.random.standard_normal((n_samples, 2))
            self.lam = np.zeros((self.n_samples, 2))

        print(self.neighbor_mat)
        print()
        print(self.neighbor_mat_l)
        print()
        print(self.neighbor_mat_r)
        print()
        print(self.lam)
        print()

