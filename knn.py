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

############################################################################

def influence(lam, x_c, x):
    inf = lam * (np.e**(-lam*(x-x_c))) 
    return inf

def mean_sq_err(y, y_hat):
    mse = (1/len(y))*sum((y_hat[i] - y[i])**2 for i in range(len(y)))
    return mse

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

            for i in l_idx:
                self.neighbor_mat[j, idx_l[i]] = 1.0
                self.neighbor_mat_l[j, idx_l[i]] = 1.0

            for i in r_idx:
                self.neighbor_mat[j, idx_r[i]] = 1.0
                self.neighbor_mat_r[j, idx_r[i]] = 1.0

            self.l_lam = np.random.sample(self.n_samples)
            self.r_lam = np.random.sample(self.n_samples)
            #self.lam = np.zeros((self.n_samples, 2))
            self.l_offsets = np.random.sample(self.n_samples)
            self.r_offsets = np.random.sample(self.n_samples)

        print(self.neighbor_mat)
        print()
        print(self.neighbor_mat_l)
        print()
        print(self.neighbor_mat_r)
        print()
        print(self.l_lam)
        print()
        print(self.r_lam)
        print()
        print("Example of influence:")

    def train(self):
        #Initialize and populate all the variables required for the optimization
        l_vals = np.zeros((self.n_samples, self.n_samples))
        r_vals = np.zeros((self.n_samples, self.n_samples))

        for i in range(self.n_samples):
            for j in range(self.n_samples):
                l_vals[i,j] = influence(self.l_lam[j], self.X[j], self.X[i])
                r_vals[i,j] = influence(self.r_lam[j], self.X[j], self.X[i])

        #multiply l_vals with self.neighbor_mat_r because the right influences must optimize left distributions and vice versa
        l_vals = np.multiply(l_vals, self.neighbor_mat_r)
        r_vals = np.multiply(r_vals, self.neighbor_mat_l)
        
        #print("LEFT VALUES:")
        #print(l_vals)
        #print("RIGHT VALUES:")
        #print(r_vals)

        y_hat = np.sum(l_vals+r_vals, axis=0)*(1/(2*self.k))
        print(y_hat)
        print("MSE = ", mean_sq_err(self.y, y_hat))

    def train_predict(self):
        y_hat = []
        for i in range(len(self.X)):
            y = 0.0
            for j in range(len(self.neighbor_mat_l[i])):
                if self.neighbor_mat_l[i,j] == 1.0:
                    #The neighbor falls on the left of test point so use the right distribution.
                    #y = y + influence(self.r_lam[j], self.r_offsets[j], self.X[j], self.X[i])   
                    y = y + influence(self.r_lam[j], self.X[j], self.X[i])   

            for j in range(len(self.neighbor_mat_r[i])):
                if self.neighbor_mat_l[i,j] == 1.0:
                    #The neighbor falls on the right of test point so use the left distribution.
                    #y = y + influence(self.l_lam[j], self.l_offsets[j], self.X[j], self.X[i])
                    y = y + influence(self.l_lam[j], self.X[j], self.X[i])

            y_hat.append(y)

        for i in range(len(self.y)):
            print(self.y[i], y_hat[i])

        mse = mean_sq_err(self.y, y_hat)
        print("The MSE is: ", mse)
