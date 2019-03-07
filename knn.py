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

###############################################################################

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

###############################################################################

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

###############################################################################

def mean_sq_err(y, y_hat):
    mse = (1/len(y))*sum((y_hat[i] - y[i])**2 for i in range(len(y)))
    return mse

class WeightedExpKNNRegressor:
    def __init__(self, X, y, k, n_init_parents, n_cross, n_best):
        self.X = X
        self.y = y
        self.k = k
        self.n_samples = len(self.X)
        self.n_init_parents = n_init_parents
        self.n_cross = n_cross
        self.n_best = n_best

    def create_initial_population(self):
        self.parents = np.random.rand(self.n_init_parents, self.n_samples)

    def eval_weights(self, weights_mat):
        mse_scores = []
        for parent in weights_mat:
            y_pred = []
            for i in range(self.n_samples):
                nbrs_dist = []
                for j in range(self.n_samples):
                    if i!= j:
                        nbrs_dist.append(np.sqrt((self.X[i] - self.X[j])**2)) #Euclidean dist
                    else:
                        nbrs_dist.append(9999.9)

                sorted_dist_idx = np.argsort(nbrs_dist)
                k_idx = sorted_dist_idx[:self.k]

                y = 0.0
                w = 0.0
                for j in k_idx:
                    y += np.e**(-abs(parent[j])*nbrs_dist[j])*self.y[j]
                    w += np.e**(-abs(parent[j])*nbrs_dist[j])
                y = y/w
                y_pred.append(y)
            mse = mean_sq_err(self.y, y_pred)
            mse_scores.append(mse)
        return mse_scores

    def crossover(self):
        self.children = np.zeros((self.n_cross, self.n_samples))
        n_rows_par = np.size(self.parents, 0)

        #Crossover random parents
        for i in range(self.n_cross):
            idx_1 = idx = np.random.randint(n_rows_par)
            idx_2 = idx = np.random.randint(n_rows_par)

            p1 = self.parents[idx_1]    
            p2 = self.parents[idx_2]
            
            for j in range(self.n_samples):
                self.children[i,j] = ((p1[j] + p2[j])/2) + np.random.normal(0.0, 0.001)

    def select_children(self):
        fitness = self.eval_weights(self.children)
        children_ranks = np.argsort(fitness)
        self.children = self.children[children_ranks[:self.n_best], :] #selecting children
        print("Best MSE:", fitness[children_ranks[0]])

    def train(self):
        self.create_initial_population()

        for i in range(0, 100):
            print("Currently in generation: ", i)
            self.crossover()
            self.select_children()
            self.parents = self.children
            #mse_scores = self.eval_weights(self.children)
        self.weights = self.parents[0]

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

###############################################################################

class WeightedDKNNRegressor():
    def __init__(self, X, y, k, n_init_parents, n_cross, n_best):
        self.X = X
        self.y = y
        self.k = k
        self.n_samples = len(self.X)
        self.n_init_parents = n_init_parents
        self.n_cross = n_cross
        self.n_best = n_best

    def create_initial_population(self):
        self.parents = np.random.rand(self.n_init_parents, self.n_samples)

    def eval_weights(self, weights_mat):
        mse_scores = []
        for parent in weights_mat:
            y_pred = []
            for i in range(self.n_samples):
                nbrs_dist = []
                for j in range(self.n_samples):
                    if i!= j:
                        nbrs_dist.append(np.sqrt((self.X[i] - self.X[j])**2)) #Euclidean dist
                    else:
                        nbrs_dist.append(9999.9)

                sorted_dist_idx = np.argsort(nbrs_dist)
                k_idx = sorted_dist_idx[:self.k]

                y = 0.0
                w = 0.0
                for j in k_idx:
                    y += (parent[j]/nbrs_dist[j])*self.y[j]
                    w += (parent[j]/nbrs_dist[j])
                y = y/w
                y_pred.append(y)
            mse = mean_sq_err(self.y, y_pred)
            mse_scores.append(mse)
        return mse_scores

    def crossover(self):
        self.children = np.zeros((self.n_cross, self.n_samples))
        n_rows_par = np.size(self.parents, 0)

        #Crossover random parents
        for i in range(self.n_cross):
            idx_1 = idx = np.random.randint(n_rows_par)
            idx_2 = idx = np.random.randint(n_rows_par)

            p1 = self.parents[idx_1]    
            p2 = self.parents[idx_2]
            
            for j in range(self.n_samples):
                self.children[i,j] = ((p1[j] + p2[j])/2) + np.random.normal(0.0, 0.001)

    def select_children(self):
        fitness = self.eval_weights(self.children)
        children_ranks = np.argsort(fitness)
        self.children = self.children[children_ranks[:self.n_best], :] #selecting children
        print("Best MSE:", fitness[children_ranks[0]])

    def train(self):
        self.create_initial_population()

        for i in range(0, 100):
            print("Currently in generation: ", i)
            self.crossover()
            self.select_children()
            self.parents = self.children
            #mse_scores = self.eval_weights(self.children)
        self.weights = self.parents[0]

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
                y += (self.weights[j]/nbrs_dist[j])*self.y[j]
                w += (self.weights[j]/nbrs_dist[j])
            y = y/w
            y_pred.append(y)
        return y_pred

###############################################################################

class AxWeightedDKNNRegressor(WeightedDKNNRegressor):
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            nbrs_dist = []
            r_i = 0.0
            l_i = 0.0
            e_i = 0.0
            for i in range(len(self.X)):
                nbrs_dist.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            for j in k_idx:
                if self.X[j] < x:
                    l_i += 1
                elif self.X[j] > x:
                    r_i += 1
                else:
                    e_i += 1

            y = 0.0
            w = 0.0
            for j in k_idx:
                if self.X[j] < x:
                    ax_b = (l_i + r_i) / l_i
                    y += ax_b*(self.weights[j]/nbrs_dist[j])*self.y[j]
                    w += ax_b*(self.weights[j]/nbrs_dist[j])
        
                elif self.X[j] > x:
                    ax_b = (l_i + r_i) / r_i
                    y += ax_b*(self.weights[j]/nbrs_dist[j])*self.y[j]
                    w += ax_b*(self.weights[j]/nbrs_dist[j])

                else:
                    y += (self.weights[j]/nbrs_dist[j])*self.y[j]
                    w += (self.weights[j]/nbrs_dist[j])

            y = y/w
            y_pred.append(y)
        return y_pred

###############################################################################

class BoxWeightedDKNNRegressor(WeightedDKNNRegressor):
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            nbrs_dist = []
            r_i = 0.0
            l_i = 0.0
            e_i = 0.0
            for i in range(len(self.X)):
                nbrs_dist.append(np.sqrt((x-self.X[i])**2)) #Euclidean dist

            sorted_dist_idx = np.argsort(nbrs_dist)
            k_idx = sorted_dist_idx[:self.k]

            y = 0.0
            w = 0.0
            for j in k_idx:
                if self.X[j] < x:
                    y += (self.weights[j]/nbrs_dist[j])*self.y[j]
                    w += (self.weights[j]/nbrs_dist[j])

                elif self.X[j] > x:
                    y += (self.weights[j]/nbrs_dist[j])*self.y[j]
                    w += (self.weights[j]/nbrs_dist[j])

                else:
                    y += 2*(self.weights[j]/nbrs_dist[j])*self.y[j]
                    w += 2*(self.weights[j]/nbrs_dist[j])

            y = y/w
            y_pred.append(y)
        return y_pred
