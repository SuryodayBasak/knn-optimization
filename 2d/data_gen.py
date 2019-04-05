import numpy as np

def return_y(x, y):
    return np.sin(2*np.pi*x)*np.sin(3.5*np.pi*y) + np.random.normal(0, 0.1)

mean_1 = (1, 1)
cov_1 = [[1, 0.8], [0.8, 1]]
X_1 = np.random.multivariate_normal(mean_1, cov_1, 100)

mean_2 = (0, -1)
cov_2 = [[1, 0.5], [0.5, 1]]
X_2 = np.random.multivariate_normal(mean_2, cov_2, 100)

X = np.concatenate((X_1, X_2), axis=0)
y = []
for sample in X:
    y_ = return_y(sample[0], sample[1])
    y.append(y_)

y = np.array(y)
data = np.insert(X, 2, y, axis=1)

#print(X)
#print(y)
#print(data)

np.savetxt('2-d-data', data, delimiter=',') 
