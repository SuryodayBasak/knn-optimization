import numpy as np
import matplotlib.pyplot as plt
from data_gen import CosineData
from knn import KNNRegressor, DKNNRegressor, ExpKNNRegressor, AxKNNRegressor, BoxKNNRegressor
from knn import WeightedExpKNNRegressor, WeightedDKNNRegressor, AxWeightedDKNNRegressor, BoxWeightedDKNNRegressor
np.set_printoptions(precision=2)

data_obj = CosineData(0, np.pi/2 + 0.5,
            [0, np.pi/8, np.pi/4+0.1, np.pi/2+0.2, np.pi/2 + 1.5],
            [0.13, 0.3, 0.2, 0.3, 0.1], 0.0)

#data_obj = CosineData(0, np.pi/2 + 0.5,
#            [0,  np.pi/2,  np.pi],
#            [0.3, 0.4, 0.3], 0.0)

data_obj.generate()
X,y = data_obj.get_rn()
#plt.scatter(X, y)

test_data_x = np.linspace(0.01, np.pi/2 + 1.4, 100)
test_data_y = np.cos(test_data_x)

plt.plot(test_data_x, test_data_y)

#Method 1
reg = KNNRegressor(X, y, 5)
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred, label='KNN Regressor')

#Method 2
reg = DKNNRegressor(X, y, 5)
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred, label='Distance KNN Regressor')

"""
#Method 3
reg = ExpKNNRegressor(X, y, 5)
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred, label='Exp KNN Regressor')

#Method 4
reg = WeightedExpKNNRegressor(X, y, 5, 10, 100, 10)
reg.train()
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred, label='Weighted Exp KNN Regressor')
"""

#Method 5
reg = WeightedDKNNRegressor(X, y, 5, 10, 100, 10)
reg.train()
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred, label='Weighted DKNN Regressor')

weights = reg.get_weights()
#plt.scatter(X, y, s=np.multiply(weights,1000))
print(weights)

#for i in range(len(X)):
#    plt.Circle((0.5, 0.5), 0.2, color='blue')

"""
#Method 6
reg = AxWeightedDKNNRegressor(X, y, 5, 10, 100, 10)
reg.train()
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred, label='Axis Bal WDKNN Regressor')
"""

"""
#Method 7
reg = BoxWeightedDKNNRegressor(X, y, 5, 10, 100, 10)
reg.train()
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred, label='Box WDKNN Regressor')
"""

"""
#Method 8
reg = AxKNNRegressor(X, y, 5)
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred, label='Axis Balanced KNN Regressor')
"""

"""
#Method 9
reg = BoxKNNRegressor(X, y, 5)
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred, label='Box KNN Regressor')
"""

plt.legend()
plt.show()
"""
#Method 3
reg = WeightOptimizedKNN(X, y, 1, 0.01)
reg.train()
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred)

plt.show()

#reg.train_predict()
#test_data_pred = reg.predict(test_data_x)
#plt.plot(test_data_x, test_data_pred)
"""
