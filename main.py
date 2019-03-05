import numpy as np
import matplotlib.pyplot as plt
from data_gen import CosineData
from knn import DirectionBalancedKNN, InvDistDirectionBalancedKNN, WeightOptimizedKNN

data_obj = CosineData(0, np.pi/2 + 0.5,
            [0, np.pi/8, np.pi/4+0.1, np.pi/2+0.2, np.pi/2 + 1.5],
            [0.08, 0.2, 0.2, 0.2, 0.1], 0.0)
data_obj.generate()
X,y = data_obj.get_rn()
plt.scatter(X, y)

test_data_x = np.linspace(0.01, np.pi/2 + 1.4, 100)
test_data_y = np.cos(test_data_x)

plt.plot(test_data_x, test_data_y)

#Method 1
reg = DirectionBalancedKNN(X, y, 2)
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred)

#Method 2
reg = InvDistDirectionBalancedKNN(X, y, 2)
test_data_pred = reg.predict(test_data_x)
plt.plot(test_data_x, test_data_pred)

plt.show()

#Method 3
#reg = DirectionBalancedKNN(X, y, 2)
#test_data_pred = reg.predict(test_data_x)
#plt.plot(test_data_x, test_data_pred)

