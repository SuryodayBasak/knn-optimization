import numpy as np
import matplotlib.pyplot as plt
from data_gen import CosineData

data_obj = CosineData(0, np.pi/2 + 0.5,
            [0, np.pi/8, np.pi/4+0.1, np.pi/2+0.2, np.pi/2 + 1.5],
            [0.08, 0.08, 0.08, 0.08, 0.1])
data_obj.generate()
X,y = data_obj.get_rn()
plt.scatter(X, y)
plt.show()
