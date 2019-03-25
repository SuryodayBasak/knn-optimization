import numpy as np
import noise 

f_type = ['parabola', 'line', 'asymmetric parabola']
n_type = ['white', 'flicker', 'pink', 'red', 'blue', 'violet']
#incorporate grey noise later
data_dist = ['uniform', 'gaussian', 'exponential', 'piecewise']

def linear_func(X):
    m = 1
    c = 1
    y = [m*x + c for x in X]
    return y

def quadratic_func(X):
    y = [-(x-2)**2 + 1 for x in X]
    return y

#We consider the following values of variances
variances = [float(i) for i in range(0, 5)]

#Create a meshgrid of variances
wv, pv, rv, bv, vv = np.meshgrid(variances, variances, variances, variances, variances)
positions = np.vstack([wv.ravel(), pv.ravel(), rv.ravel(), bv.ravel(), vv.ravel()]).T


#Considering uniform distribution of data:
X_train = np.random.uniform(low=0.0, high=1.0, size=1000)
X_test = np.random.uniform(low=0.0, high=1.0, size=200)

#Linear target function:
y_train = linear_func(X_train)
y_test = linear_func(X_test)

for v in positions:
    """
    The noise indexes of v are:
    v[0]: white
    v[1]: pink
    v[2]: red
    v[3]: blue
    v[4]: violet
    """

    """
    Generate training set first.
    """
    white_y = noise.white(X_train, y_train, v[0], 1.0) 
    pink_y = noise.pink(X_train, white_y, v[1], 1.0) 
    del white_y
    red_y = noise.red(X_train, pink_y, v[2], 1.0) 
    del pink_y
    blue_y = noise.blue(X_train, red_y, v[3], 1.0) 
    del red_y
    violet_y = noise.violet(X_train, blue_y, v[4], 1.0) 
    del blue_y
    y_train = violet_y
    del violet_y

    """
    Now generate test data.
    """
    white_y = noise.white(X_test, y_test, v[0], 1.0)
    pink_y = noise.pink(X_train, white_y, v[1], 1.0)
    del white_y
    red_y = noise.red(X_train, pink_y, v[2], 1.0)
    del pink_y
    blue_y = noise.blue(X_train, red_y, v[3], 1.0)
    del red_y
    violet_y = noise.violet(X_train, blue_y, v[4], 1.0)
    del blue_y
    y_test = violet_y
    del violet_y

    
#Considering uniform distribution of data:
#wv, p_vars, r_vars, b_vars, v_vars 

#Generate data between 1 and 3

#Generate linear data



#Generate quadratic data

