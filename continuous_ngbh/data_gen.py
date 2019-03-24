import numpy as np

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

#Generate data between 1 and 3

#Generate linear data



#Generate quadratic data

