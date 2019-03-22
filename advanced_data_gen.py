import numpy as np

#functions: parabola (concave up and down), asymmetric parabola, affine,
#piecewise with continuity breaks
#types of noise: white noise, flicker, brownian noise, functional noise
#input data distributions: uniform, gaussian, exponential, skewed

#need to alter the following params:
#N - number of data samples in a dataset
#M - number of dimensions in the dataset
#same basis function or piecewise?
#continuity breaks - model this using a distribution
#functions: sin, asymmetric sine, affine
#data distribution - uniform, gaussian, exponential, piecewise

N = [i for i in range(1,10000)]
M = [i for i in range(1,3)]
f = ['sin', 'line', 'assym']

