import numpy as np
import scipy

def white(x, y, std, a):
    y_noisy = []
    for i in range(len(x)):
        noise_i = np.random.normal(0, std)
        y_noisy.append(y[i] + noise_i)
    return y_noisy

def pink(x, y, std, a):
    y_noisy = []
    for i in range(len(x)):
        noise_i = np.random.normal(0, std/x[i])
        y_noisy.append(y[i] + noise_i)
    return y_noisy

def red(x, y, std, a):
    y_noisy = []
    for i in range(len(x)):
        noise_i = np.random.normal(0, std/(x[i]**2))
        y_noisy.append(y[i] + noise_i)
    return y_noisy

def blue(x, y, std, a):
    y_noisy = []
    for i in range(len(x)):
        noise_i = np.random.normal(0, std*x[i])
        y_noisy.append(y[i] + noise_i)
    return y_noisy

def violet(x, y, std, a):
    y_noisy = []
    for i in range(len(x)):
        noise_i = np.random.normal(0, std*(x[i]**2))
        y_noisy.append(y[i] + noise_i)
    return y_noisy
