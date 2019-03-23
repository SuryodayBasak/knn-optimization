import numpy as np
import scipy

def white(x, y, var):
    noise = np.random.normal(0, var)
    return y + noise

def pink(x, y, var):
    noise = np.random.normal(0, var/x)
    return y + noise

def red(x, y, var):
    noise = np.random.normal(0, var/(x**2))
    return y + noise

def blue(x, y, var):
    noise = np.random.normal(0, var*x)
    return y + noise

def violet(x, y, var):
    noise = np.random.normal(0, var*(x**2))
    return y + noise
