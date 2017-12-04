import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases.testCases_v3 import *
from utils.dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2

np.random.seed(1)

def initialize_parameters(n_x,n_h,n_y):
    '''

    :param n_x: size of the input layer
    :param n_h: size of the hiden layer
    :param n_y: size of the output layer
    :return:
        python dictionary contains parameters:
        W1 -- weight matrix of shape(n_h,n_x)
        b1 -- bais vector of shape(n_h,1)
        W2 -- weight matrix of shape(n_y,n_h)
        b2 -- bais vector of shape(n_y,1)
    '''

    np.random.seed(1)
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros([n_h,1])
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros([n_y,1])

    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1":W1,"W2":W2,"b1":b1,"b2":b2}



