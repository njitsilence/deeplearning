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

def initialize_parameters_deep(layer_dims):
    """

    :param layer_dims: python array(list) containing the dimensons of each layer in our network
    :return:
    parameters: --- python dictionary containing the parameters "W1","b1",...."WL","bl"

    """
    np.random.seed(3)
    parameters={}
    L=len(layer_dims) #number of layers in the network

    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters["b"+str(l)] = np.zeros([layer_dims[l],1])

    assert(parameters["W"+str(l)].shape==(layer_dims[l],layer_dims[l-1]))
    assert(parameters["b" + str(l)].shape == (layer_dims[l],1))

    return parameters

def linear_forward(A,W,b):
    """
    implement the linear part of layer's forward propagation
    Arguments:
    A -- activation from previous layer(or input data ): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer
    b -- bias vector, numpy array of shape(size of the current layer, 1)

    :return:
    Z -- the input of the activation function , also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computiing the backward pass efficiency

    """
    Z=np.dot(W,A)+b
    assert(Z.shape==(W.shape[0],A.shape[1]))
    cache=(A,W,b)

    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    """
    implement the forward propagation for the LINEAR -> ACTIVATION layer
    :param A_prev: activations from previous layer(or input data): (size of previous layer, number of examples)
    :param W: weights matrix: numpy array of shape(size of the current layer,size of previous layer)
    :param b: bias vector, numpy array of shape(size of current layer,1)
    :param activation: the activation to be used in this layer, stored as a text string :"sigmoid" or "relu"
    :return:
    A -- the output of the activation function, also called the post-activation value
    cache - a python dictionary containing "linear_cache" and "activation_cache"
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A,activation_cache = sigmoid(Z)
    elif activation == "relu":
        A,activation_cache = relu(Z)

    assert(A.shape==(W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)
    return A,cache



