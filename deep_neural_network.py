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

def L_model_forward(X,parameters):
    """
    implement forward propagation for the [linear->relu]*(L-1)->linear-sigmoid computation
    :param X: -- data, numpy array of shape(input size,number of examples)
    :param parameters: output of initialize_parameters_deep()
    :return:
    AL -- last post-activation value
    caches -- list of the caches containing:
                every cache of linear_relu_forward()(there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward()(there is one ,indexed L-1)
    """
    caches = []
    A = X
    L = len(parameters)//2 # number for layers in the neural network
    # implement [linear->relu]*(L-1). add cache to the "caches" list
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches.append(cache)

    # implement [linear->sigmoid]. add cachee to the "caches" list
    AL,cache =linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)

    assert(AL.shape==(1,X.shape[1]))
    return AL,caches
"""
X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))
"""

def compute_cost(AL,Y):
    """
    implement the cost function
    :param AL: -- probability vector corresponding to your label predictions, shape(1,number of examples)
    :param Y: -- true "label" vector (for example : containing 0 if non-cat), shape(1,number of examples)
    :return:
    cost --cross-entropy cost
    """
    m=Y.shape[1]

    cost = -np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m

    cost=np.squeeze(cost)

    assert(cost.shape==())

    return cost

"""
Y, AL = compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))
"""

def linear_backward(dZ,cache):
    """
    Implement the linear portion of backward propagation for a single layer(layer l)
    :param dZ: -- Gradient of the cost with respect to linear outputs(of current layer l)
    :param cache: -- tuple of values(A_prev,W,b) come from the forward propagation in the current layer
    :return:
    dA_prev -- Gradient of the cost with respect to activation (of the previous layer l-1), same shape as A_prev
    dW --  Gradient of the cost with respect to W (current layer l), same shape as W
    db --  Gradient of the cost with respect to b (current layer l), same shape as b

    """

    A_prev,W,b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ, axis=1).reshape(b.shape) / m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

"""
dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
"""

def linear_activation_backward(dA,cache,activation):
    """
    Implement the backward propagation for the LINEAR-ACTIVATION layer
    :param dA: post-activation gradient for current laryer
    :param cache: tuple of values(linear_cache, activation_cache) we store for computing backward propagation efficiently
    :param activation: the activation to be used in this layer, sotred as a text string "sigmoid" or "relu"
    :return:
    dA_prev: -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW: -- Gradient of the cost with respect to W(current layer l), same shape as W
    db: -- Gradient of the cost with respect to b (current layer l), same shape as b

    """

    linear_cache,activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)

    return dA_prev, dW, db

"""
AL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
"""

def L_model_backward(AL,Y,caches):
    """
    Implement the backward propagation for the Linear->RELU * (L-1) -> Linear --> SIGMOID group
    :param AL: probability vector, output of the forward propagation (L_model_forward())
    :param Y: true "label" vector
    :param caches: list of caches containing:
                every cache of linear_activation_forward() with "relu"(it's caches[l],for l in range(L-1) i.e l = 0....[L-2])
                the cache of linear_activation_forward() with "sigmoid"(it's caches[L-1])
    :return:

    grads -- a dictionary with gradients
                grads["dA"+str(l)] = ...
                grads["dW"+str(l)] = ...
                grads["db"+str(l)] = ...
    """
    grads={}
    L = len(caches)  #the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    #initializing the backpropagation
    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,"sigmoid")

    for l in reversed(range(L-1)):
        # lth layer:(RELU->LINEAR) gradients
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads["dA"+str(l+2)],current_cache,"relu")
        grads["dA"+str(l+1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

"""
AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print_grads(grads)
"""

def update_parameters(parameters,grads,learning_rate):
    """
    update parameters using gradient descent
    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradients, output of the L_model_backward
    :param learning_rate:
    :return:
    parametes -- python dictionary containing your updated parameters
    """
    L = len(parameters) // 2 #number of layers in the neural network

    #update rule for each parameters, use a for loop
    for l in range(1,L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters

"""

parameters, grads = update_parameters_test_case()

parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))
"""

