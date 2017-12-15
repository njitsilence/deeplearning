import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from utils.dnn_app_utils_v2 import *

np.random.seed(1)

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

#index = 10
#plt.imshow(train_x_orig[index])
#plt.show()
#print ("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0],-1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0],-2).T

# Standardize data to have feature values between 0 and 1
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

# Two-Layer neural work
### CONSTANTS DEFINING THE MODEL ###

n_x = 12288  # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x,n_h,n_y)

def two_layer_model(X,Y,layers_dims,learning_rate = 0.0075,num_iterations=3000,print_cost=False):
    """
    Implements a two-layer neural network : LINEAR->RELU-LINEAR->SIGMOID.
    :param X: input data, of shape(n_x,number of examples)
    :param Y: true "label" vector(containing 0 if cat, 1 if non-cat), of shape(1,number of examples)
    :param layers_dim: dimensions of the layers (n_x,n_h,n_y)
    :param learning_rate: learning rate of gradient descent update rate
    :param num_iterations:number of the iterations of the optimization loop 
    :param print_cost: If set to True, this will print the cost every 100 iterations 
    :return: 
    parameters -- a dictionary containing W1,W2,b1 and b2
       
    """

    np.random.seed(1)
    grads = {}
    costs = []     # to keep the track of cost
    m = X.shape[1]
    (n_x,n_h,n_y) = layers_dims

    # Initialize parameters dictionary, by calling one of the functions previously implemented

    parameters = initialize_parameters(n_x,n_h,n_y)

    # Get W1,b1,W2,b2 from the dictionary parameters
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    # Loop (gradient descent)
    for i in range(0,num_iterations):

        # Forward propagation : LINEAR -> RELU -> LINEAR -> SIGMOID, Inputs: X W1 b1 output A1 cache1 A2 cache2
        A1,cache1 = linear_activation_forward(X,W1,b1,'relu')
        A2,cache2 = linear_activation_forward(A1,W2,b2,'sigmoid')

        # Compute cost
        cost = compute_cost(A2,Y)

        # Initializing backward propagation

        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1,dW2,db2 = linear_activation_backward(dA2,cache2,'sigmoid')
        dA0,dW1,db1 = linear_activation_backward(dA1,cache1,'relu')

        # Set the dictionary grads
        grads["dW1"] = dW1
        grads["dW2"] = dW2
        grads["db1"] = db1
        grads["db2"] = db2

        # Update parameters
        parameters = update_parameters(parameters,grads,learning_rate)

        # Retrieve W1,b1,W2,b2 from updated parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i,np.squeeze(cost)))
            costs.append(cost)


    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate ="+str(learning_rate))
    plt.show()

    return parameters


# parameters=two_layer_model(train_x,train_y,layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
# predictions_train = predict(train_x, train_y, parameters)
# predictions_test = predict(test_x, test_y, parameters)



# n_layer_model
def L_layer_model(X,Y,layers_dims,learning_rate=0.0075,num_iterations=3000,print_cost=False): #lr was 0.009
    """
    Implements a L-layer neural network : LINEAR -> RELU *(L-1) -> LINEAR -> SIGMOID
    :param X: data, numpy array of shape (number of examples, num_px * num_px * 3)
    :param Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    :param layers_dims: list containing the input size and each layer size, of length (number of layers + 1)
    :param learning_rate: learning rate of the gradient descent update rule
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: if True, it prints the cost every 100 steps
    :return: 
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []    # keep track of cost

    # Initialize Parameters
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0,num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL,caches = L_model_forward(X,parameters)

        # Compute cost
        cost = compute_cost(AL,Y)

        # Backward propagation:
        grads = L_model_backward(AL,Y,caches)

        # Update parameters
        parameters = update_parameters(parameters,grads,learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

layers_dims = [12288,20,7,5,1] # 5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=100, print_cost=True)
pred_train = predict(train_x, train_y, parameters)
pred_train = predict(test_x, test_y, parameters)


