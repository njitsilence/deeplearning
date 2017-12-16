import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from utils.init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from utils.init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()
#plt.show()

def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization="he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    :param X: input data, of shape(2,number of examples)
    :param Y: true "label" vector (containing 0 for the red dots. 1 for the blue dots,) of shape(1, number of examples)
    :param learning_rate: learning rate for gradient descent 
    :param num_iterations:  number of iterations to run gradient descent
    :param print_cost: if True, print the cost every 1000 iterations
    :param initialization: flag to choose which initialization to use ("zeros","random" or "he")
    :return: 
    parameters  -- parameters learnt by the model
    """
    grads = {}
    costs = [] # to keep track of the loss
    m = X.shape[1] # number of examples
    layers_dims = [X.shape[0],10,5,1]

    # Initialize parameters dictionary

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    # Loop (gradient descent)
    for i in range(0,num_iterations):
        a3,cache = forward_propagation(X,parameters)
        cost = compute_loss(a3,Y)
        grads = backward_propagation(X,Y,cache)
        parameters = update_parameters(parameters,grads,learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i,cost))
            costs.append(cost)

    # plot the loss
    plt.close()
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def initialize_parameters_zeros(layer_dims):
    """
    
    :param layer_dims: python array (list) containing the size of each layer
    :return: 
    parameters -- python dictionary containing parameters W1 b1 ..... WL bL
                    W1 -- weight matrix of shape(layer_dims[1],layer_dims[0])
                    b1 -- bias vector of shape (layer_dims[1],1)
                    ...
                    WL -- weight matrix of shape(layer_dims[L],layer_dims[L-1])    
    """
    parameters = {}
    L = len(layer_dims)  # number of layers in network

    for l in range(1,L):
        parameters['W'+str(l)] = np.zeros([layer_dims[l],layer_dims[l-1]])
        parameters['b' + str(l)] = np.zeros([layer_dims[l], 1])

    return parameters


def initialize_parameters_random(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)  # This seed makes sure your "random" numbers will be the as ours
    parameters = {}
    L = len(layers_dims)  # integer representing the number of layers

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros([layers_dims[l], 1])
        ### END CODE HERE ###

    return parameters

# parameters = model(train_X, train_Y, initialization = "random")
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)   # integer representing the number of layers
    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        ### END CODE HERE ###

    return parameters


parameters = model(train_X, train_Y, initialization = "he")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)