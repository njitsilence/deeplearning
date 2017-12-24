import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from utils.opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from utils.opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases.testCases_opt import *


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        ### END CODE HERE ###

    return parameters


# parameters, grads, learning_rate = update_parameters_with_gd_test_case()

# parameters = update_parameters_with_gd(parameters, grads, learning_rate)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def random_mini_batches(X,Y,mini_batch_size=64,seed=0):
    """
    Creates a list of random minibatches from (X,Y)
    :param X: input data, of shape(input size, number of examples)
    :param Y: true "label" vector(1 for blue dot, 0 for red dot), of shape (1, number of examples)
    :param mini_batch_size: size of the mini-batches, integer
    :param seed:
    :return:

    mini-batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)
    m = X.shape[1] # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y), Minus the end case
    num_complete_minibatches = math.floor(m/mini_batch_size)




