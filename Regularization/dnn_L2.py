#-*- coding:utf-8 -*-
# author:bingo
# datetime:2019/7/9
# Deep Neural Network with L2 Regularization

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

def model(X, Y, learning_rate=0.1, iterations=10000, lambd=0, keep_prob=1.0, print_cost=True):
    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    layers_dims = [X.shape[0], 20, 5, 1]

    # Initialization of parameters
    parameters = initialization_parameters(layers_dims)

    # training process
    for i in range(iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob ==1:
            # no dropout
            A3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            # dropout
            A3, cache = forward_propagation_dropout(X, parameters, keep_prob)

        # Cost function
        if lambd == 0:
            # without regularization
            cost = compute_cost(Y, A3)
        else:
            cost = compute_cost_with_regularization(Y, A3, parameters, lambd)

        # Backward propagation:
        if lambd == 0 and keep_prob == 1:
            # without regularization
            grads = backward_propagation(X, Y, cache)
        elif lambd > 0:
            # L2 regularization
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            # with dropout
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)


        if print_cost and i % 1000 == 0:
            costs.append(cost)
            print("Cost after iteration {}: {}".format(i, cost))

        # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def initialization_parameters(layers):
    parameters = {}
    L = len(layers)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers[l], layers[l-1]) / np.sqrt(layers[l-1])
        parameters['b' + str(l)] = np.zeros((layers[l], 1))
    return parameters

def forward_propagation(X, parameters):
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return A3, cache

def forward_propagation_dropout(X, parameters, keep_prob):
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    # dropout in layer 1
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = D1 < 1
    A1 = A1 * D1
    A1 = A1 / keep_prob
    #------------------------#
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    # dropout in layer 2
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < 1
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    return A3, cache

def relu(x):
    return np.maximum(x, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cost(Y, A):
    m = Y.shape[1]
    logprob = np.multiply(Y, -np.log(A)) + np.multiply(1-Y, -np.log(1-A))
    cost = np.nansum(logprob) / m
    return cost

def compute_cost_with_regularization(Y, A, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    cross_entropy_cost = compute_cost(Y, A)
    # cost of L2 regularization
    L2_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2 * m)

    return cross_entropy_cost + L2_cost

def backward_propagation(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T) / m + lambd * W3 / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = np.dot(dZ2, A1.T) / m + lambd * W2 / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.dot(dZ1, X.T) / m + lambd * W1 / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2   # conduct dropout
    dA2 = dA2 / keep_prob
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1  # conduct dropout
    dA1 = dA1 / keep_prob
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

def update_parameters(parameters, grads, learning_rate):
    n = len(parameters) // 2  # number of layers in the neural networks

    # Update rule for each parameter
    for k in range(n):
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]

    return parameters

def predict(X, Y, parameters):
    m = X.shape[1]
    p = np.zeros((1, m), dtype=np.int)

    A3, cache = forward_propagation(X, parameters)
    for i in range(m):
        if A3[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print('Accuracy: ' + str(np.mean((p[0, :] == Y[0, :]))))

    return (p > 0)

def plot_decision_boundary(model, X, y):
    xmin, xmax = X[0, :].min() - 1, X[0, :].max() + 1
    ymin, ymax = X[1, :].min() - 1, X[1, :].min() + 2
    h = 0.01

    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.scatter(X[0, :], X[1, :], c=y)
    plt.show()

def predict_dec(parameters, x):
    A3, cache = forward_propagation(x, parameters)

    predictions = (A3 > 0.5)
    return predictions

def main():
    data = loadmat('DataSet/data.mat')  # load data
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40)
    plt.show()

    learning_rate = 0.1
    lambd = 0.5
    iterations = 10000
    keep_prob = 0.8
    # parameters = model(train_X, train_Y)
    # parameters = model(train_X, train_Y, lambd=0.5)
    parameters = model(train_X, train_Y, keep_prob=0.9)

    print("On the train set:")
    predictions_train = predict(train_X, train_Y, parameters)
    print("On the test set:")
    predictions_test = predict(test_X, test_Y, parameters)

    # predictions_train = (predictions_train == 1)
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

if __name__ == '__main__':
    main()