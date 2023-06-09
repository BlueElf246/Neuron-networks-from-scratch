import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2./layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters
def sigmoid(Z):
    result = 1/(1+np.exp(-Z))
    return result, Z

def relu(Z):
    return np.maximum(Z, 0), Z

def softmax(Z):
    t = np.exp(Z - np.max(Z))
    return t/np.sum(t, axis=0, keepdims=True), Z

### FORWARD PASS ###
def forward_pass(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = forward_pass(A_prev, W, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    if activation == 'relu':
        A, activation_cache = relu(Z)
    if activation == 'softmax':
        A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters, last_layer = 'sigmoid'):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], last_layer)
    caches.append(cache)
    return AL, caches

### COMPUTE COST ###
def compute_cost(AL, Y, last_layer = 'sigmoid'):
    m = Y.shape[1]
    Y = Y.reshape(AL.shape)
    if last_layer == 'sigmoid':
        cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)), keepdims=True)
    if last_layer == 'softmax':
        AL = np.clip(AL, 1e-7, 1-1e-7)
        a = -np.sum(np.multiply(Y, np.log(AL)), axis=0, keepdims=True) # 9999,
        cost = np.mean(a)

    cost = np.squeeze(cost)
    return cost

### BACKWARD PASS ###
def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ
def softmax_backward(dA, activation_cache):
    Z = activation_cache
    a = np.sum(np.multiply(dA, Z), axis=0, keepdims=True) # 1,9999
    dZ = np.multiply(Z,(dA-a))
    return dZ

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    if activation == 'softmax':
        dZ = softmax_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db
def L_model_backward(AL, Y, caches, last_layer = 'sigmoid'):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    if last_layer == 'sigmoid':
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        # dAL = (np.divide(1 - Y, 1 - AL) - np.divide(Y, AL)) * 1/m
    if last_layer == 'softmax':
        # dZ = last_layer_backward(AL, Y)
        dAL = -np.divide(Y,AL)
    current_cache = caches[-1]
    # dA_prev_temp, dW_temp, db_temp = linear_backward(dZ, current_cache[0])
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, last_layer)
    grads['dA' + str(L - 1)] = dA_prev_temp
    grads['dW' + str(L)] = dW_temp
    grads['db' + str(L)] = db_temp

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, 'relu')

        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp
    return grads


def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2

    for l in range(L):
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layer_dims, learning_rate=0.001, num_iterations=3000, print_cost=False, last_layer = 'sigmoid'):
    costs = []
    parameters = initialize_parameters(layer_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters, last_layer= last_layer)
        cost = compute_cost(AL, Y, last_layer=last_layer)
        grads = L_model_backward(AL, Y, caches, last_layer= last_layer)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 10 == 0:
            print(cost, f'at iter {i}')
            costs.append(cost)
    return parameters, cost


def predict(X, params, lastlayer='softmax'):
    AL, caches = L_model_forward(X, params)
    l = np.zeros_like(AL)
    if lastlayer == 'softmax':
        l = np.argmax(AL,axis=0)
        return l
    l[AL > 0.9] = 1
    return l

def acc(predict, label):
    y_hat = np.squeeze(predict)
    label = np.squeeze(label)
    c = 0
    for x in range(len(y_hat)):
        if int(y_hat[x]) == label[x]:
            c += 1
    return c / y_hat.shape[0]