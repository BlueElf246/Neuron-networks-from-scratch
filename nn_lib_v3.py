import numpy as np
import matplotlib.pyplot as plt

### ADD L2 norm, dropout
def initialize_parameters(layer_dims, method='he'):
    parameters = {}
    L = len(layer_dims)  # number of layers in the network
    for l in range(1, L):
        if method == 'he':
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

def drop_out(A, dropout_prob):
    D = np.random.rand(A.shape[0], A.shape[1])
    D = D < dropout_prob  # return True/False matrix
    A = np.multiply(A, D)
    A = A / dropout_prob
    return (A, D)
def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = forward_pass(A_prev, W, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'softmax':
        A, activation_cache = softmax(Z)
    else:
        return
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters, last_layer = 'sigmoid', dropout_prob = 1.0, mode='training'):
    caches = []
    caches_D = []
    A = X
    L = len(parameters) // 2 # number of layer except input layer
    if type(dropout_prob) != type(list):
        dropout_prob = np.repeat(dropout_prob, L+1)
    drop_out(A,dropout_prob[0])
    for l in range(1, L): # if L = n, then l = 1,...,n-1 -> only iterate hidden layer
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], 'relu')
        if mode == 'training':
            A, D = drop_out(A, dropout_prob[l])
            cache_D = (D, dropout_prob[l])
            caches_D.append(cache_D)
            caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], last_layer)
    caches.append(cache)
    return AL, caches, caches_D

### COMPUTE COST ###
def compute_cost(AL, Y, params, lamda = 0.1, last_layer = 'sigmoid'):
    m = Y.shape[1]
    L = len(params) // 2
    Y = Y.reshape(AL.shape)
    if last_layer == 'sigmoid':
        cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)), keepdims=True)
    if last_layer == 'softmax':
        AL = np.clip(AL, 1e-7, 1-1e-7)
        a = -np.sum(np.multiply(Y, np.log(AL)), axis=0, keepdims=True)# 9999,
        cost = np.mean(a)
    w_norm = 0
    for l in range(1,L+1):
        w_norm+= np.sum(np.square(params[f'W{l}']))
    w_norm = lamda * (w_norm) / (2*m)
    cost = np.squeeze(cost) + w_norm
    return cost

### BACKWARD PASS ###
def relu_backward(dA, activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_backward(dZ, cache, lamda=0.1):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T) + (lamda/m)*W
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def drop_out_backward(dA_prev, caches_D):
    D, dropout_prob = caches_D
    dA_prev = np.multiply(dA_prev, D)
    dA_prev /= dropout_prob
    return dA_prev

def linear_activation_backward(dA, cache, activation, lamda=0.1):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache, lamda)
    return dA_prev, dW, db

def last_layer_backward(AL, Y):
    dZ = (AL - Y)
    return dZ

def L_model_backward(AL, Y, caches, caches_D, lamda=0.1):
    grads = {}
    L = len(caches)
    dZ = last_layer_backward(AL, Y)
    current_cache = caches[-1]
    dA_prev_temp, dW_temp, db_temp = linear_backward(dZ, current_cache[0], lamda)
    dA_prev_temp = drop_out_backward(dA_prev_temp, caches_D[-1])
    # 3,1000
    grads['dA' + str(L - 1)] = dA_prev_temp
    grads['dW' + str(L)] = dW_temp
    grads['db' + str(L)] = db_temp
    for l in reversed(range(L - 1)):  # if L = 3, then l= 1,...0, add 1 to except 0(input)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, 'relu', lamda=lamda)
        if (l-1) != -1:
            dA_prev_temp = drop_out_backward(dA_prev_temp, caches_D[l-1])
        grads['dA' + str(l)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp
    return grads

def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2  # L=3
    for l in range(L): # l=0,...2
        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layer_dims, learning_rate=0.001,
                  num_iterations=3000, print_cost=False,
                  last_layer = 'sigmoid', lamda=0.1, dropout_prob=0.8):
    costs = []
    parameters = initialize_parameters(layer_dims)
    for i in range(0, num_iterations):
        AL, caches, caches_D = L_model_forward(X, parameters, last_layer= last_layer, dropout_prob=dropout_prob)
        cost = compute_cost(AL, Y, parameters, last_layer=last_layer, lamda=lamda)
        grads = L_model_backward(AL, Y, caches, caches_D, lamda=lamda)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print(cost, f'at iter {i}')
            costs.append(cost)
    return parameters, cost
def predict(X, params, lastlayer='softmax'):
    AL, caches, caches_D = L_model_forward(X, params, mode='predict')
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
