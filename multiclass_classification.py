import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from nn_lib_v2 import *
def preprocess(dataset, expand_dims = True, normalize = True, one_hot_encoding = False):
    dataset = np.array(dataset).copy()
    features, label = dataset[:,1:], dataset[:,0]
    label_raw = label.copy()
    if expand_dims == True:
        label = np.expand_dims(label, axis=0)
    if normalize == True:
        features = features / 255.0
    if one_hot_encoding == True:
        ohe = OneHotEncoder()
        label = np.expand_dims(label, axis=1)
        label = ohe.fit_transform(label).toarray()
    return features, label, label_raw
train = pd.read_csv("dataset/mnist/mnist_train.csv")
test = pd.read_csv("dataset/mnist/mnist_test.csv")
X_test, y_test, y_raw = preprocess(test, expand_dims= False, normalize= True, one_hot_encoding=True)
X_train, y_train, y_raw_train = preprocess(train, expand_dims= False, normalize= True, one_hot_encoding=True)

num_feature = X_test.shape[1]
num_class = y_test.shape[1]

layer_dims = [num_feature, 32,32, num_class]
print(y_test.shape)
params, costs = L_layer_model(X_test.T, y_test.T, layer_dims, last_layer='softmax', learning_rate=0.1, num_iterations=2000, print_cost=True)

y_hat = predict(X_test.T, params, lastlayer='softmax')
print(acc(y_hat, y_raw))

# params = initialize_parameters(layer_dims)
# AL, caches = L_model_forward(X_test.T, params, last_layer='softmax')
# grads = L_model_backward(AL, y_test, caches, last_layer='softmax')
# Y = y_test.reshape(AL.shape)
# dAL = -(np.divide(Y, AL))
# current_cache = caches[-1]
# linear_cache, activation_cache = current_cache
# softmax_backward(dAL, activation_cache)


