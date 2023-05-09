from nn_lib_v3 import *
import numpy as np
from sklearn.datasets import  make_moons
features, labels = make_moons(n_samples=1000, shuffle=True, random_state=10)
layer_dims = [2, 4, 3, 1]
def preprocessing(dataset):
    for col in range(dataset.shape[1]):
        mean = np.mean(dataset[:,col])
        var = np.var(dataset[:,col])
        dataset[:,col] = (dataset[:,col] - mean) / var
    return dataset
features = preprocessing(features)
X = np.array(features).T
Y = np.expand_dims(labels, axis=0)
params, cost = L_layer_model(X, Y, lamda=0.1, dropout_prob=[1.0,0.5,0.5,1.0],
                             layer_dims=layer_dims, num_iterations=2000,
                             learning_rate=0.1, print_cost=True, last_layer = 'sigmoid')

y_hat = predict(X, params, lastlayer='sigmoid')
print(y_hat)
print(acc(y_hat, Y))

