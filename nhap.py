import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from nn_lib_v3 import *
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

batch = BatchGenerator(X_train, y_train, batch_size=128)
x, y = batch.next()
print(x.shape, y.shape)