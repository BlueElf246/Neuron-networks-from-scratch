# Neuron-networks-from-scratch
Hello welcome to my weekend-project on implementing deep learning architectures(ANN, CNN, RNN) from scratch. I also provide sample dataset to test out these models.

How to use it?🧐🧐🧐:

1/ Clone this git
2/ run binary_classification.py for example, run multiclass_classification.py(you should download mnist dataset)

All the code are in nn_lib_v2,v3

If you what to understand the math behind NN, I have written a small paper that explain everything(Implementing_Neuron_Network_from_scratch.pdf). Feel free to take a look at it 😩💁🏽🤩

It is also have the comparation part. Where I will write some codes to compare accuracy with different hyperparameters choice.

Untill now, I have implemented a set of functions for making neuron networks. You can see all of it in nn_lib_v2.

Features in nn_lib_v2.
-> binary_classfication with last layer is sigmoid

-> multiclass classification with last layer is softmax

-> 'he' initilization

### UPDATE nn_lib_v3.
-> provide L2 normalization

-> provide dropout

-> adding mini-batch gradient descent

# Note: prefer using nn_lib_v2 because drop_out feature in nn_lib_v3 very slow 😳

There are two application that use this lib named binary_classification.py and multiclass_classfication.py, where you have to download mnist dataset by yourself

The code is partly borrowed from propramming exercise in course 1 in the serie of deep learning course taught by Andrew Ng.

If you do not understanding anything. You are welcome to ask me vie my gmail: lemanhdat2112@gmail.com
Thank you, 😊 
