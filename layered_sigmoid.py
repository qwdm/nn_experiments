#!/usr/bin/env python3

import numpy as np
import random
import copy


class SigmoidLayer:

    def __init__(self, n_inputs, n_neurons):
        self.w = 7 * (np.random.random((n_neurons, n_inputs)) - 0.5)
        self.b = 7 * (np.random.random(n_neurons) - .5)

    def apply(self, inputs):
        return sigmoid(np.dot(self.w, inputs) + self.b)
    
    def _adjust_weights(self, inputs, local_gradient_vector, alpha):
        k = alpha * local_gradient_vector
        self.w -= np.dot(np.array([k]).T, [inputs])
        self.b -= k



class NeuralNetwork:

    def __init__(self, n_inputs, n_neurons_list):
        self.layers = [SigmoidLayer(n_inputs, n_neurons_list[0])]
        for i in range(len(n_neurons_list) - 1):
            self.layers.append(
                SigmoidLayer(n_neurons_list[i], n_neurons_list[i+1])
            )

    def fit(self, X, Y, alpha=0.01, epochs=10):
        assert iter(Y[0])  # Y must contain some sort of vectors
        assert len(X) == len(Y)

        X = np.array(X)
        Y = np.array(Y)

        for epoch in range(epochs):
            ix = list(range(len(X)))
            random.shuffle(ix)
            for i in ix:
                self._adjust_weights(X[i], Y[i], alpha)
                
#            if not verbose:
#                continue 

#            print('Epoch: {}'.format(epoch))
#
#            for i in range(len(X)):
#                ans = self.apply(X[i])
#                print(X[i], Y[i], ans)

#            print('Weights')
#            for layer in self.layers:
#                print(layer.w, layer.b)

    def apply(self, inputs):
        outputs = copy.copy(inputs)  # input layer
        for layer in self.layers:
            newinputs = outputs  # from prev layer
            outputs = layer.apply(newinputs)
        return outputs

    def _adjust_weights(self, inputs, y, alpha):
        # calculate outputs for all layers, including "input layer"
        outputs = [inputs]
        for layer in self.layers:
            outputs.append(layer.apply(outputs[-1]))

        # calculate local gradients for all layers, backpropagation
        local_gradients = [(outputs[-1] - y) * outputs[-1] * (1 - outputs[-1])]
        for layer, output in reversed(list(zip(self.layers[1:], outputs[1:-1]))):
            local_gradients.append(np.dot(layer.w.T, local_gradients[-1]) * output * (1 - output))
        local_gradients.reverse()

        # adjust weights for all layers
        for layer, input_vector, local_gradient_vector in zip(self.layers, outputs, local_gradients):
            layer._adjust_weights(input_vector, local_gradient_vector, alpha)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    random.seed = 7 
    np.random.seed(7)
    
    
    net = NeuralNetwork(n_inputs=2, n_neurons_list=[5, 1])

    X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]

    Y = np.array([
        0,
        1,
        1,
        0,
    ]
    ).reshape((4,1))

#    Y = np.array([
#        0,
#        0,
#        0,
#        1,
#    ]
#    ).reshape((4,1))

    print('Weights init')
    for layer in net.layers:
        print(layer.w)

    net.fit(X, Y, alpha=1, epochs=100)

    print('Weights after learning')
    for layer in net.layers:
        print(layer.w)
        
    for i in range(len(X)):
        ans = net.apply(X[i])
        print(X[i], Y[i], ans)

if __name__ == '__main__':
    main()
