#!/usr/bin/env python3

import numpy as np
import random
import copy


class SigmoidLayer:

    def __init__(self, n_inputs, n_neurons):
        self.w = np.random.random((n_neurons, n_inputs)) - 0.5
        self.b = np.random.random(n_neurons) - .5

    def apply(self, inputs):
        return sigmoid(np.dot(self.w, inputs) + self.b)
    
    def _adjust_weights(self, inputs, errors, alpha):
        s = self.apply(inputs)
        k = alpha * s * (1 - s) * errors
        self.w -= np.dot(np.array([k]).T, [inputs])
        self.b -= k

#        print('OOOO')
#        print(inputs,s)
#        print(k)
#        print(self.w)
#        print(self.b)


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

            print('Epoch: {}'.format(epoch))

            for i in range(len(X)):
                ans = self.apply(X[i])
                print(X[i], Y[i], ans)

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

        # calculate errors(local gradients) for all layers
        errors = [(outputs[-1] - y) * outputs[-1] * (1 - outputs[-1])]
        for layer, output in reversed(list(zip(self.layers[1:], outputs[1:-1]))):  # backpropagation, 
                                                                                 # very first layer dont produce error
#            print(output.shape, errors[-1].shape)
#            print(output, errors[-1])
            errors.append(np.dot(layer.w.T, (errors[-1] * output * (1 - output))))
        errors.reverse()

#        # calculate errors(local gradients) for all layers
#        errors = [outputs[-1] - y]
#        errors = [(outputs[-1] - y) * outputs[-1] * (1 - outputs[-1])]
#        for layer in reversed(self.layers[1:]):  # backpropagation, 
#                                                                                 # very first layer dont produce error
#            errors.append(np.dot(layer.w.T, errors[-1]))
#        errors.reverse()

#        print('errors', errors)

        # adjust weights for all layers
        for layer, input_vector, error_vector in zip(self.layers, outputs, errors):
            layer._adjust_weights(input_vector, error_vector, alpha)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    random.seed = 17
    np.random.seed(17)
    
##    and_gate()
#    xor_gate()   # not working (suddenly =) )
    
    net = NeuralNetwork(n_inputs=2, n_neurons_list=[3, 1])

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

    net.fit(X, Y, alpha=1, epochs=300)
        

if __name__ == '__main__':
    main()
