#!/usr/bin/env python3

import numpy as np
import random


class SigmoidNeuron:

    def __init__(self, n_inputs):
        """
        params: 
            n_inputs: number of inputs for neuron
            alpha: speed of learning
        w is vector of weights. 
        b is bias.
        """
        self.w = np.random.random(n_inputs)
        self.b = np.random.random()

    def apply(self, inputs):
        return sigmoid(np.dot(self.w, inputs) + self.b)

    def fit(self, x, y, alpha=0.01, epochs=10, verbose=True):
        assert len(x) == len(y)
        x = np.array(x)

        for epoch in range(epochs):

            ix = list(range(len(x)))
            random.shuffle(ix)
            for i in ix:
                self._adjust_weights(x[i], y[i], alpha)

            if not verbose:
                continue 

            print('Epoch: {}'.format(epoch))

            for i in range(len(x)):
                ans = self.apply(x[i])
                print(x[i], y[i], ans)

            print(self.w, self.b)
        
    
    def _adjust_weights(self, inputs, true_ans, alpha):
        s = self.apply(inputs)
        k = alpha * s * (1 - s) * (s - true_ans)
        self.w -= k * inputs
        self.b -= k

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

########################################################

def and_gate():

    x = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    y = [0, 0, 0, 1]

    neuron = SigmoidNeuron(n_inputs=2)
    neuron.fit(x, y, alpha=3, epochs=100)


def xor_gate():

    x = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]
    y = [0, 1, 1, 0]

    neuron = SigmoidNeuron(n_inputs=2)
    neuron.fit(x, y, alpha=3, epochs=100)

def main():
    random.seed = 17
    np.random.seed(17)
    
#    and_gate()
    xor_gate()   # not working (suddenly =) )

if __name__ == '__main__':
    main()
