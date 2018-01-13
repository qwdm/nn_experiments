#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

from layered_sigmoid import NeuralNetwork


def zero_one():
    data = load_digits()  # Bunch datatype
    net = NeuralNetwork(n_inputs=64, n_neurons_list=[5,1])

    # get only 1 and 0 pics
    X, Y = list(zip(*[(x, y) for (x, y) in zip(data['data'], data['target']) if y in (0, 1)]))
    X = np.array(X)
    Y = np.array(Y)
    Y = Y.reshape((len(Y), 1))

    net.fit(X[4:], Y[4:], alpha=1, epochs=100)

    for i in range(20):
        ans = net.apply(X[i])
        print(Y[i], ans)


def vectorize(Y):
    maxval = max(Y)
    return [ ([0] * y + [1] + [0] * (maxval - y)) for y in Y]

def maxi(lst):
    return max(enumerate(lst), key=lambda x: x[1])[0]

def main():
    data = load_digits()  # Bunch datatype
    net = NeuralNetwork(n_inputs=64, n_neurons_list=[10, 10])

    X = data['data']
    Y = data['target']
    Y = vectorize(Y)

    X = np.array(X)
    Y = np.array(Y)

    N_TEST = 100 

    test_accs = []
    train_accs = []

    for _ in range(300):
        net.fit(X[N_TEST:], Y[N_TEST:], alpha=0.1, epochs=1)

        right = 0
        for i in range(N_TEST):
            ans = net.apply(X[i])
            right += int(maxi(Y[i]) == maxi(ans))
        test_accs.append(right / N_TEST)
#        print('test set accuracy:', test_accs[-1])

        right = 0
        for i in range(N_TEST, len(X)):
            ans = net.apply(X[i])
            right += int(maxi(Y[i]) == maxi(ans))
        train_accs.append(right / (i - N_TEST))
#        print('train set accuracy:', train_accs[-1])

#        print('============')

    plt.plot(test_accs)
    plt.plot(train_accs)

    plt.show()


if __name__ == '__main__':
    main()
