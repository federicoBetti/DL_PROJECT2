import math

import numpy as np
from torch import FloatTensor


def generate_radius(number_points):
    input = FloatTensor(number_points, 2).uniform_(-1, 1)
    print(input.shape)
    target = input.add(-0.5).pow(2).sum(1).sub(0.1).sign().add(1).div(2)

    new_target = FloatTensor(number_points, 2).zero_()
    for i in range(target.shape[0]):
        if target[i] == 0:
            new_target[i][0] = 1
        else:
            new_target[i][1] = 1

    return input, new_target


def is_inside_circle(center, x, y):
    radius = 1 / math.sqrt(2 * math.pi)
    if math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) < radius:
        return True
    else:
        return False


def generate_disc_set(nb, center):
    input_train = FloatTensor(nb, 2).uniform_(0, 1)
    target = FloatTensor(nb, 2).zero_()
    # center = (0, 0)
    for index, line in enumerate(input_train):
        if is_inside_circle(center, line[0], line[1]):
            target[index][1] = 1
        else:
            target[index][0] = 1

    return input_train, target


def generate_spiral():
    N = 100  # number of points per class
    D = 2  # dimensionality
    K = 3  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y
