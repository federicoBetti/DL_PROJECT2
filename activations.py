from math import nan

import sys
from torch import FloatTensor
import numpy as np

from Utils import exceptions_check
from modules import Module


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        exceptions_check.checkFloatTensor(input)
        return input.clamp(min=0)

    def backward(self, input):
        exceptions_check.checkFloatTensor(input)
        return input.clamp(min=0).gt(0).type(FloatTensor)

    def is_layer(self):
        return False


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        exceptions_check.checkFloatTensor(input)
        return input.tanh()

    def backward(self, input):
        exceptions_check.checkFloatTensor(input)
        return 4 * (input.exp() + input.mul(-1).exp()).pow(-2)

    def is_layer(self):
        return False


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        exceptions_check.checkFloatTensor(input)
        return input.sigmoid()

    def backward(self, input):
        exceptions_check.checkFloatTensor(input)
        return input.sigmoid().mul(1-input.sigmoid())

    def is_layer(self):
        return False


class Linear(Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, input):
        exceptions_check.checkFloatTensor(input)
        return input

    def backward(self, input):
        exceptions_check.checkFloatTensor(input)
        return FloatTensor(input.shape).fill_(1)

    def is_layer(self):
        return False


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.i = 0

    def forward(self, input):
        exceptions_check.checkFloatTensor(input)
        exps = input.exp()
        return exps / exps.sum(1).unsqueeze(-1).expand(exps.shape)

    def backward(self, input):
        exceptions_check.checkFloatTensor(input)
        soft = self.forward(input)
        return soft * (1 - soft)

    def is_layer(self):
        return False
