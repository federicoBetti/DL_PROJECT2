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
        return input.clamp(min=0).gt(0)


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        exceptions_check.checkFloatTensor(input)
        return input.tanh()

    def backward(self, input):
        exceptions_check.checkFloatTensor(input)
        return 4 * (input.exp() + input.mul(-1).exp()).pow(-2)


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, input):
        exceptions_check.checkFloatTensor(input)
        return input.sigmoid()

    def backward(self, input):
        exceptions_check.checkFloatTensor(input)
        return input.sigmoid().mul(1-input.sigmoid())


class Linear(Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, input):
        exceptions_check.checkFloatTensor(input)
        return input

    def backward(self, input):
        exceptions_check.checkFloatTensor(input)
        return inp
