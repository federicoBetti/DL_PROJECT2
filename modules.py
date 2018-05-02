from torch import FloatTensor
from Utils import exceptions_check

class Module(object):
    def __init__(self):
        self.test = 0

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Sequential(Module):
    def __init__(self):
        super(Sequential, self).__init__()

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError



class Linear(Module):
    def __init__(self, in_neurons, out_neurons):
        super(Linear, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.weigths = FloatTensor(out_neurons, in_neurons)
        self.bias = FloatTensor(out_neurons)
        self.error = 0

    def forward(self, input):
        exceptions_check.checkFloatTensor(input)
        return self.weigths.mul(input).add(self.bias)

    def backward(self, input):
        self.error = self.weigths.t().mul(input)
        return self.error

'''
Notes for future:
ogni layer deve tenere traccia della sua funzione di attivazione:
ogni layer Netowrk(ex linear) ha una funzione ugualianza y=x come attivazione
ogni layer activations ha come weigths un vettore di tutti uno
'''
