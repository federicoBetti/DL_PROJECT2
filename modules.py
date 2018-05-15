import numpy as np
from torch import FloatTensor, Size
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
    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.layers = [e for e in args if e.is_layer()]
        self.num_layers = len(self.layers)
        self.db = [FloatTensor(layer.bias.shape).zero_() for layer in self.layers]
        self.dw = [FloatTensor(layer.weigths.shape).zero_() for layer in self.layers]
        self.results = []  # to store results in the forward pass
        self.activation = []  # to store activation function results in the forward pass

    def forward(self, x):
        self.results = []
        self.activations = [x]
        output = x
        for layer in self.layers:
            output = layer.forward(output)
            self.results.append(output)
            output = layer.activation.forward(output)
            self.activations.append(output)
        return output

    def backward(self, loss, target, mini_batch):

        db = self.db
        dw = self.dw

        x_lt = self.activations[-1]  # output of activation of the last layer
        x_lb = self.activations[-2]  # output of the second last layer, without activation
        s_lt = self.results[-1]  # output of the last layer
        dsigma = self.layers[-1].activation.backward(s_lt)  # derivative of the last activation function

        dldx = loss.prime(x_lt, target)  # compute the derivtive of the Loss in the output layer
        dlds = dsigma * dldx  # error in the output layer, that must be computed alone

        db[-1].add_(dlds.sum(0))  # accumulate all the biases of the minibatch. Size = #output_neurons
        dw[-1].add_(x_lb.t().mm(dlds))

        for i in range(2, self.num_layers+1):
            x_lt = self.activations[-i]
            x_lb = self.activations[-(i + 1)]            
            s_lt = self.results[-i]

            dsigma = self.layers[-i].activation.backward(s_lt)
            w = self.layers[-i + 1].weigths

            dldx = (dlds).mm(w.t())
            dlds = dldx * dsigma  # compute the error wrt the error in the next layer

            db[-i].add_(dlds.sum(0))
            dw[-i].add_(x_lb.t().mm(dlds)) # compute the loss wrt all weighs in the network

        return dw, db

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False


class Dense(Module):
    def __init__(self, in_neurons, out_neurons, activation):
        super(Dense, self).__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.activation = activation
        self.weigths = FloatTensor(in_neurons, out_neurons).normal_()
        self.bias = FloatTensor(out_neurons).zero_()
        self.error = 0

    # x is input
    def forward(self, x):
        exceptions_check.checkFloatTensor(x)
        return x.mm(self.weigths).add(self.bias)

    def backward(self, input): # never used, because it should be linear since we don't have activations here
        self.error = self.weigths.t().mul(x)
        return self.error

    def is_layer(self):
        return True

    # def __repr__(self):
    #     return 'Linear layer\n ' + 'Input: ' + str(self.in_neurons) + ', ' + 'Output: ' + str(self.out_neurons)


'''
Notes for future:
ogni layer deve tenere traccia della sua funzione di attivazione:
ogni layer Netowrk(ex linear) ha una funzione ugualianza y=x come attivazione
ogni layer activations ha come weigths un vettore di tutti uno
'''

# Utils functions

def MiniBatch3DMul(x_lb, dlds):
    assert x_lb.shape[0] == dlds.shape[0]
    x_lb = x_lb.unsqueeze(-1)
    dlds = dlds.unsqueeze(-1)
    result = FloatTensor(x_lb.shape[0], x_lb.shape[1], dlds.shape[1]).fill_(0)
    for i in range(x_lb.shape[0]):
        result[i] = x_lb[i].mm(dlds[i].t())
    return result

