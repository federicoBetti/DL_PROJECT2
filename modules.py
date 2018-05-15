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
        self.results = [x]
        self.activations = []
        output = x
        for layer in self.layers:
            output = layer.forward(output)
            self.results.append(output)
            output = layer.activation.forward(output)
            self.activations.append(output)
        return output

    def backward(self, loss, target, mini_batch):
        # if self.db == [] and self.dw == []:
        # self.db = [FloatTensor(Size( [mini_batch] + [x for x in layer.bias.shape] )).zero_() for layer in self.layers]
        # self.dw = [FloatTensor(Size( [mini_batch] + [x for x in layer.weigths.shape] )).zero_() for layer in self.layers]
        db = self.db
        dw = self.dw

        x_lt = self.activations[-1]  # output of activation of the last layer
        s_lt = self.results[-1]  # output of the last layer
        x_lb = self.activations[-2]  # output of the second last layer, without activation
        dsigma = self.layers[-1].activation.backward(s_lt)  # derivative of the last activation function

        dldx = loss.prime(x_lt, target)  # compute the derivtive of the Loss in the output layer
        dlds = dsigma * dldx  # error in the output layer, that must be computed alone

        temp_db = dlds.sum(0)
        db[-1] += temp_db  # accumulate all the biases of the minibatch. Size = #output_neurons

        # print('before temp', x_lb.shape, dlds.shape)
        # i want temp as a tensor #minibatch_size x #neurons_last_layer x #neurons_output_layer
        temp = MiniBatch3DMul(x_lb, dlds)
        # print(temp.sum(0).shape)
        # print(dw[-1].shape)
        dw[-1] += temp.sum(0)  # temp.sum(0) does a sum on the minibatch dimension and sum it to the accumulate gradient
        # at the end i think that for each minibatch temp.sum(0) is dw, because dw was 0 and it will never be updated
        # += operations may be useless, = can be ok too probably

        for i in range(2, self.num_layers + 1):
            x_lt = self.activations[-i]
            s_lt = self.results[-i]

            # we are in the first layer, than we have to use input as activation of previous layer. Input are stored
            # in self.results[0]. This is ok now, some problems may arise when we have a net with only one layer
            # since in the first part of this function we have used self.activations[-1]. In that case there wouldn't
            #  be any self.activaitions layer
            if i + 1 < self.num_layers:
                x_lb = self.activations[-(i + 1)]
            else:
                x_lb = self.results[0]

            dsigma = self.layers[-i].activation.backward(s_lt)
            w = self.layers[-i + 1].weigths

            dldx = (dlds).mm(w.t())
            dlds = dldx * dsigma  # compute the error wrt the error in the next layer
            # print('dlds:', dlds.shape)
            db[-i] = dlds.sum(0)

            # print('before temp', x_lb.shape, dlds.shape)
            temp = MiniBatch3DMul(x_lb, dlds)  # compute the loss wrt all weighs in the network
            # print(temp.sum(0).shape)
            dw[-i] = temp.sum(0)
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
        # print("x shape:", x.shape)
        # print("w shape:", self.weigths.shape)
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

