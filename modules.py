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
        self.results = []
        self.activation = []

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

        x_lt = self.activations[-1]
        s_lt = self.results[-1]
        x_lb = self.results[-2]
        dsigma = self.layers[-1].activation.backward(s_lt)

        # print("dsigma: ", dsigma.shape)
        # print("x_lt: ",x_lt.shape)
        # print("s_lt: ",s_lt.shape)
        # print("x_lb: ", x_lb.shape)
        # print("target: ", target.shape)

        dldx = loss.prime(x_lt, target)
        dlds = dsigma * dldx

        # print("dldx: ", dldx.shape)
        # print("dlds: ", dlds.shape)

        temp_db = dlds.sum(0)
        # print("temp_db: ", temp_db.shape)
        db[-1].add_(dlds.sum(0))
        # print("dw: ", dw[-1].shape)
        #temp = FloatTensor(Size([mini_batch] + [x_lb.shape[1]] + [dlds.shape[1]]))
        #print("temp: ", temp.shape)
        #for i in range(mini_batch):
        #    temp[i] = x_lb[i].view(-1,1).mm(dlds[i].view(1,-1))

        temp = x_lb.t().mm(dlds)
        # print("temp: ", temp.shape)
        #dw[-1].add_(x_lb.view(-1,1).mm(dlds.view(1,-1)))
        dw[-1].add_(temp)
        # print(self.num_layers)
        for i in range(2, self.num_layers+1):
            x_lt = self.activations[-i]
            s_lt = self.results[-i]
            x_lb = self.results[-(i+1)]

            dsigma = self.layers[-i].activation.backward(s_lt)
            # print("dsigma -2: ", dsigma.shape)
            w = self.layers[-i+1].weigths
            

            dldx = dlds.mm(w.t())
            # print("dldx: ", dldx.shape)
            # print(type(dldx))
            # print(type(dsigma))

            dlds = dsigma * dldx
            # print("dlds: ", dlds.shape)
            db[-i].add_(dlds.sum(0))

            # temp = FloatTensor(Size([mini_batch] + [x_lb.shape[1]] + [dlds.shape[1]]))
            # for j in range(mini_batch):
            #     temp[j] = x_lb[j].view(-1,1).mm(dlds[j].view(1,-1))
            temp = x_lb.t().mm(dlds)
            #dw[-i].add_(dlds.view(-1,1).mm(x_lb.view(1,-1)))            
            # print(-i)
            dw[-i].add_(temp)

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
    #x is input
    def forward(self, x):
        exceptions_check.checkFloatTensor(x)
        # print("x shape:", x.shape)
        # print("w shape:", self.weigths.shape)
        return x.mm(self.weigths).add(self.bias)

    def backward(self, input):
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
