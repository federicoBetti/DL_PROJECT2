import modules as nn
import activations as F
import dlc_practical_prologue as prologue
import loss
import optimizer
from torch import FloatTensor, Size, LongTensor
import numpy as np
import matplotlib.pyplot as plt

from Utils.datasets import generate_radius

import sys

#train_input, train_target, test_input, test_target = prologue.load_data(cifar=False, one_hot_labels=True,normalize=True)
from Utils.support_functions import compute_nb_errors

train_input, train_target = generate_radius(1000)
val_input , val_target = generate_radius(2000)
test_input, test_target = generate_radius(2000)

color= ['red' if l == 0 else 'green' for l in val_target]
plt.scatter(val_input[:, 0], val_input[:, 1], color=color)
plt.show()

def train_model(model, optim, train_input, train_target, val_input, val_target, mini_batch_size, epoch, criterion):
    mini_batch_size_original = mini_batch_size
    for e in range(epoch):
        sum_loss = 0
        # We do this with mini-batches
        mini_batch_size = mini_batch_size_original
        for b in range(0, train_input.size(0), mini_batch_size):
            if (train_input.size(0) - b) < mini_batch_size:
                mini_batch_size = train_input.size(0) - b

            optim.zero_grad(model)

            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            # print(output)  # shape mini_batch_siize x #output_neurons

            loss = criterion.apply(output, train_target.narrow(0, b, mini_batch_size))
            # print("loss: ", loss)

            model.backward(criterion, train_target.narrow(0, b, mini_batch_size), mini_batch_size)
            optim.step(model)
            sum_loss = sum_loss + loss

        nb_errors = compute_nb_errors(model, val_input, val_target)

        # if e//20:
        #         color = ['red' if int(l) == 0 else 'green' for l in model.forward(val_input).gt(0.5)]
        #         plt.scatter(val_input[:, 0], val_input[:, 1], color=color)
        #         plt.show()
        print("epoch: ", e, " loss: ", sum_loss, 'accuracy:', 1 - (nb_errors / val_target.shape[0]))


lr = 1e-3
optim = optimizer.SGD(lr)
mini_batch_size = 100
epoch = 1000
criterion = loss.MSELoss()

test_model = nn.Sequential(
    nn.Dense(2, 20, F.Sigmoid()),
    nn.Dense(20, 1, F.Sigmoid())
)

model = test_model

print(train_input.shape)
train_model(model, optim, train_input, train_target, val_input, val_target, mini_batch_size, epoch, criterion)
nb_errors = compute_nb_errors(model, test_input, test_target)
print(1-nb_errors/test_input.shape[0])

color= ['red' if int(l) == 0 else 'green' for l in model.forward(test_input).gt(0.5)]
plt.scatter(test_input[:, 0], test_input[:, 1], color=color)
plt.show()


