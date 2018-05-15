from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from torch import LongTensor, ByteTensor

import activations as F
import loss
import modules as nn
import optimizer
from Utils.datasets import generate_disc_set
from Utils.support_functions import compute_nb_errors, normalize, plot_points, get_correct_indexes

center = np.random.random(2)  # random center
train_input, train_target = generate_disc_set(1500, center)
val_input, val_target = generate_disc_set(2000, center)  # used to check performances during training
test_input, test_target = generate_disc_set(4000, center)  # used to check final accuracy

train_input, val_input, test_input = normalize(train_input, val_input, test_input)

# plot learning objective
plot_points(test_target, test_input, test_target.max(1)[1], test_target)

# initialize plots for real-time graphs
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.show()
fig.canvas.draw()


def train_model(model, optim, train_input, train_target, val_input, val_target, mini_batch_size, epoch, criterion,
                early_stopping=False):
    previous_loss = 1e5
    patience_count = 0
    patience = 5

    for e in range(epoch):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            if (train_input.size(0) - b) < mini_batch_size:
                mini_batch_size = train_input.size(0) - b

            optim.zero_grad(model)

            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.apply(output, train_target.narrow(0, b, mini_batch_size))
            model.backward(criterion, train_target.narrow(0, b, mini_batch_size), mini_batch_size)

            optim.step(model)
            sum_loss = sum_loss + loss

        # plot semi-real time graph every 30 epochs
        if not e % 30:
            ax.clear()

            # blue = misspredicted target, green = correct positive target, red = correct negative target
            green_ind, red_ind, blue_ind = get_correct_indexes(model, val_input, val_target)

            ax.plot(val_input[:, 0][green_ind].numpy(), val_input[:, 1][green_ind].numpy(), 'go')
            ax.plot(val_input[:, 0][red_ind].numpy(), val_input[:, 1][red_ind].numpy(), 'ro')
            ax.plot(val_input[:, 0][blue_ind].numpy(), val_input[:, 1][blue_ind].numpy(), 'bo')
            fig.canvas.draw()
            sleep(0.1)

        # print epoch, loss and validation_accuracy every 100 epochs
        if not e % 100:
            nb_errors = compute_nb_errors(model, val_input, val_target)
            print("epoch: ", e, " loss: ", sum_loss, 'accuracy:', 1 - (nb_errors / val_target.shape[0]))

        # early stopping implementaion
        if early_stopping:
            if sum_loss > previous_loss:
                patience_count += 1
                if patience_count > patience:
                    return
            else:
                patience_count = 0

            previous_loss = sum_loss


lr = 1e-2
optim = optimizer.SGD(lr)
mini_batch_size = 100
epoch = 10000
criterion = loss.MSELoss()

model = nn.Sequential(
    nn.Dense(2, 25, F.ReLU()),
    nn.Dense(25, 25, F.ReLU()),
    nn.Dense(25, 25, F.ReLU()),
    nn.Dense(25, 2, F.Sigmoid())
)

print(train_input.shape)
train_model(model, optim, train_input, train_target, val_input, val_target, mini_batch_size, epoch, criterion, False)

# plot final result on test dataset: accuracy and graph
nb_errors = compute_nb_errors(model, test_input, test_target)
print(1 - nb_errors / test_input.shape[0])
plot_points(test_target, test_input, model.forward(test_input).max(1)[1], model.forward(test_input).gt(0.5))
