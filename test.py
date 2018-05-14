import modules as nn
import activations as F
import dlc_practical_prologue as prologue
import loss
import optimizer
from torch import FloatTensor, Size
import numpy as np

import sys

train_input, train_target, test_input, test_target = prologue.load_data(cifar=False, one_hot_labels=True,
                                                                        normalize=True)

def train_model(model, optim, train_input, train_target, val_input, val_target, mini_batch_size, epoch, criterion):
    mini_batch_size_original = mini_batch_size
    for e in range(epoch):
        sum_loss = 0
        # We do this with mini-batches
        mini_batch_size = mini_batch_size_original
        for b in range(0, train_input.size(0), mini_batch_size):
            if (train_input.size(0) - b) < mini_batch_size:
                mini_batch_size = train_input.size(0) - b

            #optim.zero_grad(model)
            #print("narrow: ", train_input.narrow(0, b, mini_batch_size).shape)

            output = model.forward(train_input.narrow(0, b, mini_batch_size))
            print(output[0])
            print(model.layers[0].weigths[0])

            loss = criterion.apply(output, train_target.narrow(0, b, mini_batch_size))
            #print("loss: ", loss)

            model.backward(criterion, train_target.narrow(0, b, mini_batch_size), mini_batch_size)
            optim.step(model)
            sum_loss = sum_loss + loss
        print("epoch: ", e," loss: ", sum_loss)

lr = 1e-5
optim = optimizer.SGD(lr)
mini_batch_size = 1000
epoch = 10
criterion = loss.MSELoss()

test_model = nn.Sequential(
	nn.Dense(784, 40, F.ReLU()),
	nn.Dense(40,20, F.ReLU()),
	nn.Dense(20,10, F.ReLU())
	)

model = test_model
val_input = None;
val_target = None;

print(train_input.shape)
train_model(model, optim, train_input, train_target, val_input, val_target, mini_batch_size, epoch, criterion)



# #to test stuff:
# a = FloatTensor(5,3).zero_()
# b = FloatTensor(1,3).normal_()
# print(a.add_(b))
# s = a.shape
# ss = [x for x in s]
# ss = [10] + ss
# print("ss", ss)

# c = FloatTensor(Size(ss))
# print(c.shape)


# print("HEEHEHEHEHEHEHE\n")
# a = FloatTensor([[-1,6], [-3,-4], [0.5,2], [3,4]])
# relu = F.ReLU()
# b = relu.forward(a)
# c = relu.backward(b)

# c = FloatTensor(4, 2, 2)
# for i in range(4):
# 	c[i] = a[i].view(-1,1).mm(b[i].view(1,-1))

# res = FloatTensor(2,2).zero_()
# for i in range(4):
# 	res = res + c[i] 
# print(a.mul(b))	
# print(res)
# print(a.t().mm(b))
# print(np.dot(a.t(), b))
# print(a.sum(1))
