from matplotlib.colors import ListedColormap, colorConverter
from torch import FloatTensor, LongTensor, ByteTensor
import matplotlib.pyplot as plt
import numpy as np


# Compute numbers of mispredicted labels
def compute_nb_errors(model, t_input, target):
    output = model.forward(t_input)
    if output.shape[1] > 1:  # target class in one hot encoding
        target = target.type(LongTensor)
        m_o, i_o = output.max(1)
        m_t, i_t = target.max(1)
        return sum(i_o.ne(i_t))
    else:  # used for target with only one dimension between 0 and #classes
        target = target.type(LongTensor)
        m_t, i_t = target.max(1)
        return int(
            sum(output.gt(0.5).type(LongTensor).ne(i_t.unsqueeze(-1)).type(FloatTensor)))  # 0.5 because of Sigmoid


# Normalize data wrt their mean and std
def standardization(train_input, val_input, test_input, std, mean):
    train_input -= mean
    val_input -= mean
    test_input -= mean

    train_input /= std
    val_input /= std
    test_input /= std
    return train_input, val_input, test_input


# Revert standardization process
def add_std_mean(param, std, mean):
    new_param = param.clone()
    return new_param * std + mean


# Plot points of a given dataset in two different colors. Only 2 different classes permitted
def plot_points(test_target, input, two_out_list, one_out_list, title=''):
    if len(test_target.shape) > 1:
        color = ['red' if int(i_o) == 0 else 'green' for i_o in two_out_list]  # two output
    else:
        color = ['red' if l == 0 else 'green' for l in one_out_list]
    plt.scatter(input[:, 0], input[:, 1], color=color)
    plt.title(title)
    plt.show()


# Return correct indexes, both for [0, 1] and [1, 0] labels, and mispredicted indexes
def get_correct_indexes(model, t_input, t_target):
    _, correct = model.forward(t_input).max(1)
    _, correct_index = t_target.max(1)
    blue = correct_index.ne(correct).type(LongTensor)
    opposite = (LongTensor(correct.shape).fill_(1) - correct).sub_(blue).clamp(min=0).type(ByteTensor)
    correct = correct.sub_(blue).clamp(min=0).type(ByteTensor)
    return correct, opposite, blue.type(ByteTensor)


# Initialize plot for real time visualization
def plot_initialization(std, mean):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.show()
    nb_of_xs = 200
    xs1 = np.linspace(-2, 2, num=nb_of_xs)
    xs2 = np.linspace(-2, 2, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2)  # create the grid
    ex = FloatTensor(len(xx) * len(yy), 2).fill_(0)
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            ex[nb_of_xs * i + j, 0] = xx[i, j]
            ex[nb_of_xs * i + j, 1] = yy[i, j]
    xx = xx * std[0] + mean[0]
    yy = yy * std[1] + mean[1]
    return ax, xx, yy, ex, fig


# Update real time visualization, compute prediction for all points in the [0, 1]^2 plane
def real_time_plot(model, ex, xx, yy, ax, fig, val_input_plot, val_input, val_target):
    classification_plane = model.forward(ex).max(1)[1].view(200, 200).numpy()
    cmap = ListedColormap([
        colorConverter.to_rgba('r', alpha=0.50),
        colorConverter.to_rgba('g', alpha=0.50)])
    ax.contourf(xx, yy, classification_plane, cmap=cmap)

    # blue = mispredicted target, green = correct positive target, red = correct negative target
    green_ind, red_ind, blue_ind = get_correct_indexes(model, val_input, val_target)

    ax.plot(val_input_plot[:, 0][green_ind].numpy(), val_input_plot[:, 1][green_ind].numpy(), 'go')
    ax.plot(val_input_plot[:, 0][red_ind].numpy(), val_input_plot[:, 1][red_ind].numpy(), 'ro')
    ax.plot(val_input_plot[:, 0][blue_ind].numpy(), val_input_plot[:, 1][blue_ind].numpy(), 'bo')
    ax.set_title('Dynamic results on validation dataset')
    fig.canvas.draw()
    return ax, fig
