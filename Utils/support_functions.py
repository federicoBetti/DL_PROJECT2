from torch import FloatTensor, LongTensor, ByteTensor
import matplotlib.pyplot as plt


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


def normalize(train_input, val_input, test_input):
    mean = train_input.mean(0)
    std = train_input.std(0)

    train_input -= mean
    val_input -= mean
    test_input -= mean

    train_input /= std
    val_input /= std
    test_input /= std
    return train_input, val_input, test_input


def plot_points(test_target, input, two_out_list, one_out_list):
    if len(test_target.shape) > 1:
        color = ['red' if int(i_o) == 0 else 'green' for i_o in two_out_list]  # two output
    else:
        color = ['red' if l == 0 else 'green' for l in one_out_list]
    plt.scatter(input[:, 0], input[:, 1], color=color)
    plt.show()


def get_correct_indexes(model, val_input, val_target):
    _, index = model.forward(val_input).max(1)
    _, correct_index = val_target.max(1)
    blue = correct_index.ne(index).type(LongTensor)
    opposite = (LongTensor(index.shape).fill_(1) - index).sub_(blue).clamp(min=0).type(ByteTensor)
    index = index.sub_(blue).clamp(min=0).type(ByteTensor)
    return index, opposite, blue.type(ByteTensor)
