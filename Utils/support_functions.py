from torch import FloatTensor


def compute_nb_errors(model, t_input, target):
    output = model.forward(t_input)
    target = target.unsqueeze(-1)
    output = output.gt(0.5).type(FloatTensor)  # 1 if > 0.5, 0 otherwise. It fit with Sigmoid output function
    return int(sum((target - output).ne(0).type(FloatTensor)))  # return the number of errors
