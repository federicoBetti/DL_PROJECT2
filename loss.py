from torch import FloatTensor

from Utils import exceptions_check
from modules import Module


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def apply(self, v, t):
        return (v - t).pow(2).sum()

    def prime(self, v, t):
        return 2 * (v - t)

    def is_layer(self):
        return Falses