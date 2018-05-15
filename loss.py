from torch import FloatTensor

from Utils import exceptions_check
from modules import Module


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def apply(self, v, t):
        return (v - t.resize_(v.size())).pow(2).sum() # we need this resize although v: (100,1), t:(100) produce as
        # result (100,100). We want (100,1)

    def prime(self, v, t):
        return 2 * (v - t.resize_(v.size()))  # we need this resize although v: (100,1), t:(100) produce as result (
        # 100,100). We want (100,1)

    def is_layer(self):
        return False