from modules import Module


class MSELoss(Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def apply(self, v, t):
        return (v - t.resize_(v.size())).pow(2).sum() # we need this resize although v: (#batch_size, 1),
        # t:(#batch_size) produce as result (#batch_size,#batch_size). We want (#batch_size,1)

    def prime(self, v, t):
        return 2 * (v - t.resize_(v.size())) # we need this resize although v: (#batch_size, 1), t:(#batch_size)
        # produce as result (#batch_size,#batch_size). We want (#batch_size,1)

    def is_layer(self):
        return False