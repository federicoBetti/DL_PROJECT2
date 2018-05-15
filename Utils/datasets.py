import math
from torch import FloatTensor

def generate_radius(number_points):
    input = FloatTensor(number_points, 2).uniform_(-1, 1)
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2)
    return input, target