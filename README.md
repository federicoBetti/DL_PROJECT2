# DL_PROJECT2 by Federico Betti, Gauthier Mueller

Your framework should import only torch.FloatTensor and torch.LongTensor from pytorch, and
use no pre-existing neural-network python toolbox.
Your framework must provide the necessary tools to:
• build networks combining fully connected layers, Tanh, and ReLU,
• run the forward and backward passes,
• optimize parameters with SGD for MSE.
You must implement a test executable named test.py that imports your framework and
• Generates a training and a test set of 1; 000 points sampled uniformly in [0; 1]2, each with a
label 0 if outside the disk of radius 1=p2π and 1 inside,
• builds a network with two input units, two output units, three hidden layers of 25 units
• trains it with MSE, logging the loss,
• computes and prints the final train and the test errors