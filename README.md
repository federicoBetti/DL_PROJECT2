# DL_PROJECT2 by Federico Betti, Gauthier Mueller

Your framework should import only torch.FloatTensor and torch.LongTensor from pytorch, and use no pre-existing neural-network python toolbox.  <br />
Your framework must provide the necessary tools to: <br />
- build networks combining fully connected layers, Tanh, and ReLU, <br />
- run the forward and backward passes, <br />
- optimize parameters with SGD for MSE. <br />
  
You must implement a test executable named test.py that imports your framework and <br />
- Generates a training and a test set of 1; 000 points sampled uniformly in [0; 1]2, each with a label 0 if outside the disk of radius 1=p2π and 1 inside  <br />
- builds a network with two input units, two output units, three hidden layers of 25 units  <br />
- trains it with MSE, logging the loss  <br />
- computes and prints the final train and the test errors  <br />
