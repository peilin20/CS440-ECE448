# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py,neuralnet_leaderboard -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.model = nn.Sequential(nn.Conv2d(3, 6, kernel_size = 3, padding = 1),nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(6, 12, kernel_size = 3, padding = 1),
            nn.MaxPool2d(2, stride = 2),nn.Flatten(),nn.Linear(768, 256),
            nn.ReLU(),nn.Dropout(0.2),nn.Linear(256, 32),
            nn.Softplus(), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, out_size),nn.Softplus()
        )
        self.params = self.model.parameters()
        self.optimizer = optim.Adam(self.params, lr = lrate, weight_decay=0.0001)

        # raise NotImplementedError("You need to write this part!")
        
    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        #raise NotImplementedError("You need to write this part!")
        x = x.reshape(-1, 3, 32, 32)
        return self.model(x)

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        
        yhat=self.forward(x)
        loss = self.loss_fn(yhat,y)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        res=loss.detach().cpu().numpy()
        return res
        raise NotImplementedError("You need to write this part!")
        return 0.0

def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """

    L = 0
    losses = []
    yhats = []
    varN = int(train_set.size()[0])
    vark = int(varN / batch_size)
    trainsize=train_set.size()[0]
    train_set = (train_set - train_set.mean(0))/train_set.std(0)
    net = NeuralNet(0.01, nn.CrossEntropyLoss(), 3072, 4)
    for j in range(epochs):
    
        index = torch.randperm(trainsize)
        train_set = train_set[index]
        train_labels = train_labels[index]

        for i in range(vark):
            L =net.step(train_set[i* batch_size: min(batch_size * (1+i), len(train_set))], train_labels[i * batch_size: min(batch_size * (1+i ), len(train_set))])
        losses.append(L)
    
    finalset = (dev_set - dev_set.mean(0)) / dev_set.std(0)
    yhats =np.argmax(net.forward(finalset).detach().cpu().numpy(), axis = 1)
    res=net(finalset).detach().numpy()
    return losses,yhats,net
    #raise NotImplementedError("You need to write this part!")
    #return [],[],None
