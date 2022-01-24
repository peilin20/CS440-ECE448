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
within this file, neuralnet_learderboard and neuralnet_part2.py -- the unrevised staff files will be used for all other
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

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> h ->  out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate=lrate
        self.model=nn.Sequential(nn.Linear(in_size,32),nn.ReLU(),nn.Linear(32,out_size))

        self.params = self.model.parameters()
        self.optimizer = optim.SGD(self.params,lrate, weight_decay = 0.001)

        
        
        self.dropout = torch.nn.Dropout(0.1)


    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        #raise NotImplementedError("You need to write this part!")

        
        adjusted_data = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True))
        res_model=self.model(adjusted_data)
        return res_model

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        
        y_pred=self.forward(x)
        
        loss=self.loss_fn(y_pred,y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        L=loss.detach().cpu().numpy
        #raise NotImplementedError("You need to write this part!")
        return L
        


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

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """ 
    in_size=train_set.size()[1]
    out_size=4
    lrate=0.01
    L=0
    losses = []
    lossFunction=nn.CrossEntropyLoss()
    yhats= []
    net=NeuralNet(lrate,lossFunction,len(train_set[0]),out_size)

    
    
    varN = int(train_set.size()[0])
    vark = int(varN / batch_size)
    train_set = (train_set -  train_set.mean(0))/train_set.std(0)
    for j in range(epochs):

        a = torch.randperm(train_set.size()[0])
        train_set = train_set[a]
        train_labels = train_labels[a]

        for i in range(vark):
            loss = net.step(train_set[i * batch_size: min(batch_size * (1+i), len(train_set))], train_labels[i * batch_size: min(batch_size * (1+i ), len(train_set))])
        losses.append(loss)
    
    dev_set = (dev_set - dev_set.mean(0)) / dev_set.std(0)
    yhats = np.argmax(net.forward(dev_set).detach().cpu().numpy(), axis = 1)
    res=net(dev_set).detach().numpy()
    return losses,yhats,net




'''
    for i in range(epochs):
        if i >= 75:
            bdata = train_set[(i-75) * batch_size : (i-74) * batch_size]
            blabels = train_labels[(i-75) * batch_size : (i-74) * batch_size]
        else:
            bdata = train_set[i * batch_size : (i+1) * batch_size]
            blabels = train_labels[i * batch_size : (i+1) * batch_size]
        loss = net.step(bdata, blabels)
        losses.append(loss)
    num=net(dev_set).detach().numpy()
    for i in range(len(num)):
        yhats.append(np.argmax(num[i]))
    return losses, yhats, net

'''

    
    
    



   