# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020
import numpy as np
import math
import copy
"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""
from collections import Counter


def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    weight = np.zeros(len(train_set[1]))
    weight = np.append(weight,0)
    for i in range(max_iter):
        for j in range(len(train_set)):
            feature = np.append(train_set[j],1)
            res=np.dot(weight,feature)
            if(res>0):
                res=1
            else:
                res=0
            if train_labels[j] != res:
                if train_labels[j] == 0:
                    label = -1
                else:
                    label = 1
                weight = np.add(weight, feature*learning_rate*label)
    return weight[:-1], weight[-1]

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    
    weight,bias = trainPerceptron(train_set,train_labels,learning_rate,max_iter)
    model = np.array([])

    weight = np.append(weight,bias)
    
    for f in dev_set:
        f = np.append(f,[1])
        res=np.dot(weight,f)
        if(res>0):
            model = np.append(model,1)
        else:
            model = np.append(model,0)
    python_list=list(model)
    return python_list

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    result = [True for i in range(len(dev_set))]

    for dx, dy in enumerate(dev_set):
        d_list = []
        for tx, ty in enumerate(train_set):
            ##calculate distance
            dist = np.linalg.norm(ty - dy)
            d_list.append((dist, train_labels[tx]))

        d_list.sort()
        klist = [i[1] for i in d_list[:k]]
        c = Counter(klist)
        if  c[False]>= c[True]:
            result[dx] = False
        else:
            result[dx] = True


    return result

