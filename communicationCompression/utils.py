# Design and Programming by Mojtaba Valipour @ Data Analytics Lab - UWaterloo # 2020

# COAUTHORS:

''' Resources:
-https://github.com/lightonai/supervised_random_projections
'''

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def setSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def relu(X):
   return F.relu(X) #np.maximum(0,X)

def sigmoid(X):
   #TODO: fix the overflow problem
   return F.sigmoid(X) #1/(1+np.exp(-X))

def poly2(X):
    return X * X

# count number of parameters in pytorch models
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#TODO: check if we really need this function
def scatter_plot(X, y, ax=None, s=5, title=None, xlabel=None, ylabel=None, fontsize=12, axis_off=True):
    if ax is None:
        fig, ax = plt.subplots()
    for c in np.unique(y):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=s)
    if title:
        ax.set_title(title, fontsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if axis_off:
        ax.set_axis_off()