# The code is from the video, which has the link https://www.youtube.com/watch?v=PXOzkkB5eH0&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=9
# epoch = 1 forward and backward pass of all training samples
#batch_size = number of traning samples in in one forward and backward pass
#number of iterations = number of passes, each pass using [batch_size] number of samples
#100 samples, batch_size = 20 --> 100/20 =5 iterations for 1 epoch

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn as nn

# or a 1D array, axis=0 means sum over the elements of the array
# axis=0 sums vertically (down each column)
# axis=1 sums horizontally (across each row)
def softmax(x):
    return np.exp(x) /np.sum(np.exp(x),axis = 0)
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss
y = np.array([1,0,0])
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(y,y_pred_bad)
l2 = cross_entropy(y,y_pred_good)
print(f'Loss1 numpy:{l1:.4f}')

# nn.CrossEntropyLoss() internally applies 
# log-softmax to the raw logits and then computes the negative log-likelihood.
loss = nn.CrossEntropyLoss()
# This defines the true class label.
y = torch.tensor([0])
y_pred_good = torch.tensor([0.7, 0.2, 0.1])
y_pred_bad = torch.tensor([0.1, 0.3, 0.6])
l1 = loss(y_pred_bad,y)
l2 = loss(y_pred_good,y)
print(f'Loss1 torch:{l1.item()}')