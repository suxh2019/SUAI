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

