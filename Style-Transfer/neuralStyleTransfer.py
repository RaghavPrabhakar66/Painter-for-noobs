import os

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torch import cuda
from torch.cuda import is_available
from torchvision import *

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

assert device == 'cuda:0'

model = models.vgg19(pretrained=True).features

for param in model.parameters():
    param.requires_grad = False

model.to(device)

print(model)