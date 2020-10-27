# Loads as input a fully trained en5resnet20 model, and returns the output
# state of each layer in the model

#Author: Alexa Aucoin
#University of Arizona
#Applied Math


import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn
import math

import os
import importlib

import matplotlib.pyplot as plt
import numpy as np
import re

from resnet_cifar import *
from en5resnet20_clean import *

def hook(module, input, output):
    setattr(module, "_hook_value", output)


#---------------------
#Load trained model:
#---------------------

m = torch.load('ckpt_PGD_ensemble_5_20.t7', map_location='cpu')

#m.keys()
acc = m['acc']
ep = m['epoch']
model = m['net']

#---------------------
#View model layout:
#---------------------

#model

#------------------------
# Load Cifar data
#------------------------
print('==> Preparing data...')
root = './data'
download = True

train_transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    ])

train_set = torchvision.datasets.CIFAR10(
    root=root,
    train=True,
    download=download,
    transform = train_transform
    )

test_set = torchvision.datasets.CIFAR10(
    root=root,
    train=False,
    download=download,
    transform=test_transform
    )
#----------------------
# Register hooks
#----------------------

# to view model layers:
# for idx, layer in enumerate(model.named_modules()):
#   print(idx,'-->',layer)
num_hooks = 0
for idx, layer in enumerate(model.named_modules()):
    if re.search('ensemble\.0\.layer.\..\.bn2',layer[0]):
        layer[1].register_forward_hook(hook)
        num_hooks += 1


#--------------------------------
# Get intermediate output states
#--------------------------------
test_set_len = 101 #len(test_set)
test_loader = torch.utils.data.DataLoader(dataset=[test_set[i] for i in range(test_set_len)],
                                    batch_size=1, shuffle=False)


# block_states will end up size [len(test_loader),num_hooks,2]
# the 2 coming from tuple (layer_name, hook_value)

block_out_states = []

for batch_idx, (input,target) in enumerate(test_loader):
    model(input)
    int_output = []
    for idx, layer in enumerate(model.named_modules()):
        if re.search('ensemble\.0\.layer.\..\.bn2',layer[0]):
            int_output.append((layer[0],layer[1]._hook_value))

    block_out_states.append(int_output)


#------------
# Consistency check
#------------

#We check size consistency of outputs to the expect layer output size

for i in range(num_hooks):
    output = block_out_states[0][i][1]
    print(output.shape)
