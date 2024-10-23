print("hello")

import torch
import numpy as np
import os
from PIL import Image
import glob
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .models.UNet import UNet_orig

from .train import train
from .plot import plot_losses, plot_metrics, plot_predictions

from .losses.losses import bce_loss

print("hello")

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f'Device: {device}')

## Dataloaders
from .dataloaders.PH2_loader import PH2

im_size = 572
batch_size = 32

train_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])

####PH2####
PH2_indeces = sorted([int(str[-3:]) for str in glob.glob('/dtu/datasets1/02516/PH2_Dataset_images/IMD*')])

PH2_train = PH2(indeces = PH2_indeces[:170], transform = train_transform)
PH2_test = PH2(indeces = PH2_indeces[170:], transform = test_transform)

PH2_train_loader = DataLoader(PH2_train, batch_size=batch_size, shuffle=True)
PH2_test_loader = DataLoader(PH2_test, batch_size=batch_size, shuffle=False)


model = UNet_orig(im_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = PH2_train_loader
test_loader =  PH2_test_loader

print("hello")

train_losses, test_losses, observed_eval_metrics = train(model, device, optimizer, bce_loss, 30, train_loader, test_loader)


