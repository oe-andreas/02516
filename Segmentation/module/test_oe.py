import torch
import numpy as np
import os
from PIL import Image
import glob
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .models.EncDec import EncDec
from .models.UNet import UNet, UNet_orig

from .train import train
from .plot import plot_losses, plot_metrics, plot_predictions

from .losses.losses import bce_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

## Dataloaders
from .dataloaders.retinal_loader import retinal

im_size = 256
batch_size = 32

train_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])

####DRIVE####
retinal_train = retinal(indeces = np.arange(21,33), transform = train_transform, train = True)
retinal_test = retinal(indeces = np.arange(33,41), transform = test_transform, train = False)

retinal_train_loader = DataLoader(retinal_train, batch_size=batch_size, shuffle=True)
retinal_test_loader = DataLoader(retinal_test, batch_size=batch_size, shuffle=False)

## TRAIN FULL UNET
model_Unet_orig = UNet_orig(im_size).to(device)
optimizer = torch.optim.Adam(model_Unet_orig.parameters(), lr=0.001)

train_loader = PH2_train_loader
test_loader =  PH2_test_loader

train_losses, test_losses, observed_eval_metrics = train(model_Unet_orig, device, optimizer, bce_loss, 30, train_loader, test_loader)

## Plot results for Unet
plot_losses(train_losses, test_losses, model_name='Unet_orig')
plot_metrics(observed_eval_metrics, model_name='Unet_orig')
plot_predictions(model_Unet_orig, device, train_loader, model_name='Unet_orig')

# Save model weights
torch.save(model_Unet_orig.state_dict(), 'Trained_models/Unet_orig.pth')