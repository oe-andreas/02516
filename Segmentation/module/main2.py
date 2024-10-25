#################################################################################
# Initialization
#################################################################################
import torch
import numpy as np
import os
from PIL import Image
import glob
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .models.EncDec import EncDec
from .models.UNet import UNet, UNet_orig

from .train import train, train_weak_annotation
from .plot import plot_losses, plot_metrics, plot_predictions , plot_predictions_weak

from .losses.losses import bce_loss

from .dataloaders.PH2_loader import PH2, PH2_weak
from .dataloaders.retinal_loader import retinal


#################################################################################
# Load Data
#################################################################################

#Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

#set image size, batch size and number of Epoch
im_size = 128
batch_size = 32
epoch = 20

#Defines weak annotation sampling info:
num_of_annotations = 8 #number of point collected for each class in the weak annotation set
inbetween_dist = 0 # minimum Euclidean distance between points 
edge_dist = 0 # minimum euclidean distance every point has to the edge of the image.
sample_info = [num_of_annotations, inbetween_dist , edge_dist]

#Define image argumentation for test and training data.
train_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])

#Load PH2 Dataset
PH2_indeces = sorted([int(str[-3:]) for str in glob.glob('module/PH2_Dataset_images/IMD*')])
PH2_train = PH2_weak(sample_info=sample_info,indeces = PH2_indeces[:170], transform = train_transform, train = False)
PH2_test = PH2_weak(sample_info=sample_info,indeces = PH2_indeces[170:], transform = test_transform, train= False)
PH2_train_loader = DataLoader(PH2_train, batch_size=batch_size, shuffle=True)
PH2_test_loader = DataLoader(PH2_test, batch_size=batch_size, shuffle=False)

#################################################################################
# Training data on Encoder Decoder
#################################################################################

# Define optimizer and send model to device
model_EncDec = EncDec(im_size).to(device)
optimizer = torch.optim.Adam(model_EncDec.parameters(), lr=0.001)

#Train network:
train_losses, test_losses, observed_eval_metrics = train_weak_annotation(model_EncDec, device, optimizer, epoch, PH2_train_loader, PH2_test_loader)

## Plot results for Encoder Decoder
plot_losses(train_losses, test_losses, model_name='EncDec_weak')
plot_metrics(observed_eval_metrics, model_name='EncDec_weak')
plot_predictions_weak(model_EncDec, device, PH2_train_loader,num_of_annotations, model_name='EncDec_weak')

# Save model weights
torch.save(model_EncDec.state_dict(), 'Trained_models/EncDec.pth')


#################################################################################
# Training data on UNET
#################################################################################

# Define optimizer and send model to device
model_Unet = UNet(im_size).to(device)
optimizer = torch.optim.Adam(model_Unet.parameters(), lr=0.001)

#Train network:
train_losses, test_losses, observed_eval_metrics = train_weak_annotation(model_Unet, device, optimizer, epoch,PH2_train_loader, PH2_test_loader)

## Plot results for Unet
plot_losses(train_losses, test_losses, model_name='Unet_weak')
plot_metrics(observed_eval_metrics, model_name='Unet_weak')
plot_predictions_weak(model_Unet, device, PH2_train_loader,num_of_annotations, model_name='Unet_weak')

# Save model weights
torch.save(model_Unet.state_dict(), 'Trained_models/UNet.pth')