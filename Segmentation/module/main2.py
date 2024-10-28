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

from .models.EncDec import EncDec, EncDec_reg
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
im_size = 256
batch_size = 32
epoch = 1

#Defines weak annotation sampling info:
num_of_annotations = [5,10,15,20] #number of point collected for each class in the weak annotation set
inbetween_dist = 5 # minimum Euclidean distance between points 
edge_dist = 10 # minimum euclidean distance every point has to the edge of the image.

#Define image argumentation for test and training data.
train_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])

for n in range(4):
    #Load PH2 Dataset
    PH2_indeces = sorted([int(str[-3:]) for str in glob.glob('/dtu/datasets1/02516/PH2_Dataset_images/IMD*')])
    PH2_train = PH2_weak(sample_info=sample_info,indeces = PH2_indeces[:170], transform = train_transform, train = False)
    PH2_test = PH2_weak(sample_info=sample_info,indeces = PH2_indeces[170:], transform = test_transform, train= False)
    PH2_train_loader = DataLoader(PH2_train, batch_size=batch_size, shuffle=True)
    PH2_test_loader = DataLoader(PH2_test, batch_size=batch_size, shuffle=False)

    #################################################################################
    # Training data on Encoder Decoder
    #################################################################################

    #################################################################################
    # Training data on Encoder Decoder
    #################################################################################

    # Define optimizer and send model to device
    model_EncDec = EncDec_reg(im_size).to(device)
    optimizer = torch.optim.Adam(model_EncDec.parameters(), lr=0.001, weight_decay=0.0005)
    
    #Train network:
    train_losses, test_losses, observed_eval_metrics = train_weak_annotation(model_EncDec, device, optimizer, epoch, PH2_train_loader, PH2_test_loader)

    ## Plot results for Encoder Decoder
    plot_losses(train_losses, test_losses, model_name='EncDec_weak_{}'.format(num_of_annotations[n]))
    plot_metrics(observed_eval_metrics, model_name='EncDec_weak_{}'.format(num_of_annotations[n]))
    plot_predictions_weak(model_EncDec, device, PH2_test_loader,num_of_annotations[n], model_name='EncDec_weak_{}'.format(num_of_annotations[n]))


#################################################################################
# Training data on UNET
#################################################################################
for n in range(4):
    # Define optimizer and send model to device
    model_EncDec = UNet_orig(im_size).to(device)
    optimizer = torch.optim.Adam(model_EncDec.parameters(), lr=0.001, weight_decay=0.0005)
    
    #Train network:
    train_losses, test_losses, observed_eval_metrics = train_weak_annotation(model_Unet, device, optimizer, epoch,PH2_train_loader, PH2_test_loader)

    ## Plot results for Unet
    plot_losses(train_losses, test_losses,dataset_name="PH2", model_name='Unet_weak_{}'.format(num_of_annotations[n]))
    plot_metrics(observed_eval_metrics,dataset_name="PH2", model_name='Unet_weak_{}'.format(num_of_annotations[n]))
    plot_predictions_weak(model_Unet, device, PH2_train_loader,num_of_annotations[n], model_name='Unet_weak_{}'.format(num_of_annotations[n]))

    # Save model weights
    torch.save(model_Unet.state_dict(), 'Trained_models/UNet.pth')