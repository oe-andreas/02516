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
from .models.UNet import UNet, UNet_orig, UNet_copy

from .train2 import train
from .plot import plot_losses, plot_metrics, plot_predictions

from .losses.losses import bce_weighted, bce_loss, focal_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

## Dataloaders
from .dataloaders.PH2_loader import PH2
from .dataloaders.retinal_loader import retinal

im_size = 256
batch_size = 32

train_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])

####DRIVE####
def calculate_mean_std_with_mask(dataloader):
    """Calculate mean and std of images in a dataloader using the provided masks."""
    mean = 0.0
    std = 0.0
    count = 0

    for images, _, masks in dataloader:
        batch_samples = images.size(0)  # number of images in the batch
        images = images.view(batch_samples, images.size(1), -1)  # Reshape to (batch, channels, H*W)
        
        # Reshape masks to the same dimensions as images (if necessary)
        masks = masks.view(batch_samples, 1, -1)  # Reshape to (batch, 1, H*W)

        # Masking: Only consider pixels where the mask is True (or > 0)
        masked_images = images * masks
        
        # Calculate mean and std
        mean += masked_images.sum(2).sum(0) / masks.sum(2).sum(0)  # Sum only over masked pixels
        std += ((masked_images - mean.unsqueeze(0).unsqueeze(2)) ** 2).sum(2).sum(0) / masks.sum(2).sum(0)
        count += batch_samples  # Count number of batches

    mean /= count
    std = torch.sqrt(std / count)  # Calculate final std

    return mean, std
retinal_train_no_transform = retinal(indeces = np.arange(21,33), transform = train_transform, train = False)
retinal_train_no_transform_loader = DataLoader(retinal_train_no_transform, batch_size=batch_size, shuffle=True)
# Calculate mean and std using the mask
mean, std = calculate_mean_std_with_mask(retinal_train_no_transform_loader)
normalize_params = (mean, std)
retinal_train = retinal(indeces = np.arange(21,33), transform = train_transform, normalize=normalize_params, train = True)
retinal_test = retinal(indeces = np.arange(33,41), transform = test_transform, normalize=normalize_params, train = False)

retinal_train_loader = DataLoader(retinal_train, batch_size=batch_size, shuffle=True)
retinal_test_loader = DataLoader(retinal_test, batch_size=batch_size, shuffle=False)

####PH2####
PH2_indeces = sorted([int(str[-3:]) for str in glob.glob('/dtu/datasets1/02516/PH2_Dataset_images/IMD*')])

PH2_train = PH2(indeces = PH2_indeces[:170], transform = train_transform, train = True)
PH2_test = PH2(indeces = PH2_indeces[170:], transform = test_transform, train= False)

PH2_train_loader = DataLoader(PH2_train, batch_size=batch_size, shuffle=True)
PH2_test_loader = DataLoader(PH2_test, batch_size=batch_size, shuffle=False)

# Define the loaders and their corresponding dataset names
loaders = [
    (retinal_train_loader, retinal_test_loader, "Retinal"),
    (PH2_train_loader, PH2_test_loader, "PH2")
]

losses = [(bce_weighted, 'bce_weighted')]

## Training for both datasets
for train_loader, test_loader, dataset_name in loaders:
    for loss, loss_name in losses:
        ## Full UNet
        model_Unet_orig = UNet_copy(im_size).to(device)
        optimizer = torch.optim.Adam(model_Unet_orig.parameters(), lr=0.001, weight_decay=1e-5)
        # Initialize the scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # Train model
        train_losses, test_losses, observed_eval_metrics = train(model_Unet_orig, device, optimizer, scheduler, loss, 30, train_loader, test_loader)

        ## Plot results for Unet
        plot_losses(train_losses, test_losses, dataset_name, model_name='Unet_orig_'+loss_name)
        plot_metrics(observed_eval_metrics, dataset_name, model_name='Unet_orig_'+loss_name)
        plot_predictions(model_Unet_orig, device, train_loader, dataset_name, model_name='Unet_orig_'+loss_name)

        # Save model weights
        torch.save(model_Unet_orig.state_dict(), f'Trained_models/Unet_orig_{loss_name}.pth')

        ## Encoder Decoder
        model_EncDec = EncDec(im_size).to(device)
        optimizer = torch.optim.Adam(model_EncDec.parameters(), lr=0.001, weight_decay=1e-5)
        # Initialize the scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        # Train model
        train_losses, test_losses, observed_eval_metrics = train(model_EncDec, device, optimizer, scheduler, loss, 30, train_loader, test_loader)

        ## Plot results for Encoder Decoder
        plot_losses(train_losses, test_losses, dataset_name, model_name='EncDec_'+loss_name)
        plot_metrics(observed_eval_metrics, dataset_name, model_name='EncDec_'+loss_name)
        plot_predictions(model_EncDec, device, train_loader, dataset_name, model_name='EncDec_'+loss_name)

        # Save model weights
        torch.save(model_EncDec.state_dict(), f'Trained_models/EncDec_{loss_name}.pth')

        
