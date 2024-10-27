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

from .train2 import train
from .plot import plot_losses, plot_metrics, plot_predictions, plot_all_metrics
from .utils import calculate_mean_std, calculate_mean_std_with_mask
from .losses.losses import bce_weighted, bce_loss, focal_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

PH2_train_no_transform = PH2(indeces = PH2_indeces[:170], transform = train_transform, train = True)
PH2_train_no_transform_loader = DataLoader(PH2_train_no_transform, batch_size=batch_size, shuffle=True)

# Calculate mean and std using the mask
mean, std = calculate_mean_std(retinal_train_no_transform_loader)
normalize_params = (mean, std)

PH2_train = PH2(indeces = PH2_indeces[:170], transform = train_transform,normalize=normalize_params, train = True,)
PH2_test = PH2(indeces = PH2_indeces[170:], transform = test_transform,normalize=normalize_params, train= False)

PH2_train_loader = DataLoader(PH2_train, batch_size=batch_size, shuffle=True)
PH2_test_loader = DataLoader(PH2_test, batch_size=batch_size, shuffle=False)

# Define the loaders and their corresponding dataset names
loaders = [
    (retinal_train_loader, retinal_test_loader, "Retinal"),
    (PH2_train_loader, PH2_test_loader, "PH2")
]

losses = [(bce_weighted,'bce_weighted'), (bce_loss, 'bce_loss'), (focal_loss, 'focal_loss')]

## Training for both datasets
for dataset_i, (train_loader, test_loader, dataset_name) in enumerate(loaders):
    
    #num_losses, num_models, num_splits, num_metrics
    all_final_observed_metrics = np.zeros((len(losses), 2, 2, 8))
    
    for loss_i, (loss, loss_name) in enumerate(losses):
        
        ## Full UNet
        model_Unet = UNet(im_size).to(device)
        optimizer = torch.optim.Adam(model_Unet.parameters(), lr=0.001, weight_decay=1e-5)
        # Initialize the scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # Train model
        train_losses, test_losses, observed_eval_metrics_train, observed_eval_metrics_test = train(model_Unet, device, optimizer, scheduler, loss, 30, train_loader, test_loader)
        
        all_final_observed_metrics[loss_i, 0, 0, :] = np.array(observed_eval_metrics_train[-1])
        all_final_observed_metrics[loss_i, 0, 1, :] = np.array(observed_eval_metrics_test[-1])
        
        ## Plot results for Unet
        plot_losses(train_losses, test_losses, dataset_name, model_name='Unet_'+loss_name)
        plot_metrics(observed_eval_metrics_test, dataset_name, model_name='Unet_'+loss_name)
        plot_predictions(model_Unet, device, test_loader, dataset_name, model_name='Unet_'+loss_name)

        # Save model weights
        torch.save(model_Unet.state_dict(), f'Trained_models/Unet_{loss_name}.pth')

        ## Encoder Decoder
        model_EncDec = EncDec(im_size).to(device)
        optimizer = torch.optim.Adam(model_EncDec.parameters(), lr=0.001, weight_decay=1e-5)
        # Initialize the scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        # Train model
        train_losses, test_losses, observed_eval_metrics_train, observed_eval_metrics_test = train(model_EncDec, device, optimizer, scheduler, loss, 30, train_loader, test_loader)

        all_final_observed_metrics[loss_i, 1, 0, :] = np.array(observed_eval_metrics_train[-1])
        all_final_observed_metrics[loss_i, 1, 1, :] = np.array(observed_eval_metrics_test[-1])
        
        ## Plot results for Encoder Decoder
        plot_losses(train_losses, test_losses, dataset_name, model_name='EncDec_'+loss_name)
        plot_metrics(observed_eval_metrics_test, dataset_name, model_name='EncDec_'+loss_name)
        plot_predictions(model_EncDec, device, train_loader, dataset_name, model_name='EncDec_'+loss_name)

        # Save model weights
        torch.save(model_EncDec.state_dict(), f'Trained_models/EncDec_{loss_name}.pth')
    
    plot_all_metrics(all_final_observed_metrics, dataset_name=dataset_name,
                     loss_labels = [loss_name for (_, loss_name) in losses],
                     model_labels = ['UNet', 'EncDec'],
                     split_labels = ['Train', 'Test'],
                     metric_labels = ["Dice", "IOU", "Accuracy", "Sensitivity", "Specificity", "BCE_w", "BCE", "Focal"])
        
    