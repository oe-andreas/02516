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
from .dataloaders.PH2_loader import PH2
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

####PH2####
PH2_indeces = sorted([int(str[-3:]) for str in glob.glob('/dtu/datasets1/02516/PH2_Dataset_images/IMD*')])

PH2_train = PH2(indeces = PH2_indeces[:170], transform = train_transform, train = True)
PH2_test = PH2(indeces = PH2_indeces[170:], transform = test_transform, train= False)

PH2_train_loader = DataLoader(PH2_train, batch_size=batch_size, shuffle=True)
PH2_test_loader = DataLoader(PH2_test, batch_size=batch_size, shuffle=False)

# Define the loaders and their corresponding dataset names
loaders = [
    (PH2_train_loader, PH2_test_loader, "PH2"),
    (retinal_train_loader, retinal_test_loader, "Retinal")
]

## Training for both datasets
for train_loader, test_loader, dataset_name in loaders:
    ## TRAIN Encoder Decoder
    model_EncDec = EncDec(im_size).to(device)
    optimizer = torch.optim.Adam(model_EncDec.parameters(), lr=0.001)

    train_losses, test_losses, observed_eval_metrics = train(model_EncDec, device, optimizer, bce_loss, 30, train_loader, test_loader)

    ## Plot results for Encoder Decoder
    plot_losses(train_losses, test_losses, dataset_name, model_name='EncDec')
    plot_metrics(observed_eval_metrics, dataset_name, model_name='EncDec')
    plot_predictions(model_EncDec, device, train_loader, dataset_name, model_name='EncDec')

    # Save model weights
    torch.save(model_EncDec.state_dict(), 'Trained_models/EncDec.pth')

    ## TRAIN UNET
    model_Unet = UNet(im_size).to(device)
    optimizer = torch.optim.Adam(model_Unet.parameters(), lr=0.001)

    train_losses, test_losses, observed_eval_metrics = train(model_Unet, device, optimizer, bce_loss, 30, train_loader, test_loader)

    ## Plot results for Unet
    plot_losses(train_losses, test_losses, dataset_name, model_name='Unet')
    plot_metrics(observed_eval_metrics, dataset_name, model_name='Unet')
    plot_predictions(model_Unet, device, train_loader, dataset_name,  model_name='Unet')

    # Save model weights
    torch.save(model_Unet.state_dict(), 'Trained_models/UNet.pth')

    ## TRAIN FULL UNET
    model_Unet_orig = UNet_orig(im_size).to(device)
    optimizer = torch.optim.Adam(model_Unet_orig.parameters(), lr=0.001)

    train_losses, test_losses, observed_eval_metrics = train(model_Unet_orig, device, optimizer, bce_loss, 30, train_loader, test_loader)

    ## Plot results for Unet
    plot_losses(train_losses, test_losses, model_name='Unet_orig')
    plot_metrics(observed_eval_metrics, model_name='Unet_orig')
    plot_predictions(model_Unet, device, train_loader, model_name='Unet_orig')

    # Save model weights
    torch.save(model_Unet.state_dict(), 'Trained_models/UNet.pth')

    ## TRAIN FULL UNET
    model_Unet_orig = UNet_orig(im_size).to(device)
    optimizer = torch.optim.Adam(model_Unet_orig.parameters(), lr=0.001)

<<<<<<< HEAD
    train_loader = PH2_train_loader
    test_loader =  PH2_test_loader

    train_losses, test_losses, observed_eval_metrics = train(model_Unet_orig, device, optimizer, bce_loss, 30, train_loader, test_loader)

    ## Plot results for Unet
    plot_losses(train_losses, test_losses, model_name='Unet_orig')
    plot_metrics(observed_eval_metrics, model_name='Unet_orig')
    plot_predictions(model_Unet, device, train_loader, model_name='Unet_orig')

    # Save model weights
    torch.save(model_Unet.state_dict(), 'Trained_models/Unet_orig.pth')
=======
## Plot results for Unet
plot_losses(train_losses, test_losses, model_name='Unet_orig')
plot_metrics(observed_eval_metrics, model_name='Unet_orig')
plot_predictions(model_Unet_orig, device, train_loader, model_name='Unet_orig')

# Save model weights
torch.save(model_Unet_orig.state_dict(), 'Trained_models/Unet_orig.pth')
>>>>>>> d9e7c673eacc0cfcebbe46aedc5574015ec1890d
