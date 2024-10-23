import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import glob
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .models.EncDec import EncDec
from .models.UNet import UNet

from .losses.losses import bce_loss, dice_loss, iou_loss, focal_loss, bce_total_variation, accuracy, sensitivity, specificity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

## Dataloaders
from .dataloaders.PH2_loader import PH2
from .dataloaders.retinal_loader import retinal


#a second comment from Ø

im_size = 128
batch_size = 32

train_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])

####DRIVE####
retinal_train = retinal(indeces = np.arange(21,33), transform = train_transform)
retinal_test = retinal(indeces = np.arange(33,41), transform = test_transform)

retinal_train_loader = DataLoader(retinal_train, batch_size=batch_size, shuffle=True)
retinal_test_loader = DataLoader(retinal_test, batch_size=batch_size, shuffle=False)

####PH2####
PH2_indeces = sorted([int(str[-3:]) for str in glob.glob('/dtu/datasets1/02516/PH2_Dataset_images/IMD*')])

PH2_train = PH2(indeces = PH2_indeces[:170], transform = train_transform)
PH2_test = PH2(indeces = PH2_indeces[170:], transform = test_transform)

PH2_train_loader = DataLoader(PH2_train, batch_size=batch_size, shuffle=True)
PH2_test_loader = DataLoader(PH2_test, batch_size=batch_size, shuffle=False)

## TRAIN UNET
def train(model, opt, loss_fn, epochs, train_loader, test_loader):
    
    eval_metrics = [dice_loss, iou_loss, accuracy, sensitivity, specificity]
    
    observed_eval_metrics = []
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        # Training
        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss.detach().cpu() / len(train_loader)
        print(' - loss: %f' % avg_loss)
        train_losses.append(avg_loss)

        # Testing
        avg_loss = 0
        avg_eval_metrics = []
        model.eval()
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            with torch.no_grad():
                Y_pred = model(X_batch)
                loss = loss_fn(Y_batch, Y_pred)

            avg_loss += loss.detach().cpu() / len(test_loader)
            
    
            for eval_metric in eval_metrics:
                avg_eval_metrics.append(eval_metric(Y_batch, Y_pred).cpu() / len(test_loader))
        
        observed_eval_metrics.append(avg_eval_metrics)
        print(' - val_loss: %f' % avg_loss)
        test_losses.append(avg_loss)
    
    return train_losses, test_losses, observed_eval_metrics


model = UNet(im_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_loader = PH2_train_loader
test_loader =  PH2_test_loader

train_losses, test_losses, observed_eval_metrics = train(model, optimizer, bce_loss, 30, train_loader, test_loader)


#plot history
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.savefig('graphics/loss_history.png')

observed_eval_metrics = np.array(observed_eval_metrics)
plt.plot(observed_eval_metrics[:,0], label='dice')
plt.plot(observed_eval_metrics[:,1], label='iou')
plt.plot(observed_eval_metrics[:,2], label='accuracy')
plt.plot(observed_eval_metrics[:,3], label='sensitivity')
plt.plot(observed_eval_metrics[:,4], label='specificity')
plt.legend()
plt.savefig('graphics/score_history.png')

# visualize predictions
to_im_shape = lambda x : x.permute(1,2,0).numpy()

model.eval()
X_batch, y_batch = next(iter(train_loader))
X, y = X_batch[0], y_batch[0]
X = X.unsqueeze(0).to(device)
y_pred = model(X)
y_pred = torch.sigmoid(y_pred).detach().cpu().numpy().squeeze()

fig, axs = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})

#Input image
axs[0].imshow(to_im_shape(X.cpu().squeeze()))

# Ground truth
axs[1].imshow(to_im_shape(y), vmin=0, vmax=1)
axs[1].set_title('Ground Truth')

# Prediction
im = axs[2].imshow(y_pred, vmin=0, vmax=1)
axs[2].set_title('Prediction')

# Colorbar
fig.colorbar(im, cax=axs[3])

plt.tight_layout()

plt.savefig('graphics/predictions.png')


