print("Loading packages")

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from time import time

# Load local modules
from module.dataloaders.loader import Dataloader
from module.models.efficientnet import EfficientNetWithBBox
from module.losses.losses import MultiTaskLoss
from train import train
from plots import plot_losses, plot_steps
from utils import get_input_size

import pickle

print("Creating TIMM model")
t = time()
# Initialize model and data loader
model_name = 'efficientnet_b0'

model = EfficientNetWithBBox(model_name=model_name, num_classes=1, bbox_output_size=4, pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Created TIMM model in {time() - t:.2}s")
print("Initialize data loader")
t = time()


input_size = get_input_size(model_name)
train_loader = Dataloader(train="test", dim=[input_size, input_size], batch_size=64)
val_loader = Dataloader(train="val", dim=[input_size, input_size], batch_size=64)

print(f"Initialized Data Loader in {time() - t:.2}s")
print("Define loss etc")
t = time()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Initialize the scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Start training
combined_loss = MultiTaskLoss()

print(f"Defined loss etc in {time() - t:.2}s")
print(f"Train")

hist = train(
                                   model=model,
                                   train_loader=train_loader,
                                   val_loader=val_loader,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   combined_loss = combined_loss,
                                   epochs=2,
                                   device=device,
                                   print_memory_usage = True,
                                   return_losses_dict = True
                                )

all_losses_train, all_losses_val = hist['train_loss'], hist['val_loss']

plot_losses(all_losses_train, all_losses_val)

current_time = datetime.now().strftime("%Y%m%d_%H%M")

pickle.dump(hist, open(f"dumps/all_losses_{current_time}.pkl", "wb"))

print("Saving model")

# Extract the shortened model name (e.g., 'b0')
short_model_name = model_name.split('_')[-1]
t = time()
torch.save(model.state_dict(), f'Trained_models/{short_model_name}_model_{current_time}.pth')
print(f"Saved model in {time() - t:.2}s")
