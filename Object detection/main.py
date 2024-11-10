print("Loading packages")

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from time import time

# Load local modules
from module.dataloaders.loader import load_images_fixed_batch
from module.models.efficientnet import EfficientNetWithBBox
from module.losses.losses import conditional_bbox_mse_loss, MultiTaskLoss
from train import train, train_oe

print("Creating TIMM model")
t = time()
# Initialize model and data loader
model = EfficientNetWithBBox(model_name='efficientnet_b5', num_classes=1, bbox_output_size=4, pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Created TIMM model in {time() - t}")
print("Initialize data loader")
t = time()


# Load data loader
train_data_loader = load_images_fixed_batch(train=True, dim=[128, 128], batch_size=64)

print(f"Initialized Data Loader in {time() - t}")
print("Define loss etc")
t = time()

# Define the loss functions
#classification_loss_fn = nn.BCEWithLogitsLoss()  # Binary classification loss
#bbox_loss_fn = conditional_bbox_mse_loss

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Initialize the scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Start training
combined_loss = MultiTaskLoss()

print(f"Defined loss etc in {time() - t}")
print(f"Train")

train_oe(
    model=model,
    data_loader=train_data_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    #classification_loss_fn=classification_loss_fn,
    #bbox_loss_fn=bbox_loss_fn,
    loss = combined_loss,
    epochs=5,
    device=device
)

current_time = datetime.now().strftime("%Y%m%d_%H%M")


print("Saving model")
t = time()
torch.save(model.state_dict(), f'Trained_models/model_{current_time}.pth')
print(f"Saved model in {time() - t}")
