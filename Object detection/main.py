import torch
import torch.nn as nn
import torch.optim as optim

# Load local modules
from module.dataloaders.loader import load_images_fixed_batch
from module.models.efficientnet import EfficientNetWithBBox
from module.losses.losses import conditional_bbox_mse_loss
from train import train

# Initialize model and data loader
model = EfficientNetWithBBox(model_name='efficientnet_b5', num_classes=1, bbox_output_size=4, pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data loader
train_data_loader = load_images_fixed_batch(train=True, dim=[128, 128], batch_size=64)

# Define the loss functions
classification_loss_fn = nn.BCEWithLogitsLoss()  # Binary classification loss
bbox_loss_fn = conditional_bbox_mse_loss

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Initialize the scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Start training
train(
    model=model,
    data_loader=train_data_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    classification_loss_fn=classification_loss_fn,
    bbox_loss_fn=bbox_loss_fn,
    epochs=5,
    device=device
)

torch.save(model.state_dict(), f'Trained_models/model.pth')