from tqdm import tqdm  # For progress bar

def train(model, data_loader, optimizer, scheduler, classification_loss_fn, bbox_loss_fn, epochs=10, device='cpu'):
    """
    Trains the EfficientNetWithBBox model.

    Parameters:
    - model: The EfficientNetWithBBox model to train.
    - data_loader: DataLoader providing batches of (X_batch, Y_batch, gt_bbox, t_batch).
    - classification_loss_fn: Loss function for classification.
    - bbox_loss_fn: Loss function for bounding box regression.
    - epochs: Number of training epochs.
    - scheduler: Learning rate scheduler (e.g., ReduceLROnPlateau).
    - device: Device to run training on ('cpu' or 'cuda').
    """
    # Move model to the specified device
    model = model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0  # Track total loss for the epoch

        # Loop over batches
        for X_batch, Y_batch, gt_bbox, t_batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move data to device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device).float()  # Ensure target is float for BCE loss
            gt_bbox = gt_bbox.to(device)
            t_batch = t_batch.to(device)

            # Forward pass
            class_score, t_vals = model(X_batch)

            # Compute losses
            class_loss = classification_loss_fn(class_score.squeeze(), Y_batch)
            bbox_loss = bbox_loss_fn(t_vals, t_batch, Y_batch)

            # Combine losses (you could tune the weighting between classification and bbox losses)
            total_loss = class_loss + bbox_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track the total loss for this epoch
            epoch_loss += total_loss.item()

        # Optionally, print the epoch's total loss
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(data_loader)}")

        # Reduce LR based on the total loss for the epoch
        scheduler.step(epoch_loss)

