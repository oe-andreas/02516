from tqdm import tqdm  # For progress bar
import torch

def train(model, data_loader, optimizer, scheduler, classification_loss_fn, bbox_loss_fn, epochs=10, device='cpu'):
    """
    Trains the EfficientNetWithBBox model.

    Parameters:
    - model: The EfficientNetWithBBox model to train.
    - data_loader: DataLoader providing batches of (X_batch, Y_batch, gt_bbox, t_batch). gt_bbox unused
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



def train_oe(model, train_loader, val_loader, optimizer, scheduler, combined_loss, epochs=10, device='cpu'):
    """
    Training function with evaluation on validation set after each epoch.
    """
    all_losses_train = []
    all_losses_val = []

    # Move model to the specified device
    model = model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0  # Track total loss for the epoch
        class_epoch_loss = 0  # Track class loss for the epoch
        bbox_epoch_loss = 0   # Track bbox regression loss for the epoch

        # Loop over training batches
        for X_batch, Y_batch, gt_bbox, t_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move data to device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device).float()  # Ensure target is float for BCE loss
            gt_bbox = gt_bbox.to(device)
            t_batch = t_batch.to(device)

            # Forward pass
            class_score, t_vals = model(X_batch)  # y_pred, t_pred

            # Compute combined loss
            total_loss, class_loss, bbox_loss = combined_loss(class_score.squeeze(), Y_batch, t_vals, t_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Track the total loss for this epoch
            total_epoch_loss += total_loss.item()
            class_epoch_loss += class_loss.item()
            bbox_epoch_loss += bbox_loss.item()

        # Store training losses for this epoch
        all_losses_train.append([total_epoch_loss, class_epoch_loss, bbox_epoch_loss])
        
        # Validation loop
        model.eval()
        val_total_loss = 0
        val_class_loss = 0
        val_bbox_loss = 0

        with torch.no_grad():  # No gradients needed for validation
            for X_val, Y_val, gt_bbox_val, t_val in val_loader:
                # Move data to device
                X_val = X_val.to(device)
                Y_val = Y_val.to(device).float()
                gt_bbox_val = gt_bbox_val.to(device)
                t_val = t_val.to(device)

                # Forward pass
                class_score_val, t_vals_val = model(X_val)

                # Compute validation loss
                val_loss, val_class, val_bbox = combined_loss(class_score_val.squeeze(), Y_val, t_vals_val, t_val)
                val_total_loss += val_loss.item()
                val_class_loss += val_class.item()
                val_bbox_loss += val_bbox.item()

        # Store validation losses for this epoch
        all_losses_val.append([val_total_loss, val_class_loss, val_bbox_loss])

        # Adjust learning rate
        scheduler.step(total_epoch_loss)

    return all_losses_train, all_losses_val
