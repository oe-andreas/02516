from tqdm import tqdm  # For progress bar
import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='best_model.pth'):
        self.patience = patience  # Number of epochs to wait before stopping
        self.min_delta = min_delta  # Minimum change to qualify as improvement
        self.best_score = None  # Track best score so far
        self.epochs_no_improve = 0  # Counter for non-improving epochs
        self.early_stop = False  # Flag to trigger early stopping
        self.path = path  # Path to save the best model

    def __call__(self, val_loss, model):
        # Initialize best_score if it's the first call
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_score - self.min_delta:
            # Improvement found; reset counter and save model
            self.best_score = val_loss
            self.epochs_no_improve = 0
            self.save_checkpoint(model)
        else:
            # No improvement; increment counter
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Save the best model checkpoint."""
        torch.save(model.state_dict(), self.path)

def train(model, train_loader, val_loader, optimizer, scheduler, combined_loss, epochs=10, device='cpu'):
    """
    Training function with evaluation on validation set after each epoch.
    """
    all_losses_train = []
    all_losses_val = []

    # Move model to the specified device
    model = model.to(device)

    # Initialize EarlyStopping instance
    early_stopping = EarlyStopping(patience=5, min_delta=0, path='best_model.pth')

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0  # Track total loss for the epoch
        class_epoch_loss = 0  # Track class loss for the epoch
        bbox_epoch_loss = 0   # Track bbox regression loss for the epoch
        n = 0
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
            n += 1
        # Noramlize loss
        total_epoch_loss /= n
        class_epoch_loss /= n
        bbox_epoch_loss /= n

        # Store training losses for this epoch
        all_losses_train.append([total_epoch_loss, class_epoch_loss, bbox_epoch_loss])
        
        # Validation loop
        model.eval()
        val_total_loss = 0
        val_class_loss = 0
        val_bbox_loss = 0
        n = 0
        with torch.no_grad():  # No gradients needed for validation
            for X_val, Y_val, gt_bbox_val, t_batch_val in val_loader:
                # Move data to device
                X_val = X_val.to(device)
                Y_val = Y_val.to(device).float()
                gt_bbox_val = gt_bbox_val.to(device)
                t_batch_val = t_batch_val.to(device)

                # Forward pass
                class_score_val, t_vals_val = model(X_val)

                # Compute validation loss
                val_loss, val_class, val_bbox = combined_loss(class_score_val.squeeze(), Y_val, t_vals_val, t_batch_val)
                val_total_loss += val_loss.item()
                val_class_loss += val_class.item()
                val_bbox_loss += val_bbox.item()
                n += 1
        
        # Noramlize loss
        val_total_loss /= n
        val_class_loss /= n
        val_bbox_loss /= n
        # Store validation losses for this epoch
        all_losses_val.append([val_total_loss, val_class_loss, val_bbox_loss])

        # Check for early stopping
        early_stopping(val_total_loss, model)
    
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Load the best model after training
        model.load_state_dict(torch.load('best_model.pth'))

        # Adjust learning rate
        scheduler.step(total_epoch_loss)

    return all_losses_train, all_losses_val
