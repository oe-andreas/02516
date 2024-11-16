from tqdm import tqdm  # For progress bar
import torch
import matplotlib.pyplot as plt
from utils import alter_box, calculate_iou


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='Trained_models/best_model.pth'):
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

    total_train_acc = []
    total_val_acc = []
    total_train_iou = []
    total_val_iou = []
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0  # Track total loss for the epoch
        class_epoch_loss = 0  # Track class loss for the epoch
        bbox_epoch_loss = 0   # Track bbox regression loss for the epoch
        n = 0
        # Loop over training batches
        correct_predictions = 0
        train_acc = []
        train_iou = []
        for X_batch, Y_batch, bbox_batch, gt_bbox_batch, tvals_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Move data to device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device).float()  # Ensure target is float for BCE loss
            gt_bbox = gt_bbox_batch.to(device)
            t_batch = tvals_batch.to(device)
            bbox = bbox_batch.to(device)

            # Forward pass
            class_score, t_vals = model(X_batch)  # y_pred, t_pred

            #Compute classifier accuracy:
            predicted_labels = (class_score > 0).float()
            correct_predictions = ((predicted_labels[:,0] == Y_batch).sum().item() ) / Y_batch.size(0)
            train_acc.append(correct_predictions)

            # Save altered boxes
            altered_boxes_list = []

            # Compute bbox accuracy
            for i in range(Y_batch.size(0)):  
                if Y_batch[i].cpu().item() == 1:  # Filter for positive samples
                    alt_box = alter_box(bbox[i].cpu().numpy(), t_vals[i].detach().cpu().numpy())
                    altered_boxes_list.append(alt_box)
                    iou = calculate_iou(alt_box, gt_bbox[i].cpu().numpy())
                    train_iou.append(iou)
                else:
                    # Append a placeholder (e.g., zeros) for non-positive samples to keep alignment
                    altered_boxes_list.append([0, 0, 0, 0])
            
            # Convert altered_boxes_list to a tensor
            altered_boxes_tensor = torch.tensor(altered_boxes_list).to(device)

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
        
        total_train_acc.append( sum(train_acc)/len(train_acc) )
        total_train_iou.append( sum(train_iou)/len(train_iou) )
        
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
        correct_predictions = 0
        val_acc = []
        val_iou = []
        with torch.no_grad():  # No gradients needed for validation
            for X_val, Y_val, bbox_val, gt_bbox_val, tvals_val in val_loader:
                # Move data to device
                X_val = X_val.to(device)
                Y_val = Y_val.to(device).float()
                gt_bbox_val = gt_bbox_val.to(device)
                t_batch_val = tvals_val.to(device)
                bbox = bbox_val.to(device)

                # Forward pass
                class_score_val, t_vals_val = model(X_val)

                #Compute classifier accuracy:
                predicted_labels = (class_score_val > 0).float()
                correct_predictions = ((predicted_labels[:,0] == Y_val).sum().item() ) / Y_val.size(0)
                val_acc.append(correct_predictions)


                # Save altered boxes
                altered_boxes_list = []

                # Compute bbox accuracy
                for i in range(Y_val.size(0)):  
                    if Y_val[i].cpu().item() == 1:  # Filter for positive samples
                        alt_box = alter_box(bbox[i].cpu().numpy(), t_vals_val[i].detach().cpu().numpy())
                        altered_boxes_list.append(alt_box)
                        iou = calculate_iou(alt_box, gt_bbox_val[i].cpu().numpy())
                        val_iou.append(iou)
                    else:
                        # Append a placeholder (e.g., zeros) for non-positive samples to keep alignment
                        altered_boxes_list.append([0, 0, 0, 0])
                
                # Convert altered_boxes_list to a tensor
                altered_boxes_tensor = torch.tensor(altered_boxes_list).to(device)

                # Compute validation loss
                val_loss, val_class, val_bbox = combined_loss(class_score_val.squeeze(), Y_val, t_vals_val, t_batch_val)
                val_total_loss += val_loss.item()
                val_class_loss += val_class.item()
                val_bbox_loss += val_bbox.item()
                n += 1
        
        total_val_acc.append( sum(val_acc)/len(val_acc) )
        total_val_iou.append( sum(val_iou)/len(val_iou) )
        
        # Noramlize loss
        val_total_loss /= n
        val_class_loss /= n
        val_bbox_loss /= n
        # Store validation losses for this epoch
        all_losses_val.append([val_total_loss, val_class_loss, val_bbox_loss])

        # Check for early stopping
        #early_stopping(val_total_loss, model)
    
        #if early_stopping.early_stop:
            #print("Early stopping triggered")
            #break

        # Load the best model after training
        #model.load_state_dict(torch.load('Trained_models/best_model.pth'))

        # Adjust learning rate
        scheduler.step(total_epoch_loss)

    print("train acc",total_train_acc)
    print("val acc",total_val_acc)
    print("train iou",total_train_iou)
    print("val iou",total_val_iou)

    plot_and_save(total_train_acc, total_val_acc, xlabel="Epoch", ylabel="Accuracy", title="classifier Accuracy")
    plot_and_save(total_train_iou, total_val_iou, xlabel="Epoch", ylabel="IOU", title="Bbox Accuracy", path="graphics/iou_plot_b0.png")
    
    return all_losses_train, all_losses_val



def plot_and_save(list1, list2, xlabel="X-axis", ylabel="Y-axis", title="Plot", path="graphics/acc_plot_b0.png"):
    """
    Plots two lists on the same graph and saves the plot as acc_plot.png.
    
    Parameters:
        list1 (list): First dataset for plotting.
        list2 (list): Second dataset for plotting.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(list1, label="train", marker='o')
    plt.plot(list2, label="val", marker='s')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print("Plot saved as acc_plot.png")