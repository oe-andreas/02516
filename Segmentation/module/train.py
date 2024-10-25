import torch
from .losses.losses import dice_loss, iou_loss, accuracy, sensitivity, specificity
import torch.nn as nn


def train(model, device, opt, scheduler, loss_fn, epochs, train_loader, test_loader):
    
    eval_metrics = [dice_loss, iou_loss, accuracy, sensitivity, specificity]
    
    observed_eval_metrics = []
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        # Training
        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch, Z_batch in train_loader:
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
        for X_batch, Y_batch, Z_batch in test_loader:
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

        # Step the scheduler with the validation loss
        scheduler.step(avg_loss)

    return train_losses, test_losses, observed_eval_metrics


def train_weak_annotation(model, device, opt, epochs, train_loader, test_loader):

    
    eval_metrics = [dice_loss, iou_loss, accuracy, sensitivity, specificity]
    
    observed_eval_metrics = []
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        # Training
        avg_loss = 0
        model.train()  # train mode
        for X_batch, _,  Y_batch in train_loader:
            #X_batch: Our input image
            # _: the full ground truth for the image (not used during training)
            # Y_batch: our weak annotations

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            y_pred = torch.sigmoid(Y_pred)

            loss = point_level_loss(y_pred, Y_batch, weight=0.5)
            
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
        for X_batch, Y_real, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_real = Y_real.to(device)
            Y_batch = Y_batch.to(device)

            with torch.no_grad():
                Y_pred = model(X_batch)
                y_pred = torch.sigmoid(Y_pred)
                loss = point_level_loss(y_pred, Y_batch, weight=0.2)

            avg_loss += loss.detach().cpu() / len(test_loader)
            
    
            for eval_metric in eval_metrics:
                avg_eval_metrics.append(eval_metric(Y_real, Y_pred).cpu() / len(test_loader))
        
        observed_eval_metrics.append(avg_eval_metrics)
        print(' - val_loss: %f' % avg_loss)
        test_losses.append(avg_loss)
    
    return train_losses, test_losses, observed_eval_metrics


#computes the point level loss for a batch
def point_level_loss(pred, target, weight=1.0):
    """
    Computes point-level supervised loss.
    
    Args:
        pred (torch.Tensor): The predicted segmentation map of shape [B, C, H, W].
        target (torch.Tensor): The ground truth segmentation of shape [B, H, W], with values 0, 1, ... (binary or multi-class).
        labeled_points (list of tuples): A list of labeled points as (x, y) coordinates. These are the points where supervision is provided.
        weight (float): Weight for the smoothness regularization.
    
    Returns:
        torch.Tensor: The computed loss (supervised + regularization).
    """
    # Initialize the loss
    loss = 0.0
    criterion = nn.BCELoss()

    # Get the batch size
    batch_size = pred.size(0)
    
    # Iterate over the batch and compute loss for each image
    for b in range(batch_size):
        pred_b = pred[b]  # Prediction for image b (shape: [C, H, W])
        
        # Supervised loss on labeled points
        supervised_loss = 0.0
        k = 0
        for (x, y, t) in target[b]:  # List of points for the current image
            
            pred_point = pred_b[0,x, y].unsqueeze(0)  # Predicted class logits at (x, y) (shape: [1, C])
            target_point = t.view(-1).float()  # Ground truth class at (x, y) (shape: [1])

            supervised_loss += criterion(pred_point, target_point)
            k = k+1
            
        
        # Average supervised loss over the labeled points
        supervised_loss = supervised_loss / len(target[b])
        
        # Add smoothness regularization (TV regularization)
        smoothness_loss = total_variation_loss(pred_b)
        
        #Total loss for this batch element
        loss += supervised_loss + weight * smoothness_loss

    # Average the loss over the batch
    loss = loss / batch_size

    return loss


def total_variation_loss(pred):
    """
    Total Variation (TV) loss for smoothness regularization.
    This encourages smooth transitions in the segmentation map.
    
    Args:
        pred (torch.Tensor): The predicted segmentation map for a single image (shape: [C, H, W]).
    
    Returns:
        torch.Tensor: The total variation loss.
    """
    # Compute differences between neighboring pixels (vertical and horizontal)
    diff_h = torch.abs(pred[0,1:, :] - pred[0,:-1, :]).mean()
    diff_w = torch.abs(pred[0,:, 1:] - pred[0,:, :-1]).mean()
    
    # TV loss is the sum of these differences
    tv_loss = diff_h + diff_w
    return tv_loss








