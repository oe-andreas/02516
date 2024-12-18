import torch
import torch.nn.functional as F

def bce_loss(y_real, y_pred_logits):
    #takes logits
    return torch.mean(torch.maximum(torch.tensor(0.0), y_pred_logits) - y_real*y_pred_logits + torch.log(1 + torch.exp(-torch.abs(y_pred_logits))))

def dice_loss(y_real, y_pred_logits):
    """
    Compute Dice score between ground truth and predictions.

    Args:
        y_real (torch.Tensor): Ground truth binary mask, shape (N, H, W) or (N, 1, H, W)
        y_pred_logits (torch.Tensor): Predicted logits, shape (N, H, W) or (N, 1, H, W)
        threshold (float): Threshold for converting logits to binary mask
        smooth (float): Small constant for numerical stability

    Returns:
        torch.Tensor: Dice score
    """
    # Convert logits to predicted binary mask
    
    threshold = 0.5
    smooth = 1e-6
    
    y_pred = (torch.sigmoid(y_pred_logits) > threshold).float()
    
    # Flatten tensors
    y_real_flat = y_real.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    # Calculate intersection and union
    intersection = (y_real_flat * y_pred_flat).sum()
    union = y_real_flat.sum() + y_pred_flat.sum()
    
    # Compute Dice score
    dice = (2. * intersection + smooth) / (union + smooth)
    
    if dice > 1:
        print(f"dice > 1!: {y_pred.min(), y_pred.max(), intersection, union}")
    
    return dice

def iou_loss(y_real, y_pred_logits):
    #intersection over union loss
    
    threshold = 0.5
    smooth = 1
    
    # Convert logits to predicted binary mask
    y_pred = (torch.sigmoid(y_pred_logits) > threshold).float()
    
    # Flatten tensors
    y_real_flat = y_real.view(-1)
    y_pred_flat = y_pred.view(-1)
    
    # Calculate intersection and union
    intersection = (y_real_flat * y_pred_flat).sum()
    union = y_real_flat.sum() + y_pred_flat.sum() - intersection
    
    # Compute IoU score
    iou = (intersection + smooth) / (union + smooth)
    return iou

def focal_loss(y_real, y_pred_logits, gamma=2.0, alpha=0.8, epsilon=1e-6):
    # Apply sigmoid to logits to get probabilities
    y_pred = torch.sigmoid(y_pred_logits)

    # Clamp predictions more conservatively to prevent log(0) calculations and NaNs
    y_pred = torch.clamp(y_pred, min=epsilon, max=1 - epsilon)

    # Calculate focal loss components for positive and negative samples
    pos_loss = -alpha * (1 - y_pred) ** gamma * y_real * torch.log(y_pred)
    neg_loss = -(1 - alpha) * y_pred ** gamma * (1 - y_real) * torch.log(1 - y_pred)
    
    # Sum positive and negative losses
    loss = pos_loss + neg_loss
    
    return torch.mean(loss)


def bce_total_variation(y_real, y_pred_logits):
    
    y_pred = torch.sigmoid(y_pred_logits)
    
    total_variation = torch.mean(torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])) + \
                     torch.mean(torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]))
    
    return bce_loss(y_real, y_pred_logits) + 0.1*total_variation


def bce_weighted(y_real, y_pred_logits):
    """
    Computes weighted binary cross-entropy loss with automatically calculated weights
    based on class frequencies (for handling class imbalance).

    Parameters:
    y_real: Tensor of true binary labels (0 or 1).
    y_pred_logits: Tensor of predicted logits (not probabilities).

    Returns:
    Weighted binary cross-entropy loss.
    """
    # Calculate the number of positive and negative samples
    num_positives = torch.sum(y_real)
    num_negatives = torch.sum(1.0 - y_real)
    
    # Compute the positive class weight: neg / pos
    pos_weight = num_negatives / (num_positives + 1e-8)  # Avoid division by zero
    pos_weight = pos_weight.clone().detach().to(y_real.device)
    # Compute binary cross-entropy with logits, applying the pos_weight to positive samples
    loss = F.binary_cross_entropy_with_logits(y_pred_logits, y_real, pos_weight=pos_weight)
    
    return loss  # Return loss



def accuracy(y_real, y_pred_logits):
    y_pred = torch.sigmoid(y_pred_logits) > 0.5
    correct = torch.sum(y_pred == y_real)
    return 1 - correct / y_real.numel()

def sensitivity(y_real, y_pred_logits):
    y_pred = torch.sigmoid(y_pred_logits) > 0.5
    true_positive = torch.sum((y_pred == 1) & (y_real == 1))
    actual_positive = torch.sum(y_real == 1)
    return 1 - true_positive / actual_positive

def specificity(y_real, y_pred_logits):
    y_pred = torch.sigmoid(y_pred_logits) > 0.5
    true_negative = torch.sum((y_pred == 0) & (y_real == 0))
    actual_negative = torch.sum(y_real == 0)
    return 1 - true_negative / actual_negative
