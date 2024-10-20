import torch


def bce_loss(y_real, y_pred_logits):
    #takes logits
    return torch.mean(torch.maximum(torch.tensor(0.0), y_pred_logits) - y_real*y_pred_logits + torch.log(1 + torch.exp(-torch.abs(y_pred_logits))))

def dice_loss(y_real, y_pred_logits):
    y_pred = torch.sigmoid(y_pred_logits)
    return 1 - torch.mean(2*y_real*y_pred + 1) / (torch.mean(y_real + y_pred) + 1)

def iou_loss(y_real, y_pred_logits):
    #intersection over union loss
    y_pred = torch.sigmoid(y_pred_logits)  # Apply sigmoid to logits
    intersection = torch.sum(y_real * y_pred)
    union = torch.sum(y_real) + torch.sum(y_pred) - intersection
    iou = (intersection + 1) / (union + 1)
    return 1 - iou

def focal_loss(y_real, y_pred_logits, gamma=2):
    # expects logits
    y_pred = torch.sigmoid(y_pred_logits)
    
    # calculate focal loss
    loss = -y_real * (1 - y_pred)**gamma * torch.log(y_pred) - (1 - y_real) * y_pred**gamma * torch.log(1 - y_pred)
    
    return torch.mean(loss)

def bce_total_variation(y_real, y_pred_logits):
    
    y_pred = torch.sigmoid(y_pred_logits)
    
    total_variation = torch.mean(torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])) + \
                     torch.mean(torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]))
    
    return bce_loss(y_real, y_pred_logits) + 0.1*total_variation


def accuracy(y_real, y_pred_logits):
    y_pred = torch.sigmoid(y_pred_logits) > 0.5
    correct = torch.sum(y_pred == y_real)
    return correct / y_real.numel()

def sensitivity(y_real, y_pred_logits):
    y_pred = torch.sigmoid(y_pred_logits) > 0.5
    true_positive = torch.sum((y_pred == 1) & (y_real == 1))
    actual_positive = torch.sum(y_real == 1)
    return true_positive / actual_positive

def specificity(y_real, y_pred_logits):
    y_pred = torch.sigmoid(y_pred_logits) > 0.5
    true_negative = torch.sum((y_pred == 0) & (y_real == 0))
    actual_negative = torch.sum(y_real == 0)
    return true_negative / actual_negative
