import torch.nn as nn

def conditional_bbox_mse_loss(t_vals, t_batch, Y_batch):
    # Ensure Y_batch is a float tensor for multiplication
    Y_batch = Y_batch.float()
    
    # Expand Y_batch to match the bounding box dimensions
    Y_batch = Y_batch.unsqueeze(1)  # Shape becomes [batch_size, 1]
    
    # Filter t_vals and t_batch based on Y_batch
    mask = Y_batch.squeeze(1) > 0
    t_vals = t_vals[mask]
    t_batch = t_batch[mask]
    
    # Calculate the element-wise squared error
    elementwise_loss = Y_batch[mask].unsqueeze(1) * ((t_vals - t_batch) ** 2)
    
    # Sum over the bounding box dimensions (4) and take the mean over the batch
    loss = elementwise_loss.sum(dim=1).mean()
    return loss


# Define the multi-task loss function
class MultiTaskLoss(nn.Module):
    def __init__(self, classification_weight=1, bbox_weight=1.0, ignore_negative_for_bbox = True):
        super(MultiTaskLoss, self).__init__()
        self.classification_loss = nn.BCEWithLogitsLoss()  # Binary classification loss
        self.bbox_loss = nn.SmoothL1Loss()  # Bounding box regression loss
        self.classification_weight = classification_weight
        self.bbox_weight = bbox_weight
        
        self.ignore_negative_for_bbox = ignore_negative_for_bbox

    def forward(self, class_pred, class_true, bbox_pred, bbox_true):
        # Classification loss 
        class_loss = self.classification_loss(class_pred, class_true)
        #print("class: ",class_loss)
        # Bounding box loss
        
        if self.ignore_negative_for_bbox:
            # Create a mask to ignore negative samples for bbox loss
            bbox_mask = class_true > 0
            bbox_pred = bbox_pred[bbox_mask]
            bbox_true = bbox_true[bbox_mask]
        
        
        bbox_loss = self.bbox_loss(bbox_pred, bbox_true)
        #print("bbox: ",bbox_loss)

        #Normalize loss
        loss_1_normalized = class_loss / (class_loss + bbox_loss).mean()
        loss_2_normalized = bbox_loss / (class_loss + bbox_loss).mean()

        # Combine losses
        total_loss = loss_2_normalized * class_loss + loss_1_normalized * bbox_loss
        #print("weight class: ",loss_2_normalized * class_loss) 
        #print("weight box: ",loss_1_normalized * bbox_loss) 
        #print("total: ",total_loss)
        #print(" ")
        return total_loss, class_loss, bbox_loss