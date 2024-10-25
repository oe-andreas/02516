import torch
import torch.nn as nn



# Samples pixels from ground truth image
def sample_pixels_dist(tensor, num_samples=5, min_distance=10, edge_distance=5):
    """
    Samples pixels from the tensor, ensuring 0-value pixels are a certain distance from edges 
    and both 0-value and 1-value pixels are a minimum distance apart from each other.
    
    Args:
        tensor (torch.Tensor): The input tensor of size [H, W] with values 0 or 1.
        num_samples (int): Number of samples to pick for each class.
        min_distance (int): Minimum distance between sampled pixels.
        edge_distance (int): Minimum distance away from the edges for 0-value pixels.
        
    Returns:
        zero_sampled (torch.Tensor): Sampled 0-value pixel coordinates.
        one_sampled (torch.Tensor): Sampled 1-value pixel coordinates.
    """
    H, W = tensor.shape

    # Find indices of all 0s and 1s
    zero_indices = (tensor == 0).nonzero(as_tuple=False)
    one_indices = (tensor == 1).nonzero(as_tuple=False)

    # Filter 0-value pixels that are at least `edge_distance` away from the edges
    zero_indices = zero_indices[
        (zero_indices[:, 0] >= edge_distance) &  # y >= edge_distance
        (zero_indices[:, 0] < H - edge_distance) &  # y < H - edge_distance
        (zero_indices[:, 1] >= edge_distance) &  # x >= edge_distance
        (zero_indices[:, 1] < W - edge_distance)   # x < W - edge_distance
    ]

    def is_far_enough(selected, candidate, min_dist):
        """Check if candidate pixel is far enough from all selected pixels."""
        if selected.size(0) == 0:
            return True  # No previous points selected, so it's valid
        distances = torch.norm(selected.float() - candidate.float(), dim=1)
        return torch.all(distances >= min_dist)

    def sample_with_distance(indices, num_samples, min_dist):
        """Sample num_samples indices ensuring they are min_dist apart."""
        seed = 42
        torch.manual_seed(seed)
        selected = []
        while len(selected) < num_samples:
            candidate = indices[torch.randint(0, indices.size(0), (1,))].squeeze(0)
            if is_far_enough(torch.stack(selected) if selected else torch.empty((0, 2)), candidate, min_dist):
                selected.append(candidate)
        return torch.stack(selected)
    
    # Randomly sample 0-value and 1-value pixels with distance constraints
    zero_sampled = sample_with_distance(zero_indices, num_samples, min_distance)
    one_sampled = sample_with_distance(one_indices, num_samples, min_distance)
    
    return zero_sampled, one_sampled


# Creates weak annotations for every image in dataset
def create_weak_annotations(dataloader,num_samples=5, min_distance=10, edge_distance=5):
    target_weak = []


    for X_batch, Y_batch in dataloader:
        n = len(X_batch)
        for i in range(n):
            X, y = X_batch[i], Y_batch[i]
            zero_sampled, one_sampled = sample_pixels_dist(y.cpu().squeeze(),num_samples, min_distance, edge_distance)
            labeled_points = torch.cat((zero_sampled, one_sampled), dim=0)

            target = labeled_points.tolist()
            target_weak.append(target)

    return target_weak


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




