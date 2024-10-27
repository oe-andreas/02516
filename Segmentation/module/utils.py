import torch

def calculate_mean_std(dataloader):
    """Calculate mean and std of images in a dataloader without masks."""
    mean = 0.0
    std = 0.0
    count = 0

    for images, _, _ in dataloader:
        batch_samples = images.size(0)  # number of images in the batch
        images = images.view(batch_samples, images.size(1), -1)  # Reshape to (batch, channels, H*W)
        
        # Incremental mean and std calculations
        mean += images.mean(2).sum(0)  # Mean over each channel
        std += images.std(2).sum(0)    # Std over each channel
        count += batch_samples

    mean /= count
    std /= count

    return mean, std

def calculate_mean_std_with_mask(dataloader):
    """Calculate mean and std of images in a dataloader using the provided masks."""
    mean = 0.0
    std = 0.0
    count = 0

    for images, _, masks in dataloader:
        batch_samples = images.size(0)  # number of images in the batch
        images = images.view(batch_samples, images.size(1), -1)  # Reshape to (batch, channels, H*W)
        
        # Reshape masks to the same dimensions as images (if necessary)
        masks = masks.view(batch_samples, 1, -1)  # Reshape to (batch, 1, H*W)

        # Masking: Only consider pixels where the mask is True (or > 0)
        masked_images = images * masks
        
        # Calculate mean and std
        mean += masked_images.sum(2).sum(0) / masks.sum(2).sum(0)  # Sum only over masked pixels
        std += ((masked_images - mean.unsqueeze(0).unsqueeze(2)) ** 2).sum(2).sum(0) / masks.sum(2).sum(0)
        count += batch_samples  # Count number of batches

    mean /= count
    std = torch.sqrt(std / count)  # Calculate final std

    return mean, std
