def conditional_bbox_mse_loss(t_vals, t_batch, Y_batch):
    # Expand Y_batch to match the bounding box dimensions
    Y_batch = Y_batch.unsqueeze(1)  # Shape becomes [batch_size, 1]
    # Calculate the element-wise squared error
    elementwise_loss = Y_batch * ((t_vals - t_batch) ** 2)
    # Sum over the bounding box dimensions (4) and take the mean over the batch
    loss = elementwise_loss.sum(dim=1).mean()
    return loss