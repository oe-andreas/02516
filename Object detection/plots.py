import matplotlib.pyplot as plt
import os

def plot_losses(all_losses_train, all_losses_val, save_path='graphics'):
    """
    Plot training and validation losses over epochs and save to file.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Unpack losses
    train_total_loss, train_class_loss, train_bbox_loss = zip(*all_losses_train)
    val_total_loss, val_class_loss, val_bbox_loss = zip(*all_losses_val)

    # Plot total loss
    epochs = range(1, len(train_total_loss) + 1)
    plt.figure(figsize=(12, 8))

    # Plot total loss
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_total_loss, 'b-', label='Training Total Loss')
    plt.plot(epochs, val_total_loss, 'r-', label='Validation Total Loss')
    plt.title('Total Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot classification loss
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_class_loss, 'b-', label='Training Classification Loss')
    plt.plot(epochs, val_class_loss, 'r-', label='Validation Classification Loss')
    plt.title('Classification Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot bounding box regression loss
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_bbox_loss, 'b-', label='Training BBox Loss')
    plt.plot(epochs, val_bbox_loss, 'r-', label='Validation BBox Loss')
    plt.title('Bounding Box Regression Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    
    # Save plot
    file_path = os.path.join(save_path, 'loss_plot.png')
    plt.savefig(file_path)
    plt.close()  # Close the figure to free memory

    print(f"Plot saved to {file_path}")
