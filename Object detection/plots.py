import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from utils import parse_xml
import random

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

def load_image(image_id):
    path = os.path.join('Potholes', 'annotated-images', f"img-{image_id}.jpg")
    return Image.open(path)

def load_GT_boxes(image_id):
    with open(os.path.join('Potholes', 'annotated-images', f"img-{image_id}.xml")) as f:
        return parse_xml(f)

def load_SS_boxes(image_id):
    with open(os.path.join('Potholes', 'annotated-images', f"img-{image_id}_ss.json")) as f:
        boxes = json.load(f)
    return [box['bbox'] for box in boxes]

def plot_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

def save_img(save_path):
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_boxes(boxes, color, linestyle, linewidth, save_path, label=None):
    for box in boxes:
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    

def plot_steps(image_id, steps):
    path = "graphics/"
    image = load_image(image_id)
    GT_boxes = load_GT_boxes(image_id)
    SS_boxes = load_SS_boxes(image_id)
    
    for step in steps:
        save_path = os.path.join(path, f"step_{step}.jpg")
        
        if step == 1:
            # Step 1: Plot the original image
            plot_image(image)
            save_img(save_path)
        
        elif step == 2:
            # Step 2: Plot GT boxes in yellow
            plot_image(image)
            plot_boxes(GT_boxes, color='y', linestyle='-', linewidth=3, save_path=save_path, label='GT')
            save_img(save_path)
        
        elif step == 3:
            # Step 3: Plot a random sample of 5 SS boxes in dashed blue
            sampled_SS_boxes = random.sample(SS_boxes, 5)
            plot_image(image)
            plot_boxes(sampled_SS_boxes, color='b', linestyle='--', linewidth=2, save_path=save_path, label='SS')
            save_img(save_path)

        elif step == 4:
            # Step 4: Plot a random sample of 5 SS boxes in dashed blue and GT boxes in yellow
            sampled_SS_boxes = random.sample(SS_boxes, 5)
            plot_image(image)
            plot_boxes(GT_boxes, color='y', linestyle='-', linewidth=3, save_path=save_path, label='GT')
            plot_boxes(sampled_SS_boxes, color='b', linestyle='--', linewidth=2, save_path=save_path, label='SS')
            save_img(save_path)

