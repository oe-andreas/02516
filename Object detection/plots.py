import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import random
import torch
import numpy as np

from utils import parse_xml
from utils import get_input_size
from utils import alter_box
from module.models.efficientnet import EfficientNetWithBBox
from module.processing.non_maximum_suppression import non_maximum_suppression

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
    """Loads an image by its ID."""
    path = os.path.join('Potholes', 'annotated-images', f"img-{image_id}.jpg")
    return Image.open(path)

def load_GT_boxes(image_id):
    """Loads ground truth boxes from XML."""
    with open(os.path.join('Potholes', 'annotated-images', f"img-{image_id}.xml")) as f:
        return parse_xml(f)

def load_SS_boxes(image_id, num_GT_boxes, num_class_0, all_boxes=False):
    """Loads selective search boxes and filters by class."""
    with open(os.path.join('Potholes', 'annotated-images', f"img-{image_id}_ss.json")) as f:
        boxes = json.load(f)
    
    if all_boxes:
        # Return all boxes regardless of class
        all_ss_boxes = [box['bbox'] for box in boxes]
        return all_ss_boxes
    
    # Separate boxes by class
    class_1_boxes = [box['bbox'] for box in boxes if box['class'] == 1]
    class_0_boxes = [box['bbox'] for box in boxes if box['class'] == 0]
    
    # Shuffle class 1 boxes before selecting
    random.shuffle(class_1_boxes)

    # Load as many class 1 boxes as there are GT boxes, which are randomly sampled, or all if not enough
    selected_class_1_boxes = class_1_boxes[:num_GT_boxes]
    if len(selected_class_1_boxes) < num_GT_boxes:
        print("Warning: Not enough class 1 boxes. Using available boxes only.")
    
    # Randomly sample the specified number of class 0 boxes
    selected_class_0_boxes = random.sample(class_0_boxes, min(num_class_0, len(class_0_boxes)))

    # Combine both sets of boxes
    selected_boxes = selected_class_1_boxes + selected_class_0_boxes
    return selected_boxes

def plot_image(image):
    """Displays a plot image without axis."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')

def save_img(save_path):
    """Saves the current plot."""
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_boxes(boxes, color, linestyle, linewidth, label=None):
    """Plots boxes on an image with optional label for legend."""
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        # Add label only to the first box of each type to avoid duplicates in the legend
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, linestyle=linestyle, linewidth=linewidth, label=label if i == 0 else None)
    
def plot_probability_text(box, class_prob, color, offset=-5):
    """
    Plots the probability text above the top-left corner of a bounding box.

    Parameters:
        box (list or tuple): The bounding box coordinates (x1, y1, x2, y2).
        class_prob (torch.Tensor or float): The predicted class probability.
        color (str): The color of the text.
        offset (int): Vertical offset for the text position (default is -5).
    """
    # Get top-left corner of the box
    x1, y1, _, _ = box

    # Ensure class_prob is a float
    prob = class_prob.item() if isinstance(class_prob, torch.Tensor) else class_prob

    # Plot the probability text
    plt.text(
        x1, y1 + offset,  # Position above the top-left corner with offset
        f'{prob:.2f}',  # Display probability with 2 decimal points
        color=color,
        fontsize=10,
        backgroundcolor='white',
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
    )

def plot_altered(image, boxes, model, dim, device, linestyle='-', linewidth=3):
    """Plots altered boxes based on model predictions."""
    used_labels = set()  # Set to track labels that have already been used
    for box in boxes:
        crop = image.crop(box)
        resized_crop = crop.resize(dim, Image.LANCZOS)
        tensor_crop = torch.tensor(np.array(resized_crop), dtype=torch.float32).permute(2, 0, 1) / 255.0 
        tensor_crop = tensor_crop.unsqueeze(0).to(device)  # Ensure tensor_crop is on the same device
        
        class_prob_logit, t_values = model(tensor_crop)
        
        class_prob = torch.sigmoid(class_prob_logit)
        t_values = t_values.squeeze().tolist()

        # Alter the box using the t_values
        altered_box = [alter_box(box, t_values)] # Wrap in list for plot_boxes to interpret as a single box

        color = 'g' if class_prob > 0.5 else 'r'
        label = f'Altered, {"positive" if class_prob > 0.5 else "negative"}'
        if label not in used_labels:
            plot_boxes(altered_box, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            used_labels.add(label)  # Mark this label as used
        else:
            plot_boxes(altered_box, color=color, linestyle=linestyle, linewidth=linewidth, label=None)

        plot_probability_text(altered_box[0], class_prob, color)

def plot_altered_NMS(image, boxes, model, dim, device, discard_threshold, consideration_threshold, linestyle='-', linewidth=3):
    """Applies NMS to altered boxes based on model predictions and plots results."""
    boxes_w_probs = []
    used_labels = set()  # Set to track labels that have already been used
    for box in boxes:
        crop = image.crop(box)
        resized_crop = crop.resize(dim, Image.LANCZOS)
        tensor_crop = torch.tensor(np.array(resized_crop), dtype=torch.float32).permute(2, 0, 1) / 255.0 
        tensor_crop = tensor_crop.unsqueeze(0).to(device)  # Ensure tensor_crop is on the same device
        class_prob_logit, t_values = model(tensor_crop)
        
        class_prob = torch.sigmoid(class_prob_logit)
        t_values = t_values.squeeze().tolist()

        # Alter the box using the t_values
        altered_box = alter_box(box, t_values)

        boxes_w_probs.append((altered_box, class_prob))

    # Apply NMS with chosen thresholds
    nms_boxes = non_maximum_suppression(boxes_w_probs, discard_threshold, consideration_threshold)
    # Plot the resulting boxes after NMS
    for (box, class_prob) in nms_boxes:
        color = 'g' if class_prob > 0.5 else 'r'
        label = f'Altered, {"positive" if class_prob > 0.5 else "negative"}'
        # Add the label only if it hasn't been used before
        if label not in used_labels:
            plot_boxes([box], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            used_labels.add(label)  # Mark this label as used
        else:
            plot_boxes([box], color=color, linestyle=linestyle, linewidth=linewidth, label=None)  # No label
        plot_probability_text(box, class_prob, color)
    
def plot_altered_NMS_positive(image, boxes, model, dim, device, discard_threshold, consideration_threshold, linestyle='-', linewidth=3):
    """Applies NMS to all altered boxes that the model predicts to be potholes and plots results."""    
    boxes_w_probs = []
    used_labels = set()  # Set to track labels that have already been used
    for box in boxes:
        crop = image.crop((box[0],box[1],box[2],box[3]))
        resized_crop = crop.resize(dim, Image.LANCZOS)
        tensor_crop = torch.tensor(np.array(resized_crop), dtype=torch.float32).permute(2, 0, 1) / 255.0 
        tensor_crop = tensor_crop.unsqueeze(0).to(device)  # Ensure tensor_crop is on the same device
        class_prob_logit, t_values = model(tensor_crop)
        
        class_prob = torch.sigmoid(class_prob_logit)
        t_values = t_values.squeeze().tolist()

        # Alter the box using the t_values
        altered_box = alter_box(box, t_values)
        if class_prob > 0.5:
            boxes_w_probs.append((altered_box, class_prob))

    # Apply NMS with chosen thresholds
    nms_boxes = non_maximum_suppression(boxes_w_probs, discard_threshold, consideration_threshold)
    # Plot the resulting boxes after NMS
    for (box, class_prob) in nms_boxes:
        color = 'g' if class_prob > 0.5 else 'r'
        label = f'Altered, {"positive" if class_prob > 0.5 else "negative"}'
        if label not in used_labels:
            plot_boxes([box], color=color, linestyle=linestyle, linewidth=linewidth, label=label)
            used_labels.add(label)  # Mark this label as used
        else:
            plot_boxes([box], color=color, linestyle=linestyle, linewidth=linewidth, label=None)  # No label
        plot_probability_text(box, class_prob, color)

def plot_steps(image_id, steps, model_path, model_name, device, num_class_0 = 5, discard_threshold = 0.5, consideration_threshold = None):
    path = "graphics/"
    image = load_image(image_id)
    GT_boxes = load_GT_boxes(image_id)
    
    # Load SS boxes with the specified number of GT and class 0 boxes
    SS_boxes = load_SS_boxes(image_id, num_GT_boxes=len(GT_boxes), num_class_0=num_class_0)

    SS_boxes_all = load_SS_boxes(image_id, num_GT_boxes=len(GT_boxes), num_class_0=num_class_0, all_boxes=True)

    # Load the model
    model = EfficientNetWithBBox(model_name, pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    input_size = get_input_size(model_name)
    dim = (input_size, input_size)
    
    for step in steps:
        save_path = os.path.join(path, f"step_{step}.jpg")
        
        if step == 1:
            # Step 1: Plot the original image
            plot_image(image)
            save_img(save_path)
        
        elif step == 2:
            # Step 2: Plot GT boxes in yellow
            plot_image(image)
            plot_boxes(GT_boxes, color='y', linestyle='-', linewidth=3, label='GT')
            plt.legend()
            save_img(save_path)
        
        elif step == 3:
            # Step 3: Plot selected SS boxes in dashed blue
            plot_image(image)
            plot_boxes(SS_boxes, color='b', linestyle='--', linewidth=2, label='SS')
            plt.legend()
            save_img(save_path)

        elif step == 4:
            # Step 4: Plot selected SS boxes and GT boxes
            plot_image(image)
            plot_boxes(GT_boxes, color='y', linestyle='-', linewidth=3, label='GT')
            plot_boxes(SS_boxes, color='b', linestyle='--', linewidth=2, label='SS')
            plt.legend()
            save_img(save_path)
        
        elif step == 5:
            # Step 5: Plot SS boxes, altered predictions from the model, and GT boxes
            plot_image(image)
            plot_boxes(GT_boxes, color='y', linestyle='-', linewidth=3, label='GT')
            plot_boxes(SS_boxes, color='b', linestyle='--', linewidth=2, label='SS')
            plot_altered(image, SS_boxes, model, dim, device)
            plt.legend()
            save_img(save_path)

        elif step == 6:
            # Step 6: Apply NMS to the predicted boxes after alteration
            plot_image(image)
            plot_boxes(GT_boxes, color='y', linestyle='-', linewidth=3, label='GT')
            plot_boxes(SS_boxes, color='b', linestyle='--', linewidth=2, label='SS')
            plot_altered_NMS(image, SS_boxes, model, dim, device, discard_threshold=discard_threshold, consideration_threshold=consideration_threshold)
            plt.legend()
            save_img(save_path)
        
        elif step == 7:
            plot_image(image)
            plot_boxes(GT_boxes, color='y', linestyle='-', linewidth=3, label='GT')
            plot_altered_NMS_positive(image, SS_boxes_all, model, dim, device, discard_threshold=discard_threshold, consideration_threshold=consideration_threshold)
            plt.legend()
            save_img(save_path)