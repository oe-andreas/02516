from utils import calculate_iou
import numpy as np

def AP(gt_boxes, proposed_boxes, proposed_box_probs, threshold=0.5):
    #gt_boxes: Actual positive class boxes in [xmin, ymin, xmax, ymax] format
    #proposed_boxes: All proposed boxes, in [xmin, ymin, xmax, ymax] format
    #proposed_box_probs: Probability of each proposed box being positive class
    #threshold: IOU threshold for a correct detection
    
    # Sort proposed boxes by probability
    order = np.argsort(proposed_box_probs)[::-1]
    proposed_boxes = np.array(proposed_boxes)[order]
    proposed_box_probs = np.array(proposed_box_probs)[order]
    
    # Use IoU to determine if a box is a true positive
    ious = np.array([[calculate_iou(gt_box, proposed_box) for gt_box in gt_boxes] for proposed_box in proposed_boxes])
    max_ious = ious.max(axis=1)
    matches_gt_box = max_ious > threshold
    
    # Calculate precision and recall
    TP = np.cumsum(matches_gt_box)
    FP = np.cumsum(~matches_gt_box)
    
    precision = TP / (TP + FP)
    recall = TP / len(gt_boxes)
    
    # Calculate AP
    AP = np.sum(precision * np.diff([0, *recall]))
    
    return AP