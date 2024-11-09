from utils import calculate_iou

def non_maximum_suppression(boxes_w_probs, threshold_for_discard, threshold_for_consideration=None):
    # NMS for binary classification
    
    # boxes_w_probs should be a list of (bbox, p) tuples
    #    bbox = [xmin, ymin, xmax, ymax] should be AFTER applying alter_box
    #    p should be the probability of the box being in the positive class
    # threshold_for_discard is the minimum IOU for a box to be discarded (i.e. if box1 and box2 have IOU > threshold_for_discard, discard the one with lower probability score)
    # threshold_for_consideration is the minimum probability for a box to be considered. If none, all boxes are considered
    
    if threshold_for_consideration is not None:
        boxes_w_probs = [(box, prob) for box, prob in boxes_w_probs if prob > threshold_for_consideration]
    
    # Sort the boxes by their probabilities in descending order
    boxes_w_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize an empty list to store the selected boxes
    selected_boxes = []
    
    while boxes_w_probs:
        # Select the box with the highest probability
        current_box, current_prob = boxes_w_probs.pop(0)
        selected_boxes.append((current_box, current_prob))
        
        # Remove boxes that have high IOU with the current box
        boxes_w_probs = [
            (box, prob) for box, prob in boxes_w_probs
            if calculate_iou(current_box, box) < threshold_for_discard
        ]
    
    return selected_boxes