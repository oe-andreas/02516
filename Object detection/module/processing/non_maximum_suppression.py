from utils import calculate_iou

def non_maximum_suppression(boxes_w_probs, discard_threshold, consideration_threshold=None):
    # NMS for binary classification
    
    # boxes_w_probs should be a list of (bbox, p) tuples
    #    bbox = [xmin, ymin, xmax, ymax] should be AFTER applying alter_box
    #    p should be the probability of the box being in the positive class
    # discard_threshold is the minimum IOU for a box to be discarded (i.e. if box1 and box2 have IOU > discard_threshold, discard the one with lower probability score)
    # consideration_threshold is the minimum probability for a box to be considered. If none, all boxes are considered
    
    if consideration_threshold is not None:
        boxes_w_probs = [(box, prob) for box, prob in boxes_w_probs if prob > consideration_threshold]
    
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
            if calculate_iou(current_box, box) < discard_threshold
        ]
    
    return selected_boxes