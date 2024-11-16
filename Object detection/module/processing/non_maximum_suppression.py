from utils import calculate_iou

def non_maximum_suppression(boxes_w_probs, discard_threshold, consideration_threshold=None, return_indeces = False):
    # NMS for binary classification
    
    # boxes_w_probs should be a list of (bbox, p) tuples
    #    bbox = [xmin, ymin, xmax, ymax] should be AFTER applying alter_box
    #    p should be the probability of the box being in the positive class
    # discard_threshold is the minimum IOU for a box to be discarded (i.e. if box1 and box2 have IOU > discard_threshold, discard the one with lower probability score)
    # consideration_threshold is the minimum probability for a box to be considered. If none, all boxes are considered
    
    
    #add idx
    boxes_w_probs = [(box, prob, idx) for idx, (box, prob) in enumerate(boxes_w_probs)]
    
    
    if consideration_threshold is not None:
        boxes_w_probs = [(box, prob, idx) for box, prob, idx in boxes_w_probs if prob > consideration_threshold]
    
    # Sort the boxes by their probabilities in descending order
    boxes_w_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize an empty list to store the selected boxes
    selected_boxes = []
    selected_indeces = []
    
    while boxes_w_probs:
        # Select the box with the highest probability
        current_box, current_prob, current_idx = boxes_w_probs.pop(0)
        selected_boxes.append((current_box, current_prob))
        selected_indeces.append(current_idx)
        
        # Remove boxes that have high IOU with the current box
        boxes_w_probs = [
            (box, prob, idx) for box, prob, idx in boxes_w_probs
            if calculate_iou(current_box, box) < discard_threshold
        ]
    
    if return_indeces:
        return selected_boxes, selected_indeces
    else:
        return selected_boxes