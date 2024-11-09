import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import parse_xml, calculate_iou
import json

def return_edge_box_proposals(image, max_boxes=100, min_score=0.01, Canny_lower_threshold=50, Canny_upper_threshold=150):
    # Finds edge box proposals for a single image
    #input:
    # image: image object (np array)
    # max_boxes: maximum number of boxes to return
    # min_score: minimum score for a box to be considered
    # Canny_lower_threshold: lower threshold for Canny edge detection
    # Canny_upper_threshold: upper threshold for Canny edge detection
    #output: list of bounding boxes in [x, y, width, height] format. OBS! NOT [xmin, ymin, xmax, ymax]
    
    
    
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Image must be a valid NumPy array.")

    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(max_boxes)
    edge_boxes.setMinScore(min_score)

    edges = cv2.Canny(image, Canny_lower_threshold, Canny_upper_threshold)
    edges = edges.astype(np.float32)  # Convert edge map to CV_32F
    orientation_map = np.zeros_like(edges, dtype=np.float32)  # Placeholder orientation map as CV_32F

    bbs = edge_boxes.getBoundingBoxes(edges, orientation_map)[0]

    return bbs