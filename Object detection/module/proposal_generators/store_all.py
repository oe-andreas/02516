import cv2
import os
import json
from tqdm import tqdm
from utils import parse_xml, calculate_iou



def store_all_proposals(proposal_function, suffix, **proposal_kwargs):
    #computes proposals for all images and saves them in json files. 
    #
    # input:
    # The proposal function must take an image as input and return a list of bounding boxes in [x, y, width, height] format
    # suffix: string, suffix to add to the json file name
    # proposal_kwargs: keyword arguments to pass to the proposal function
    #
    #storage format: For image i, stores 'img-i_{suffix}.json', which contains a list of dicts. Each dict is
    #{
    #           'bbox': [x_min, y_min, x_max, y_max], #list of ints
    #           'class': pothole_class,  #0, 1 or None. 0 if max_iou < 0.3, 1 if max_iou > 0.7, None otherwise
    #           'iou': max_iou #the max iou value between the bbox and any GT pothole bbox
    #       }
    
    
    for i in tqdm(range(1, 666), desc="Processing Images", unit="image"):
        # Load the image
        image = cv2.imread(f'Potholes/annotated-images/img-{i}.jpg', 1) #,1 means read as color image

        # Run selective search
        proposals = proposal_function(image, **proposal_kwargs)

        # Parse the XML to get the pothole bounding boxes
        xml_file = f'Potholes/annotated-images/img-{i}.xml'
        pothole_bboxes = parse_xml(xml_file)

        # Prepare the results list with IoU calculations
        results_list = []
        for bbox in proposals:
            # Convert bbox from (x, y, width, height) to (xmin, ymin, xmax, ymax)
            x, y, w, h = bbox
            bbox_list = [int(x), int(y), int(x + w), int(y + h)]

            # Calculate maximum IoU
            iou = 0
            for pothole_bbox in pothole_bboxes:
                if calculate_iou(pothole_bbox, bbox_list) > iou:
                    iou = calculate_iou(pothole_bbox, bbox_list)
            if iou >= 0.7:
                pothole_class = 1
            elif iou <= 0.3:
                pothole_class = 0
            else:
                pothole_class = None
            
            # Create a dictionary for each bounding box with IoU
            bbox_dict = {
                'bbox': bbox_list,
                'class': pothole_class,
                'iou': iou
            }
            results_list.append(bbox_dict)

        # Save results to JSON
        with open(f'Potholes/annotated-images/img-{i}_{suffix}.json', "w") as file:
            json.dump(results_list, file)
