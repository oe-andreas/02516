import cv2
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import re
import torch
import numpy as np


#Reads xml files
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # List to hold bounding boxes for all potholes
    pothole_bboxes = []

    # Get all bounding boxes from the XML
    for obj in root.findall('object'):
        if obj.find('name').text == 'pothole':
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            pothole_bboxes.append((xmin, ymin, xmax, ymax))

    return pothole_bboxes

# computes iou of two bounding boxes
def calculate_iou(boxA, boxB):
    # Calculate the (x, y) coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection area
    # and dividing it by the sum of prediction + ground truth areas - the interection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# Does selective search for a single image
def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ss_results = ss.process()
    return ss_results

# Does selective search for all images
def selective_search_all():
    for i in tqdm(range(1, 666), desc="Processing Images", unit="image"):
        # Load the image
        image = cv2.imread(f'Potholes/annotated-images/img-{i}.jpg', 1)

        # Run selective search
        ss_results = selective_search(image)

        # Parse the XML to get the pothole bounding box
        xml_file = f'Potholes/annotated-images/img-{i}.xml'
        pothole_bboxes = parse_xml(xml_file)

        # Prepare the results list with IoU calculations
        results_list = []
        for bbox in ss_results:
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
        with open(f'Potholes/annotated-images/img-{i}_ss.json', "w") as file:
            json.dump(results_list, file)


# Loads the train and test splist from the "splits" json file
def load_test_and_train():
    # Loads the train and test splist from the "splits" json file

    #path to file
    path = "Potholes/splits.json"
    
    # Load the JSON file
    with open(path, 'r') as f:
        data = json.load(f)

    # Separate the train and test data
    train_data = data["train"]
    test_data = data["test"]

    return train_data, test_data

#reads xml file
def read_content(xml_file: str):
    # Reads content of given xml file
    # Return image name and list of bounding box values

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

# plots image from data with its ground truth bounding boxes
def plot_image_with_boxes(image_path):
    """
    Plots an image with bounding boxes in xmin, ymin, xmax, ymax format.
    
    Parameters:
    - image_path (str): Path to the image file.
    - boxes (list): List of bounding boxes, where each box is [xmin, ymin, xmax, ymax].
    """
    path = "Potholes/annotated-images/"
    name, boxes = read_content(path + image_path)

    # Load image
    image = Image.open(path + name)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Add each bounding box as a rectangle
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        # Create a rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        # Add the rectangle to the plot
        ax.add_patch(rect)

#opens and returns image given its name in string format
def return_image(image_path):
    path = "Potholes/annotated-images/"
    name, boxes = read_content(path + image_path)

    # Load image
    return Image.open(path + name)

#returns path to image
def return_image_path(image_path):
    path = "Potholes/annotated-images/"
    name, boxes = read_content(path + image_path)

    # Load image
    return path + name

#converts from [x,y,w,h] to [xmin, ymin, xmax, ymax]
def wh_to_minmax(x, y, width, height):
    # Converts the bounding box values, x, y, width to xmin, ymin, xmax, ymax
    xmin = x
    ymin = y
    xmax = x + width
    ymax = y + height
    return xmin, ymin, xmax, ymax

#converts from [xmin, ymin, xmax, ymax] to [x,y,w,h] 
def minmax_to_wh(xmin, ymin, xmax, ymax):
    # Converts the bounding box values [xmin, ymin, xmax, ymax] to  ]x, y, width]
    x = xmin
    y = ymin
    w = xmax - xmin
    h = ymax - ymin
    return x, y, w, h

# compares proposal BB with all GT BB from image. Also sorts them using k1 and k2
def compare_bb(image_path, bb, k1, k2):
    # NOT USED 
    proposals = []
    background = []

    path = "Potholes/annotated-images/"
    name, boxes = read_content(path + image_path)


    # Load image
    image = Image.open(path + name)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        box2 = [xmin, ymin, xmax, ymax]
        width = xmax - xmin
        height = ymax - ymin
        # Create a rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='black', facecolor='none')
        # Add the rectangle to the plot
        ax.add_patch(rect)
        
        for box1 in bb:
            iou = calculate_iou(box1, box2)
            if iou > k2:
                proposals.append(box1)
            elif iou < k1:
                background.append(box1)
    
    for box in proposals:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        # Create a rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        # Add the rectangle to the plot
        ax.add_patch(rect)

    plt.show()
    image = Image.open(path + name)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box in background:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        # Create a rectangle patch
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        # Add the rectangle to the plot
        ax.add_patch(rect)


    return proposals, background



## ____________ Dataset creation _______________________
# functions used fro creating the fina dataset.

#reads json file from path "Potholes/annotated-images/"
def read_json(file_path):
    # Reads the json file.
    # Takes only name of file as input

    path = "Potholes/annotated-images/"
    with open(path + file_path, 'r') as file:
        return json.load(file)

#Splist list into three lists with different 'class' values
def split_json_by_class(json):
    #takes output from "read_json()" as input

    class_0 = [entry for entry in json if entry.get("class") == 0]
    class_1 = [entry for entry in json if entry.get("class") == 1]
    class_none = [entry for entry in json if entry.get("class") is None]
    
    return class_0, class_1, class_none

#Merges two lists
def merge_classes(list_0, list_1):
    # mearges two lists
    return list_0 + list_1

#finds id numbers for training data.
def get_id(string):
    # gets id's of training or test data
    # Input is output from "load_test_and_train()" function

    def extract_number(string):
        match = re.search(r'\d+', string)
        return int(match.group()) if match else None
    
    id = [extract_number(s) for s in string]
    return id

#reads all "img-1_ss.json" typed files in the training set and returns
# three lists  with different 'class' values
def create_1_0_none_splits():
    sets = []
    class_0 = []
    class_1 = []
    class_none = []
    train, _ = load_test_and_train("Potholes/splits.json")

    Ids = get_id(train)
    
    for id in Ids:
        name = "img-"+str(id)+"_ss.json"
        json = read_json(name)
        cl_0, cl_1, cl_none = split_json_by_class(json)
        class_0 = merge_classes(class_0,cl_0)
        class_1 = merge_classes(class_1,cl_1)
        class_none = merge_classes(class_none,cl_none)

    # class_0 has length:     621147
    # class_1 has length:     5256
    # class_none has length:  31028

    return class_0, class_1, class_none

# given output from "create_1_0_none_splits()" function creates a list of size batch_size 
# with p1 % background proposals and 1-p1 pothole proposals.
def create_batch(class_0, class_1, batch_size, p1):

    num_background = int(batch_size * (p1))
    num_class = batch_size - num_background

    background = random.sample(class_0, min(num_background, len(class_0)))
    class1 = random.sample(class_1, min(num_class, len(class_1)))

    return merge_classes(background,class1)



# Loads image and takes all its proposals and makes crops of them. 
#Returns batch of 75/25 split of negative/positive proposals
#(also resizes)
def load_and_crop_image(dim, path, class_1,class_0,id):
    # Resize dimensions
    n = dim[0]
    m = dim[1]  # Replace with desired dimensions

    #finds how many positive proposals and negative proposals so we have a 75/25 split
    num_of_class_1 = len(class_1)
    num_of_class_0 = 3*num_of_class_1

    # Load the image
    image = Image.open(path+"img-"+str(id)+".jpg")

    class_0_ran = random.sample(class_0, num_of_class_0)

    # Initialize X_batch and Y_batch
    X_batch = []
    Y_batch = []

    # Process each annotation
    for annotation in class_1:
        bbox = annotation['bbox']
        class_value = annotation['class']

        # Crop the image using the bounding box
        crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            
        # Resize the crop
        
        resized_crop = crop.resize((n, m), Image.LANCZOS)
        # Convert to a tensor (normalizing pixel values to [0, 1])
        tensor_crop = torch.tensor(np.array(resized_crop), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Add the tensor crop and class value to the respective batches
        X_batch.append(tensor_crop)
        Y_batch.append(class_value)

    for annotation in class_0_ran:
        bbox = annotation['bbox']
        class_value = annotation['class']

        # Crop the image using the bounding box
        crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        #print(f"Original crop size: {crop.size}")
        # Resize the crop
        # Resize the crop
        
        resized_crop = crop.resize((n, m), Image.LANCZOS)
        
        #print(f"Resized crop size: {resized_crop.size}")
        #print("")
        # Convert to a tensor (normalizing pixel values to [0, 1])
        tensor_crop = torch.tensor(np.array(resized_crop), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Add the tensor crop and class value to the respective batches
        X_batch.append(tensor_crop)
        Y_batch.append(class_value)
    


    # Stack X_batch into a single tensor for batched processing
    X_batch = torch.stack(X_batch)
    Y_batch = torch.tensor(Y_batch, dtype=torch.long)

    # Shuffle X_batch and Y_batch
    indices = torch.randperm(X_batch.size(0))  # Generate shuffled indices
    X_batch = X_batch[indices]
    Y_batch = Y_batch[indices]

    return X_batch, Y_batch


#Extracts the number from a image name
def extract_number(string):
        match = re.search(r'\d+', string)
        return int(match.group()) if match else None



