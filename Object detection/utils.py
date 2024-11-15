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
import os


#map from efficientnet models to good input sizes
def get_input_size(model_name):
    match model_name:
        case "efficientnet_b0":
            return 224
        case "efficientnet_b1":
            return 240
        case "efficientnet_b2":
            return 260
        case "efficientnet_b3":
            return 300
        case "efficientnet_b4":
            return 380
        case "efficientnet_b5":
            return 456
        case "efficientnet_b6":
            return 528
        case "efficientnet_b7":
            return 600
        case _:
            raise ValueError("Invalid model name. Please choose from 'efficientnet_b0' to 'efficientnet_b7', or add another model to get_input_size.")

#Reads xml files
def parse_xml(xml_file):
    #input: xml_file object
    #output: list of bounding boxes for potholes in [xmin, ymin, xmax, ymax] format
    
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
    #input: two bounding boxes in [xmin, ymin, xmax, ymax] format
    #output: iou value
    
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
    # Ensure the denominator is not zero before performing the division
    denominator = float(boxAArea + boxBArea - interArea)
    if denominator > 0:
        iou = interArea / denominator
    else:
        print("!!!!!!!!!!!Something bad happend!!!!!!!!!!!")
        iou = 0  # or set it to a default value (e.g., 0 if no intersection)

    return iou

# Does selective search for a single image
def selective_search(image):
    #input: image object
    #output: list of bounding boxes in [x, y, width, height] format. OBS! NOT [xmin, ymin, xmax, ymax]
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ss_results = ss.process()
    return ss_results

# Does selective search for all images
def selective_search_all():
    #computes SS proposals for all images and saves them in json files
    #storage format: For image i, stores 'img-i_ss.json', which contains a list of dicts. Each dict is
    #{
    #           'bbox': [x_min, y_min, x_max, y_max], #list of ints
    #           'class': pothole_class,  #0, 1 or None. 0 if max_iou < 0.3, 1 if max_iou > 0.7, None otherwise
    #           'iou': max_iou #the max iou value between the bbox and any GT pothole bbox
    #       }
    
    
    
    for i in tqdm(range(1, 666), desc="Processing Images", unit="image"):
        # Load the image
        image = cv2.imread(f'Potholes/annotated-images/img-{i}.jpg', 1) #,1 means read as color image

        # Run selective search
        ss_results = selective_search(image)

        # Parse the XML to get the pothole bounding boxes
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
    #output: train_data, test_data where each is a list of strings

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
    
    # duplicate of parse_xml function, except this one returns image name as well

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
def plot_image_with_boxes(image_path, additional_boxes = None):
    """
    Plots an image with GT bounding boxes in xmin, ymin, xmax, ymax format.
    
    Parameters:
    - image_path (str): Basename of the image file, such as "img-1.jpg".
    - additional_boxes (list of lists): List of additional bounding boxes to plot, each in [xmin, ymin, xmax, ymax] format.
    """
    path = "Potholes/annotated-images/"
    name, boxes = read_content(path + image_path)

    boxes = boxes + additional_boxes if additional_boxes is not None else boxes
    
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
    # Converts the bounding box values, x, y, width, height to xmin, ymin, xmax, ymax
    xmin = x
    ymin = y
    xmax = x + width
    ymax = y + height
    return xmin, ymin, xmax, ymax

#converts from [xmin, ymin, xmax, ymax] to [x,y,w,h] 
def minmax_to_wh(xmin, ymin, xmax, ymax):
    # Converts the bounding box values [xmin, ymin, xmax, ymax] to  [x, y, width, height]
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
def load_and_crop_image(dim, path, class_1, class_0, id,gtbbox,t_vals):
    #inputs
    # dim: tuple of two integers, desired dimensions after resizing
    # path: path at which to find image files (i.e. "Potholes/annotated-images/")
    # class_1: list of dictionaries with 'bbox' and 'class' keys for positive proposals
    # class_0: list of dictionaries with 'bbox' and 'class' keys for negative proposals
    # id: integer, image id number
    
    
    # Resize dimensions
    n = dim[0]
    m = dim[1]  # Replace with desired dimensions

    # Finds how many positive proposals and negative proposals so we have a 75/25 split
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
        
        # Resize the cropped image to (n,m)
        resized_crop = crop.resize((n, m), Image.LANCZOS)

        # Convert to a tensor (normalizing pixel values to [0, 1])
        tensor_crop = torch.tensor(np.array(resized_crop), dtype=torch.float32).permute(2, 0, 1) / 255.0 #permute to Channels x Height x Width

        # Add the tensor crop and class value to the respective batches
        X_batch.append(tensor_crop)
        Y_batch.append(class_value)

    for annotation in class_0_ran:
        bbox = annotation['bbox']
        class_value = annotation['class']

        # Crop the image using the bounding box
        crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        #print(f"Original crop size: {crop.size}")

        # Resize the cropped image to (n,m)
        resized_crop = crop.resize((n, m), Image.LANCZOS)
        
        #print(f"Resized crop size: {resized_crop.size}")
        #print("")
        # Convert to a tensor (normalizing pixel values to [0, 1])
        tensor_crop = torch.tensor(np.array(resized_crop), dtype=torch.float32).permute(2, 0, 1) / 255.0


        #?? ved ik om vi skal gøre det på den her måde
        gtbbox.append(torch.tensor(bbox, dtype=torch.long)) #So gtbbox is its own bbox
        t_vals.append(torch.tensor([0,0,0,0], dtype=torch.long)) # tvals are zero

        # Add the tensor crop and class value to the respective batches
        X_batch.append(tensor_crop)
        Y_batch.append(class_value)
    
    gtbbox = torch.stack(gtbbox)
    t_vals = torch.stack(t_vals)

    # Stack X_batch into a single tensor for batched processing
    X_batch = torch.stack(X_batch)
    Y_batch = torch.tensor(Y_batch, dtype=torch.long)

    # Shuffle X_batch and Y_batch
    indices = torch.randperm(X_batch.size(0))  # Generate shuffled indices
    X_batch = X_batch[indices]
    Y_batch = Y_batch[indices]

    gtbbox_batch = gtbbox[indices]
    t_vals_batch = t_vals[indices]

    return X_batch, Y_batch, gtbbox_batch, t_vals_batch



#Extracts the number from a image name
def extract_number(string):
    #as get_id function, but for a single string
    
    match = re.search(r'\d+', string)
    return int(match.group()) if match else None


def alter_box(proposal_box, t):
    #proposal_box should be in [xmin, ymin, xmax, ymax] format
    #t should be tx, ty, tw, th as defined on the slides
    #see also compute_t
    
    px, py, pw, ph = minmax_to_wh(*proposal_box)
    tx, ty, tw, th = t
    
    bx = px + pw*tx
    by = py + ph*ty
    bw = pw*np.exp(tw)
    bh = ph*np.exp(th)
    
    return wh_to_minmax(bx, by, bw, bh)
    

def compute_t(true_box, proposal_box):
    #both boxes should be in [xmin, ymin, xmax, ymax] format
    #returns tx, ty, tw, th as defined on the slides
    
    bx, by, bw, bh = minmax_to_wh(*true_box)
    px, py, pw, ph = minmax_to_wh(*proposal_box)
    
    tx = (bx - px)/pw
    ty = (by - py)/ph
    tw = np.log(bw/pw)
    th = np.log(bh/ph)
    
    return tx, ty, tw, th
    
    
    

def update_json_classes(folder_path, k1, k2, custom_ending="_updated"):
# Define the path to the annotated images folder
    annotated_folder = os.path.join(folder_path, "annotated-images")
    
    # Iterate over each JSON file in the annotated-images folder
    for filename in os.listdir(annotated_folder):
        if filename.endswith("_ss.json"):
            file_path = os.path.join(annotated_folder, filename)
            
            # Open and load the JSON data
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # Update the class field based on iou values
            for obj in data:
                if obj['iou'] < k1:
                    obj['class'] = 0
                elif obj['iou'] > k2:
                    obj['class'] = 1
                else:
                    obj['class'] = None
            
            # Create the new filename with the custom ending
            new_filename = filename.replace("_ss.json", f"{custom_ending}.json")
            new_file_path = os.path.join(annotated_folder, new_filename)
            
            # Save the updated data to the new file
            with open(new_file_path, 'w') as file:
                json.dump(data, file, indent=4)

    print("JSON files updated successfully.")





