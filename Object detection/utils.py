import cv2
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET

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

def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ss_results = ss.process()
    return ss_results

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



def load_test_and_train(path):
    # Load the JSON file
    with open(path, 'r') as f:
        data = json.load(f)

    # Separate the train and test data
    train_data = data["train"]
    test_data = data["test"]

    return train_data, test_data

def read_content(xml_file: str):

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

def return_image(image_path):
    path = "Potholes/annotated-images/"
    name, boxes = read_content(path + image_path)

    # Load image
    return Image.open(path + name)

def return_image_path(image_path):
    path = "Potholes/annotated-images/"
    name, boxes = read_content(path + image_path)

    # Load image
    return path + name

def convert_bbox(x, y, width, height):
    xmin = x
    ymin = y
    xmax = x + width
    ymax = y + height
    return xmin, ymin, xmax, ymax

def corerct_bbs(bbs):


    def return_edge_box_proposals(image, max_boxes=100, min_score=0.01):
    if image is None or not isinstance(image, np.ndarray):
        return

    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(max_boxes)
    edge_boxes.setMinScore(min_score)

    edges = cv2.Canny(image, 50, 150)
    edges = edges.astype(np.float32)  # Convert edge map to CV_32F
    orientation_map = np.zeros_like(edges, dtype=np.float32)  # Placeholder orientation map as CV_32F

    bbs = edge_boxes.getBoundingBoxes(edges, orientation_map)[0]

   
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)

    for bb in bbs:
        x, y, w, h = bb
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis("off")
    plt.show()
    return bbs

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def compare_bb(image_path, bb, k1, k2):
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

