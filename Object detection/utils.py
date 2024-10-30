import cv2

def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    ss_results = ss.process()
    return ss_results



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

