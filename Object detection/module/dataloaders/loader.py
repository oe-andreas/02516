from xml.etree import ElementTree as ET
import os

from utils import read_json, split_json_by_class, read_content, extract_number
from utils import load_and_crop_image, load_test_and_train

class load_images():
    def __init__(self, train = True, dir = "Potholes/splits.json", dim = [128,128]):
        """
        Loads list of training or test image names. 
        Also initializes the crop dimension
        """

        # loads a list of training image names and test image names
        train_data, test_data = load_test_and_train()
        
        #given train input we define what data we use.
        if train:
            self.data = train_data
        else:
            self.data = test_data   
        
        self.dim = dim
    
    
    def __len__(self):
        'Returns the total number of samples'
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return:

        X_batch: size - tensor[n, 3, self.dim[0], self.dim[1]]
        (n = number of negative and positive proposals so their is a 75/25 split between them)
        - X_batch is a tensor of cropped and resized images taken from our original image

        Y_batch: size - tensor[n]
        - Y_batch is just our class value for the given crop

        Note: X_batch and Y_batch are shuffled.
        """

        #loads name of image
        image = self.data[idx]

        #gets id number of image 
        id = extract_number(image)

        #Reads images corresponding "img-{id}_ss.json" file
        json = read_json("img-"+str(id)+"_ss.json")
        
        #Reads "img-{id}.xml" file
        path = "Potholes/annotated-images/"
        _, list_with_all_boxes = read_content(path+image)

        #splits all proposals into three. one for background, foreground and none
        class_0, class_1, class_none = split_json_by_class(json)

        #adds ground truth BB to the set of positive proposals.
        # we give ground truth BB class=1 and IOU = 1
        for bbox in list_with_all_boxes:
            class_1.append({'bbox': bbox, 'class': 1, 'iou': 1.0})
        
        #loads proposals and their class value
        X_batch, Y_batch= load_and_crop_image(self.dim,path,class_1,class_0,id)


        return X_batch, Y_batch



"""   
class PotholesDataset(Dataset):
    def __init__(self, train = True, transform, dir = "Potholes", k1 = 0.3, k2 = 0.7):
        
        assert k2 >= k1
        
        self.transform = transform

        with open(os.path.join(dir, 'splits.json')) as f:
            splits = json.load(f)
            
        # extract xml file names
        keyword = "train" if train else "test"
        xml_names = splits[keyword]
        
        #strip .xml
        image_names = [xml_name[:-4] for xml_name in xml_names]
        
        background_boxes = []
        pothole_boxes = []
        

        #read all bounding boxes from .json files
        for image in image_names:
            with open(os.path.join(dir, 'annotated_images', image + '.json')) as f:
                boxes = json.load(f)
                
            for box in boxes:
            
                if box['iou'] >= k2:
                    pothole_boxes.append(
                        {"box": box['box'], "image": image}
                        )
                elif box['iou'] < k1:
                    background_boxes.append(
                        {"box": box['box'], "image": image}
                        )
                else: #don't import ambigious boxes
                    pass

        
        self.boxes = background_boxes + pothole_boxes
        self.n_background = len(background_boxes)
        self.n_potholes = len(pothole_boxes)
        
    
        
    def __len__(self):
        return len(self.boxes)
    
    def __getitem__(self, idx):
        #when wrapping this in a dataloader, use
        
        # sampler = WeightedRandomSampler(weights=[self.n_potholes/self.__len__(), self.n_background/self.__len__()], num_samples=self.__len__, replacement=True)
        
        image_name = self.boxes[idx]['image']
        box = self.boxes[idx]['box']
        
        image = cv2.imread(os.path.join(dir, 'annotated-images', image_name + '.jpg'))
        
        image = image[box[1]:box[3], box[0]:box[2]] #ymin:ymax, xmin:xmax
        
        image = transform(image)
        
        
        
        
        
    
    

name, boxes = read_content("file.xml")
""" 