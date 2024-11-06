from xml.etree import ElementTree as ET
import os

class load_images():
    def __init__(self, train = True, dir = "Potholes"):
        # load splits.json
        with open(os.path.join(dir, 'splits.json')) as f:
            splits = json.load(f)
            
        # extract xml file names
        keyword = "train" if train else "test"
        self.xml_names = splits[keyword]        
    
    
    def __len__(self):
        'Returns the total number of samples'
        return len(self.xml_names)

    def __getitem__(self, idx):
        #open xml file
        xml_file = self.xml_names[idx]
        
        #parse xml file
        tree = ET.parse(os.path.join(dir, 'annotated-images', xml_file))
        root = tree.getroot()
        
        filename = root.find('filename').text

        list_with_all_boxes = []
        
        #strip .xml and replace with .jpg
        filename = filename[:-4] + ".jpg"
        
        image = cv2.imread(os.path.join(dir, 'annotated-images', filename))

        for box in root.iter('object'):

            ymin, xmin, ymax, xmax = None, None, None, None

            ymin = int(box.find("bndbox/ymin").text)
            xmin = int(box.find("bndbox/xmin").text)
            ymax = int(box.find("bndbox/ymax").text)
            xmax = int(box.find("bndbox/xmax").text)

            list_with_single_boxes = [xmin, ymin, xmax, ymax]
            list_with_all_boxes.append(list_with_single_boxes)

        return image, list_with_all_boxes
    
    
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