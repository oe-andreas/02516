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

name, boxes = read_content("file.xml")