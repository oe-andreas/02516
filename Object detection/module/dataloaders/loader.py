from xml.etree import ElementTree as ET
import os
import torch
import random
from PIL import Image
import numpy as np
from utils import read_json, split_json_by_class, read_content, extract_number
from utils import load_and_crop_image, load_test_and_train
from utils import calculate_iou, compute_t

class load_images():
    # a generator that yields batches. A batch consists of every positive proposal and a random sample of negative proposals FROM THE SAME IMAGE
    
    def __init__(self, train = True, dir = "Potholes/splits.json", dim = [128,128], xmls = None):
        """
        Loads list of training or test image names. 
        Also initializes the crop dimension
        
        xmls: If None, use train/test split. Otherwise, list of xml file names to load: ['img-1.xml', 'img-2.xml', ...].
        """
        
        if xmls is None:
            # loads a list of training image names and test image names
            train_data, test_data = load_test_and_train()
            
            #given train input we define what data we use.
            if train:
                self.data = train_data
            else:
                self.data = test_data   
                
        
        else:
            self.data = xmls
        
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




class load_images_fixed_batch():
    #as above, except for fixed batch size
    
    def __init__(self, train = "train", dir = "Potholes/splits.json", dim = [128,128], batch_size = 64, xmls = None):
        """
        Loads list of training or test image names. 
        Also initializes the crop dimension
        
        xmls: If None, use train/test split. Otherwise, list of xml file names to load: ['img-1.xml', 'img-2.xml', ...].
        """
        self.dim = dim
        self.batch_size = batch_size
        
        if xmls is None:
            # Loads a list of training image names and test image names
            train_data, test_data = load_test_and_train()
            
            #splits test_data into two parts
            mid = len(test_data) // 2
            test = test_data[:mid]
            val = test_data[mid:]
            # Given train input we define what data we use.
            if train == "train":
                self.data = train_data
            elif train == "test":
                self.data = test 
            else:
                self.data = val 
        else:
            self.data = xmls
        
        
        self.len = len(self.data)
        
        # Initialize lists to collect tensors
        X_batches = []
        Y_batches = []
        gtbbox_batch = []
        t_vals_batch = []

        # Loops over each image
        for idx in range(self.len):
            # Loads name of image
            image = self.data[idx]  

            gtbbox = []
            t_vals = []

            # Gets id number of image 
            id = extract_number(image)

            # Reads images corresponding "img-{id}_ss.json" file
            json = read_json("img-"+str(id)+"_ss.json")
            
            # Reads "img-{id}.xml" file
            path = "Potholes/annotated-images/"
            _, list_with_all_boxes = read_content(path+image)

            # Splits all proposals into three. one for background, foreground and none
            class_0, class_1, class_none = split_json_by_class(json)
            
            # Loops over each positive proposal 
            for prop in class_1:
                # Initialize best iou and gtbbox
                best_iou = 0
                best_btbbox = 0
                prop_bbox = prop['bbox']
                
                # Loop over all gt bbox in image
                for bbox in list_with_all_boxes:
                    # Computes the iou
                    iou = calculate_iou(prop_bbox,bbox)
                    # Updates best gtbbox
                    if iou > best_iou:
                        best_iou = iou
                        best_btbbox = bbox
                
                # Saves the best bbox to a list
                gtbbox.append(torch.tensor(best_btbbox, dtype=torch.long))

                # Computes and saves the gt t-values
                t_vals.append(torch.tensor(compute_t(best_btbbox,prop_bbox), dtype=torch.long) )


            # Adds ground truth BB to the set of positive proposals.
            # We give ground truth BB class=1 and IOU = 1
            for bbox in list_with_all_boxes:
                class_1.append({'bbox': bbox, 'class': 1, 'iou': 1.0})

                #?? ved ik om vi skal gøre det på den her måde
                gtbbox.append(torch.tensor(bbox, dtype=torch.long))
                t_vals.append(torch.tensor([0,0,0,0], dtype=torch.long)) #?? ved ik om vi skal gøre det på den her måde
            
            #loads proposals and their class value
            X_batch, Y_batch, gtbbox, t_vals = load_and_crop_image(self.dim, path, class_1, class_0, id, gtbbox, t_vals)
            # Append tensors to the lists
            X_batches.append(X_batch)
            Y_batches.append(Y_batch)


            #print("gtbos: ",gtbbox)
            #print("t: ",t_vals)
            gtbbox_batch.append(gtbbox)
            t_vals_batch.append(t_vals)
        #image loop ends:

        
       
        
        #print(X_batches[0])
        #print(Y_batches[0])
        # Stack all tensors after the loop
        X_stacked = torch.cat(X_batches, dim=0)  # Concatenate along the batch dimension
        Y_stacked = torch.cat(Y_batches, dim=0)  # Concatenate along the batch dimension
        gtbbox_stacked = torch.cat(gtbbox_batch, dim=0)  # Concatenate along the batch dimension
        t_stacked = torch.cat(t_vals_batch, dim=0)  # Concatenate along the batch dimension

        # Shuffle the indices
        indices = torch.randperm(X_stacked.size(0))

        # Split indices into batches
        batches = [indices[i:i + batch_size] for i in range(0, X_stacked.size(0), self.batch_size)]

        # Create the batched lists
        X_batches = [X_stacked[batch] for batch in batches]
        Y_batches = [Y_stacked[batch] for batch in batches]

        gtbbox_batches = [gtbbox_stacked[batch] for batch in batches]
        t_batches = [t_stacked[batch] for batch in batches]

        self.X_data = X_batches
        self.Y_data = Y_batches
        self.gtbbox = gtbbox_batches
        self.t = t_batches

        # Output number of batches
        #print(f"Number of batches: {len(X_batches)}")
        #print(f"First batch X shape: {X_batches[0].shape}")
        #print(f"First batch Y shape: {Y_batches[0].shape}")


        #print(X_stacked.shape)
        #print(Y_stacked.shape)

    
    
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
        X_batch = self.X_data[idx]
        Y_batch = self.Y_data[idx]
        gtbbox_batch = self.gtbbox[idx] 
        t_batch = self.t[idx] 
        

        return X_batch, Y_batch, gtbbox_batch, t_batch





class Dataloader():
    # a generator that yields batches. A batch consists of every positive proposal and a random sample of negative proposals FROM THE SAME IMAGE
    
    def __init__(self, train = "train", dir = "Potholes/splits.json", dim = [128,128], batch_size = 64, xmls = None):
        
        self.dim = dim
        self.batch_size = batch_size

        #Load list with training and test images
        train_data, test_data = load_test_and_train()
        
        #saves test, train or val as our data
        if train == "train":
            self.data = train_data
        elif train == "test":
            mid = len(test_data) // 2
            self.data = test_data[:mid]
        else:
            mid = len(test_data) // 2
            self.data =test_data[mid:]
        
        self.len = len(self.data)

        #Shuffles data images around. 
        random.shuffle(self.data)
        total_proposals = []
        num_of_proposals = []
        #Loop over all images:
        for idx in range(self.len):
            image_proposals = []
            # Loads name of image
            image_name = self.data[idx]
            #extracs id number of image:
            id = extract_number(image_name)

            # Reads images corresponding "img-{id}_ss.json" file
            json = read_json("img-"+str(id)+"_ss.json")

            # Reads "img-{id}.xml" file
            path = "Potholes/annotated-images/"
            _, list_with_all_gtboxes = read_content(path+image_name)
            
            #filter tiny boxes
            list_with_all_gtboxes = [box for box in list_with_all_gtboxes if (box[2]-box[0]) >= 1 and (box[3]-box[1]) >= 1]

            # Splits all proposals into three. one for background, foreground and none
            class_0, class_1, class_none = split_json_by_class(json)

            n_pos = len(class_1) + len(list_with_all_gtboxes)

            #loop over positive proposals:
            for annotation in class_1:
                bbox = annotation['bbox']
                class_value = annotation['class']

                best_iou = 0
                best_gtbbox = 0
                prop_bbox = bbox
                
                # Loop over all gt bbox in image
                for gtbbox in list_with_all_gtboxes:
                    # Computes the iou
                    iou = calculate_iou(prop_bbox,gtbbox)
                    # Updates best gtbbox
                    if iou > best_iou:
                        best_iou = iou
                        best_gtbbox = gtbbox

                tvals = compute_t(best_gtbbox,prop_bbox)

                proposal = {
                    "id": id,
                    "bbox" : bbox,
                    "gtbbox": best_gtbbox,
                    "class" : class_value,
                    "t_val" : tvals
                }
                image_proposals.append(proposal)

            #loop over gt bbox and add dem as proposals:
            for gtbbox in list_with_all_gtboxes:
                proposal = {
                    "id": id,
                    "bbox" : gtbbox,
                    "gtbbox": gtbbox,
                    "class" : 1,
                    "t_val" : [0,0,0,0]
                }
                image_proposals.append(proposal)

            n_neg = 3*n_pos
            class_0_ran = random.sample(class_0, n_neg)

            for annotation in class_0_ran:
                bbox = annotation['bbox']
                class_value = annotation['class']

                proposal = {
                    "id": id,
                    "bbox" : bbox,
                    "gtbbox": bbox,
                    "class" : class_value,
                    "t_val" : [0,0,0,0]
                }
                image_proposals.append(proposal)
            
            #Shuffle proposals from image around
            random.shuffle(image_proposals)
            #add proposals to the list of all proposals
            total_proposals = total_proposals + image_proposals

        #create batches from total_proposals list
        batches = [total_proposals[i:i + self.batch_size] for i in range(0, len(total_proposals), self.batch_size)]

        self.batch = batches
        

    def __len__(self):
        'Returns the total number of samples'
        return len(self.data)

    def __getitem__(self, idx):
        
        #Gets batch corresponding to idx 
        batch = self.batch[idx]

        #length of batch
        n = len(batch)

        #initialization: 
        X_batch = []
        Y_batch = []
        bbox_batch = []
        gt_bbox_batch = []
        tvals_batch = []

        #loop over proposals in batch
        for i in range(n):
            #gets info from proposal i in the batch
            id = batch[i]["id"]
            bbox = batch[i]["bbox"]
            gt_bbox = batch[i]["gtbbox"]
            y = batch[i]["class"]
            t_val = batch[i]["t_val"]

            # Load the image
            path = "Potholes/annotated-images/"
            image = Image.open(path+"img-"+str(id)+".jpg")

            #Crop image
            crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

            #Rezise crop
            resized_crop = crop.resize((self.dim[0], self.dim[1]), Image.LANCZOS)

            #Trun rezised crop into tensor and normalize values
            tensor_crop = torch.tensor(np.array(resized_crop), dtype=torch.float32).permute(2, 0, 1) / 255.0
            
            #Save all info as tensors
            X_batch.append(tensor_crop)
            Y_batch.append(torch.tensor(y))
            bbox_batch.append(torch.tensor(bbox))
            gt_bbox_batch.append(torch.tensor(gt_bbox))
            tvals_batch.append(torch.tensor(t_val))

        #Stack everything
        X_batch = torch.stack(X_batch)
        Y_batch =  torch.stack(Y_batch)
        bbox_batch =  torch.stack(bbox_batch)
        gt_bbox_batch =  torch.stack(gt_bbox_batch)
        tvals_batch =  torch.stack(tvals_batch)

        return X_batch, Y_batch, bbox_batch, gt_bbox_batch, tvals_batch







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