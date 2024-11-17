from utils import load_test_and_train, read_json, read_content, extract_number, compute_t, calculate_iou
from PIL import Image
import torch
import numpy as np
from module.models.efficientnet import EfficientNetWithBBox
from module.processing.non_maximum_suppression import non_maximum_suppression


class Dataloader_test_time():
    #loads per im and does not split on classes
    #X_batch: all SS proposals, cropped and resized, with no class distinctions
    #boxes: The [xmin, ymin, xmax, ymax] coordinates of the proposals in X_batch (same order)
    #gt_bbox: all GT boxes (not in the same order or same length as X_batch)
    
    
    def __init__(self, train = "train", dir = "Potholes/splits.json", dim = [128,128], shuffle_ims = False, shuffle_proposals_in_im = False):
        
        self.dim = dim
        
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
            
        if shuffle_ims:
            np.random.shuffle(self.data)
        
        self.shuffle_proposals_in_im = shuffle_proposals_in_im

            
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
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

        #loop over ALL proposals (no splitting on classes)
        for annotation in json:
            bbox = annotation['bbox']
            
            proposal = {
                "id": id,
                "bbox" : bbox,
            }
            
            image_proposals.append(proposal)
            
        
            
        #now copy the stuff from __getitem__ in the other dataloader
        if self.shuffle_proposals_in_im:
            np.random.shuffle(image_proposals)
        
        batch = image_proposals
        n = len(batch)
        
        X_batch = []
        bbox_batch = []
        
        #loop over proposals in batch
        for i in range(n):
            #gets info from proposal i in the batch
            id = batch[i]["id"]
            bbox = batch[i]["bbox"]

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
            bbox_batch.append(torch.tensor(bbox))

        #Stack everything
        X_batch = torch.stack(X_batch)
        bbox_batch =  torch.stack(bbox_batch)
        

        return X_batch, bbox_batch, np.array(list_with_all_gtboxes), id
        

        