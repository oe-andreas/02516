import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

identity = lambda x : x

class retinal(torch.utils.data.Dataset):
    def __init__(self, train = True, transform = identity, label_transform = None, indeces = np.arange(21,41), data_path='/dtu/datasets1/02516/DRIVE/training'):
        'Initialization. indeces should be those integers between 21 and 40 that are to be included in this loader'
        self.transform = transform
        self.label_transform = label_transform if label_transform is not None else transform
        
        self.image_paths = [os.path.join(data_path, 'images', f'{i:02}_training.tif') for i in indeces]
        self.label_paths = [os.path.join(data_path, '1st_manual', f'{i:02}_manual1.gif') for i in indeces]
        
        #not currently used
        self.mask_paths = [os.path.join(data_path, 'mask', f'{i:02}_training_mask.gif') for i in indeces]
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.train:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                image = transforms.functional.hflip(image)
                label = transforms.functional.hflip(label)

            # Random vertical flip
            if np.random.rand() > 0.5:
                image = transforms.functional.vflip(image)
                label = transforms.functional.vflip(label)
            # Generate a random rotation angle
            angle = np.random.uniform(0, 360)
            image = transforms.functional.rotate(image, angle)
            label = transforms.functional.rotate(label, angle)

        Y = self.transform(label)
        X = self.transform(image)
        return X, Y
    
class retinal_test_no_labels(torch.utils.data.Dataset):
    def __init__(self, transform = identity, label_transform = None, indeces = np.arange(1,21), data_path='/dtu/datasets1/02516/DRIVE/test'):
        'Initialization. indeces should be those integers between 1 and 20 that are to be included in this loader. Note: The dataset does not have labels.'
        self.transform = transform
        
        self.label_transform = label_transform if label_transform is not None else transform
        
        self.image_paths = [os.path.join(data_path, 'images', f'{i:02}_test.tif') for i in indeces]
        
        #not currently used
        self.mask_paths = [os.path.join(data_path, 'mask', f'{i:02}_test_mask.gif') for i in indeces]
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        X = self.transform(image)
        return X
    