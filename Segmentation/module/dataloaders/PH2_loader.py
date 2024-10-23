import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


identity = lambda x : x

class PH2(torch.utils.data.Dataset):
    def __init__(self, train = True, transform = identity, label_transform = None, indeces = np.arange(2, 438), data_path = '/dtu/datasets1/02516/PH2_Dataset_images'):
        'Initialization. Assumes the "lesion" folders contain the labels'
        self.transform = transform
        self.train = train
        self.label_transform = label_transform if label_transform is not None else transform
        self.image_paths = [os.path.join(data_path, f'IMD{i:03}', f'IMD{i:03}_Dermoscopic_Image', f'IMD{i:03}.bmp') for i in indeces]
        self.label_paths = [os.path.join(data_path, f'IMD{i:03}', f'IMD{i:03}_lesion', f'IMD{i:03}_lesion.bmp') for i in indeces]
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        if self.train:
            # Generate a random rotation angle
            angle = np.random.uniform(0, 360)
            image = transforms.functional.rotate(image, angle)
            label = transforms.functional.rotate(label, angle)
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y