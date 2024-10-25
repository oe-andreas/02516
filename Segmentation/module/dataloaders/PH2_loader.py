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
        Z = np.ones_like(Y)
        return X, Y, Z


class PH2_weak(torch.utils.data.Dataset):
    def __init__(self, sample_info = [5,0,0], train = True, transform = identity, label_transform = None, indeces = np.arange(2, 438), data_path = 'module/PH2_Dataset_images'):
        'Initialization. Assumes the "lesion" folders contain the labels'
        self.transform = transform
        self.train = train
        self.label_transform = label_transform if label_transform is not None else transform
        self.image_paths = [os.path.join(data_path, f'IMD{i:03}', f'IMD{i:03}_Dermoscopic_Image', f'IMD{i:03}.bmp') for i in indeces]
        self.label_paths = [os.path.join(data_path, f'IMD{i:03}', f'IMD{i:03}_lesion', f'IMD{i:03}_lesion.bmp') for i in indeces]
        
        #stores information about weak annotation sampling
        self.sample_info = sample_info
        
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


        #################################################################################
        # Computes weak annotations:
        #################################################################################

        #Gets weak annotations
        zero_sampled, one_sampled = sample_pixels_dist(Y.cpu().squeeze(),self.sample_info[0], self.sample_info[1], self.sample_info[2])
        
        # Creates tensor of 1's and 0's that is length of number of weak annotations (ground truth of annotation)
        zero = torch.zeros(zero_sampled.size(0), 1, dtype=zero_sampled.dtype)
        one = torch.ones(one_sampled.size(0), 1, dtype=one_sampled.dtype)

        #Merges annotations and 1 and 0 tensor, so we have annotation and ground truth together
        zero_sampled = torch.cat((zero_sampled, zero), dim=1)
        one_sampled = torch.cat((one_sampled, one), dim=1)

        #Collects all weak annotations in a tensor.
        labeled_points = torch.cat((zero_sampled, one_sampled), dim=0)
    

        return X, Y, labeled_points

 
    
