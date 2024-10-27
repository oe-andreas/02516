import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

identity = lambda x : x

class PH2(torch.utils.data.Dataset):

    def __init__(self, train = True, transform = identity, label_transform = None, 
                 indeces = np.arange(2, 438), data_path = '/dtu/datasets1/02516/PH2_Dataset_images',normalize = None):
        'Initialization. Assumes the "lesion" folders contain the labels'
        self.transform = transform
        self.train = train
        self.label_transform = label_transform if label_transform is not None else transform
        self.image_paths = [os.path.join(data_path, f'IMD{i:03}', f'IMD{i:03}_Dermoscopic_Image', f'IMD{i:03}.bmp') for i in indeces]
        self.label_paths = [os.path.join(data_path, f'IMD{i:03}', f'IMD{i:03}_lesion', f'IMD{i:03}_lesion.bmp') for i in indeces]
        # Define normalization transform
        if normalize is not None:
            # Unpack the mean and std from the tuple
            self.normalize_transform = transforms.Normalize(mean=normalize[0], std=normalize[1])
        else:
            self.normalize_transform = identity 
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

        # Apply normalization
        X = self.normalize_transform(X)
        return X, Y, Z


class PH2_weak(torch.utils.data.Dataset):
    def __init__(self, sample_info = [5,0,0], train = True, transform = identity, 
                 label_transform = None, indeces = np.arange(2, 438), 
                 data_path = '/dtu/datasets1/02516/PH2_Dataset_images', normalize = None):
        'Initialization. Assumes the "lesion" folders contain the labels'
        self.transform = transform
        self.train = train
        self.label_transform = label_transform if label_transform is not None else transform
        self.image_paths = [os.path.join(data_path, f'IMD{i:03}', f'IMD{i:03}_Dermoscopic_Image', f'IMD{i:03}.bmp') for i in indeces]
        self.label_paths = [os.path.join(data_path, f'IMD{i:03}', f'IMD{i:03}_lesion', f'IMD{i:03}_lesion.bmp') for i in indeces]
        
        #stores information about weak annotation sampling
        self.sample_info = sample_info

        # Define normalization transform
        if normalize is not None:
            # Unpack the mean and std from the tuple
            self.normalize_transform = transforms.Normalize(mean=normalize[0], std=normalize[1])
        else:
            self.normalize_transform = identity
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

        # Apply normalization
        X = self.normalize_transform(X)

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

 
    

# Samples pixels from ground truth image
def sample_pixels_dist(tensor, num_samples=5, min_distance=10, edge_distance=5):
    """
    Samples pixels from the tensor, ensuring 0-value pixels are a certain distance from edges 
    and both 0-value and 1-value pixels are a minimum distance apart from each other.
    
    Args:
        tensor (torch.Tensor): The input tensor of size [H, W] with values 0 or 1.
        num_samples (int): Number of samples to pick for each class.
        min_distance (int): Minimum distance between sampled pixels.
        edge_distance (int): Minimum distance away from the edges for 0-value pixels.
        
    Returns:
        zero_sampled (torch.Tensor): Sampled 0-value pixel coordinates.
        one_sampled (torch.Tensor): Sampled 1-value pixel coordinates.
    """
    H, W = tensor.shape

    # Find indices of all 0s and 1s
    zero_indices = (tensor == 0).nonzero(as_tuple=False)
    one_indices = (tensor == 1).nonzero(as_tuple=False)

    # Filter 0-value pixels that are at least `edge_distance` away from the edges
    zero_indices = zero_indices[
        (zero_indices[:, 0] >= edge_distance) &  # y >= edge_distance
        (zero_indices[:, 0] < H - edge_distance) &  # y < H - edge_distance
        (zero_indices[:, 1] >= edge_distance) &  # x >= edge_distance
        (zero_indices[:, 1] < W - edge_distance)   # x < W - edge_distance
    ]

    def is_far_enough(selected, candidate, min_dist):
        """Check if candidate pixel is far enough from all selected pixels."""
        if selected.size(0) == 0:
            return True  # No previous points selected, so it's valid
        distances = torch.norm(selected.float() - candidate.float(), dim=1)
        return torch.all(distances >= min_dist)

    def sample_with_distance(indices, num_samples, min_dist):
        """Sample num_samples indices ensuring they are min_dist apart."""
        seed = 42
        torch.manual_seed(seed)
        selected = []
        while len(selected) < num_samples:
            candidate = indices[torch.randint(0, indices.size(0), (1,))].squeeze(0)
            if is_far_enough(torch.stack(selected) if selected else torch.empty((0, 2)), candidate, min_dist):
                selected.append(candidate)
        return torch.stack(selected)
    
    # Randomly sample 0-value and 1-value pixels with distance constraints
    zero_sampled = sample_with_distance(zero_indices, num_samples, min_distance)
    one_sampled = sample_with_distance(one_indices, num_samples, min_distance)
    
    return zero_sampled, one_sampled

