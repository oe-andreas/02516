from .dataloaders.retinal_loader import retinal
import numpy as np

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

im_size = 512

train_transform = transforms.Compose([transforms.Resize((im_size, im_size)), 
                                    transforms.ToTensor()])


retinal_train = retinal(indeces = np.arange(21,33), transform = train_transform, train = True, data_path = '/Users/andreas/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Offline og Online/UNI/9. semester/02516 DLinCV/Segmentation/DRIVE/training')

X, Y, Z = retinal_train.__getitem__(0)

Y_masked = Y[Z > 0]
print(Y_masked.shape)

#show the image
plt.imshow(Y[0])
plt.show()

plt.hist(Y_masked.flatten())
plt.show()


