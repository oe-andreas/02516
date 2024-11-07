
# Loads dataloader class
from module.dataloaders.loader import load_images

#models input size
dim = [128,128]

#Loads training data
loader_train = load_images(train=True,dim=dim)
#Loads test data
loader_test = load_images(train=False,dim=dim)

print("Number of training images: ", len(loader_train))
print("Number of test images: ", len(loader_test))

#Prints size of the first 5 batches in training data
for i in range(5):
    X_batch , Y_batch =  loader_train[i]
    print(X_batch.shape)