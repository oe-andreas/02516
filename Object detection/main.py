
"""
# Loads dataloader class
# from module.dataloaders.loader import load_images

#models input size
#dim = [128,128]

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
"""


from module.dataloaders.loader import load_images_fixed_batch
train = True
batch_size = 64
dim = [128,128]
loader = load_images_fixed_batch(train,dim=dim,batch_size=batch_size)

for X_batch, Y_batch, gt_batch, t_batch in loader:
    print("X shape: ",X_batch.shape)
    print("Y shape: ",Y_batch.shape)
    print("gt shape: ",gt_batch.shape)
    print("t shape: ",t_batch.shape)