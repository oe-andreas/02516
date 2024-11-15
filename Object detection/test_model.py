from module.dataloaders.loader import Dataloader
from utils import get_input_size, alter_box, calculate_iou

model_name = 'efficientnet_b0'

import torch
from module.models.efficientnet import EfficientNetWithBBox  # Replace with your actual model class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model =  EfficientNetWithBBox(model_name=model_name, num_classes=1, bbox_output_size=4, pretrained=True)  # Replace with your model initialization
model = model.to(device)

# Load the saved state dictionary
checkpoint_path = "Trained_models/model_20241115_0814.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Set the model to evaluation mode (important for inference)
model.eval()

#Load test data (batch size 8 to save memory)
input_size = get_input_size(model_name)
test_loader = Dataloader(train="test", dim=[input_size, input_size], batch_size=8)


#Initialize
train_iou = []
train_acc = []

for X_val, Y_val, bbox_val, gt_bbox_val, tvals_val in test_loader:
    # Move data to device
    X_batch = X_val.to(device)
    Y_batch = Y_val.to(device).float()  # Ensure target is float for BCE loss
    gt_bbox = gt_bbox_val.to(device)
    t_batch = tvals_val.to(device)
    bbox = bbox_val.to(device)
    
    # Forward pass
    class_score, t_vals = model(X_batch)

    #Compute classifier accuracy:
    predicted_labels = (class_score > 0).float()
    correct_predictions = ((predicted_labels[:,0] == Y_batch).sum().item() ) / Y_batch.size(0)
    train_acc.append(correct_predictions)

    #Compute bbox accuracy:
    for i in range(Y_batch.size(0)):
        if Y_batch[i].cpu().numpy() == 1:
            alt_box = alter_box(bbox[i].cpu().numpy(), t_vals[i].detach().cpu().numpy())
            iou = calculate_iou(alt_box,gt_bbox[i].cpu().numpy())
            train_iou.append(iou)

    #_______ Add AP calculations here:____________________

    #________________________________________________

# Computes average accuracy and iuo across entire test set
avg_train_acc = sum(train_acc)/len(train_acc) 
avg_train_iou = sum(train_iou)/len(train_iou)


print(f"Average test Accuracy: {avg_train_acc * 100:.2f}%")
print(f"Average test Iou     : {avg_train_iou :.2f}")