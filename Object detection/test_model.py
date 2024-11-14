from module.dataloaders.loader import Dataloader
from utils import get_input_size

model_name = 'efficientnet_b0'

import torch
from module.models.efficientnet import EfficientNetWithBBox  # Replace with your actual model class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize your model
model =  EfficientNetWithBBox(model_name=model_name, num_classes=1, bbox_output_size=4, pretrained=True)  # Replace with your model initialization
model = model.to(device)
# Load the saved state dictionary
checkpoint_path = "Trained_models/model_20241114_1358.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# Set the model to evaluation mode (important for inference)
model.eval()


input_size = get_input_size(model_name)
test_loader = Dataloader(train="test", dim=[input_size, input_size], batch_size=64)


print("hello")
# Initialize accuracy tracking variables
total_samples = 0
correct_predictions = 0

for X_val, Y_val, bbox_val, gt_bbox_val, tvals_val in test_loader:
    # Move data to device
    X_val = X_val.to(device)
    Y_val = Y_val.to(device).float()
    gt_bbox_val = gt_bbox_val.to(device)
    t_batch_val = tvals_val.to(device)
    
    # Forward pass
    class_score_val, t_vals_val = model(X_val)
    
    # Compute predictions (apply threshold of 0.5)
    predicted_labels = (class_score_val > 0).float()  # Thresholding at 0 for logits
    
    # Compare with ground truth
    correct_predictions += (predicted_labels == Y_val).sum().item()
    print(correct_predictions)
    total_samples += Y_val.size(0)

# Calculate overall accuracy
accuracy = correct_predictions / total_samples
print(f"Accuracy: {accuracy * 100:.2f}%")