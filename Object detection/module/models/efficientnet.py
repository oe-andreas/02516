import timm
import torch.nn as nn

# Define model name and number of output classes
model_name = 'efficientnet_b0'
num_classes = 1  # For binary classification
bbox_output_size = 4  # For bounding box (x_min, y_min, x_max, y_max)

# Load pretrained EfficientNet model
model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

# Modify the classifier (already done by setting num_classes=1)
if hasattr(model, 'classifier'):
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Unfreeze the final layer (classifier) to allow fine-tuning
for param in model.classifier.parameters():
    param.requires_grad = True

# To fine-tune some additional layers unfreeze them too:
for param in model.features.parameters():  
    param.requires_grad = True

# Add a bounding box regression head
class EfficientNetWithBBox(nn.Module):
    def __init__(self, model, bbox_output_size):
        super(EfficientNetWithBBox, self).__init__()
        self.model = model
        
        # Add a regression head for bounding box prediction
        self.bbox_regressor = nn.Linear(model.classifier.in_features, bbox_output_size)

    def forward(self, x):
        # Get classification score
        class_score = self.model(x)
        
        # Get bounding box coordinates
        bbox = self.bbox_regressor(self.model.features(x).mean([2, 3]))  # Global average pooling
        
        return class_score, bbox

# Wrap the model
model = EfficientNetWithBBox(model, bbox_output_size)

# Define the multi-task loss function
class MultiTaskLoss(nn.Module):
    def __init__(self, classification_weight=1.0, bbox_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.classification_loss = nn.BCEWithLogitsLoss()  # Binary classification loss
        self.bbox_loss = nn.SmoothL1Loss()  # Bounding box regression loss
        self.classification_weight = classification_weight
        self.bbox_weight = bbox_weight

    def forward(self, class_pred, class_true, bbox_pred, bbox_true):
        # Classification loss
        class_loss = self.classification_loss(class_pred, class_true)
        
        # Bounding box loss
        bbox_loss = self.bbox_loss(bbox_pred, bbox_true)
        
        # Combine losses
        total_loss = self.classification_weight * class_loss + self.bbox_weight * bbox_loss
        return total_loss


