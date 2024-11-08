import timm
import torch
import torch.nn as nn

class EfficientNetWithBBox(nn.Module):
    def __init__(self, model_name='efficientnet_b5', num_classes=1, bbox_output_size=4, pretrained=True):
        super(EfficientNetWithBBox, self).__init__()
        
        # Load the model directly within the class
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # Modify the classifier (if necessary) for binary classification
        if hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        
        # Unfreeze the final layer (classifier) to allow fine-tuning
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        # Optionally, unfreeze some layers in `features` for fine-tuning
        for param in self.model.features.parameters():
            param.requires_grad = True

        # Add a bounding box regression head
        self.bbox_regressor = nn.Linear(self.model.classifier.in_features, bbox_output_size)

    def forward(self, x):
        # Get classification score
        class_score = self.model(x)
        
        # Get bounding box coordinates using global average pooling on feature maps
        bbox = self.bbox_regressor(self.model.features(x).mean([2, 3]))

        return class_score, bbox


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


# Instantiate the model
# model = EfficientNetWithBBox(model_name='efficientnet_b0', num_classes=1, bbox_output_size=4)

# Example usage of the MultiTaskLoss
# loss_fn = MultiTaskLoss(classification_weight=1.0, bbox_weight=1.0)
