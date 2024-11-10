import timm
import torch.nn as nn

class EfficientNetWithBBox(nn.Module):
    def __init__(self, model_name='efficientnet_b5', num_classes=1, bbox_output_size=4, pretrained=True):
        super(EfficientNetWithBBox, self).__init__()
        
        # Load the model using timm and set num_classes for classification output
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # Modify the classifier if it exists, for binary classification
        if hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        
        # Unfreeze final classifier layer for fine-tuning
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # Add a bounding box regression head
        self.bbox_regressor = nn.Linear(self.model.classifier.in_features, bbox_output_size)

    def forward(self, x):
        # Extract features using forward_features
        features = self.model.forward_features(x)
        
        # Global average pooling
        features = self.model.global_pool(features)
        
        # Get classification score
        class_score = self.model.classifier(features)
        
        # Extract features using forward_features for bbox regression
        bbox = self.bbox_regressor(features)  # Apply global average pooling

        return class_score, bbox


