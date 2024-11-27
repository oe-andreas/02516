from plots import plot_steps
import torch
import warnings
import json

# Suppress FutureWarning from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'Trained_models/b0_model_20241117_0044.pth'
model_name = 'efficientnet_b0'
#639 is the one with most positive proposals before NMS
plot_steps(612,[1,2,3,4,5,6,7], model_path, model_name, device, num_class_0=5, discard_threshold=0.1, consideration_threshold=None)
