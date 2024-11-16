from plots import plot_steps
import torch
import warnings

# Suppress FutureWarning from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'Trained_models/model_20241113_1537.pth'
model_name = 'efficientnet_b0'
plot_steps(5,[5,6,7], model_path, model_name, device, num_class_0=5, discard_threshold=0.1, consideration_threshold=None)