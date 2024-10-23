import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses, model_name):
    plt.figure(figsize=(12, 6))
    
    # Plot training and testing losses with improved styling
    plt.plot(train_losses, label='Training Loss', color='blue', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.plot(test_losses, label='Testing Loss', color='red', marker='s', linestyle='--', linewidth=2, markersize=5)

    # Titles and labels
    plt.title(f'Loss History for {model_name}', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Better legend with frame
    plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)

    # Save the figure with high resolution
    plt.savefig(f'graphics/loss_history_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics(observed_eval_metrics, model_name):
    observed_eval_metrics = np.array(observed_eval_metrics)

    plt.figure(figsize=(12, 6))
    
    # Plot each metric with improved styling
    plt.plot(observed_eval_metrics[:, 0], label='Dice', color='blue', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.plot(observed_eval_metrics[:, 1], label='IoU', color='orange', marker='s', linestyle='--', linewidth=2, markersize=5)
    plt.plot(observed_eval_metrics[:, 2], label='Accuracy', color='green', marker='^', linestyle=':', linewidth=2, markersize=5)
    plt.plot(observed_eval_metrics[:, 3], label='Sensitivity', color='red', marker='D', linestyle='-', linewidth=2, markersize=5)
    plt.plot(observed_eval_metrics[:, 4], label='Specificity', color='purple', marker='x', linestyle='--', linewidth=2, markersize=5)

    # Titles and labels
    plt.title(f'Metric Scores Over Time for {model_name}', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Better legend with frame
    plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)

    # Save the figure with high resolution
    plt.savefig(f'graphics/score_history_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(model, device, train_loader, model_name):
    # visualize predictions
    to_im_shape = lambda x : x.permute(1,2,0).numpy()

    model.eval()
    X_batch, y_batch = next(iter(train_loader))
    X, y = X_batch[0], y_batch[0]
    X = X.unsqueeze(0).to(device)
    y_pred = model(X)
    y_pred = torch.sigmoid(y_pred).detach().cpu().numpy().squeeze()

    fig, axs = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})

    #Input image
    axs[0].imshow(to_im_shape(X.cpu().squeeze()))

    # Ground truth
    axs[1].imshow(to_im_shape(y), vmin=0, vmax=1)
    axs[1].set_title('Ground Truth')

    # Prediction
    im = axs[2].imshow(y_pred, vmin=0, vmax=1)
    axs[2].set_title('Prediction')

    # Colorbar
    fig.colorbar(im, cax=axs[3])

    plt.tight_layout()

    plt.savefig(f'graphics/predictions_{model_name}.png')


