import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses, dataset_name, model_name):
    plt.figure(figsize=(12, 6))
    
    # Plot training and testing losses with improved styling
    plt.plot(train_losses, label='Training Loss', color='blue', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.plot(test_losses, label='Testing Loss', color='red', marker='o', linestyle='-', linewidth=2, markersize=5)

    # Titles and labels
    plt.title(f'Loss History for {model_name} on {dataset_name} Dataset', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Better legend with frame
    plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)

    # Save the figure with high resolution
    plt.savefig(f'graphics/loss_history_{model_name}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics(observed_eval_metrics, dataset_name, model_name):
    observed_eval_metrics = np.array(observed_eval_metrics)

    plt.figure(figsize=(12, 6))
    
    # Plot each metric with improved styling
    plt.plot(observed_eval_metrics[:, 0], label='Dice', color='blue', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.plot(observed_eval_metrics[:, 1], label='IoU', color='orange', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.plot(observed_eval_metrics[:, 2], label='Accuracy', color='green', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.plot(observed_eval_metrics[:, 3], label='Sensitivity', color='red', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.plot(observed_eval_metrics[:, 4], label='Specificity', color='purple', marker='o', linestyle='-', linewidth=2, markersize=5)

    # Titles and labels
    plt.title(f'Metric Scores Over Time for {model_name} on {dataset_name} Dataset', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.6)

    # Better legend with frame
    plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)

    # Save the figure with high resolution
    plt.savefig(f'graphics/score_history_{model_name}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(model, device, train_loader, dataset_name, model_name):
    # visualize predictions
    to_im_shape = lambda x : x.permute(1,2,0).numpy()

    model.eval()
    X_batch, y_batch, _ = next(iter(train_loader))
    X, y = X_batch[0], y_batch[0]
    X = X.unsqueeze(0).to(device)
    y_pred = model(X)
    y_pred = torch.sigmoid(y_pred).detach().cpu().numpy().squeeze()

    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5), gridspec_kw={'width_ratios': [1, 1, 1, 0.05]})

    # Input image processing: normalize per color channel
    original_image = X.cpu().squeeze()
    for c in range(original_image.shape[0]):  # Loop over color channels
        channel = original_image[c, :, :]
        channel_min, channel_max = channel.min(), channel.max()
        original_image[c, :, :] = (channel - channel_min) / (channel_max - channel_min)
        
    axs[0].imshow(to_im_shape(original_image))
    axs[0].set_title('Normalized input Image')

    # Ground truth
    axs[1].imshow(to_im_shape(y), vmin=0, vmax=1)
    axs[1].set_title('Ground Truth')

    # Prediction
    im = axs[2].imshow(y_pred, vmin=0, vmax=1)
    axs[2].set_title('Prediction')
    
    for ax in axs[:-1]:
        ax.axis('off')


    # Colorbar
    fig.colorbar(im, cax=axs[3])

    plt.tight_layout()

    plt.savefig(f'graphics/predictions_{model_name}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions_weak(model, device, train_loader, NoA, model_name):
    # NoA = Number of Annotatinos
    
    # visualize predictions
    to_im_shape = lambda x : x.permute(1,2,0).numpy()

    model.eval()
    X_batch,Y_real, y_batch = next(iter(train_loader))
    X, y, y_real = X_batch[0], y_batch[0], Y_real[0]
    X = X.unsqueeze(0).to(device)
    y_pred = model(X)
    y_pred = torch.sigmoid(y_pred).detach().cpu().numpy().squeeze()
    
    #print("noa", NoA)
    # Convert sampled pixel coordinates into separate x and y arrays for plotting
    zero_y, zero_x = y[:NoA, 0], y[:NoA, 1]
    one_y, one_x = y[NoA:, 0], y[NoA:, 1]

    fig, axs = plt.subplots(1, 5, figsize=(18, 5), gridspec_kw={'width_ratios': [1,1, 1, 1,0.05]})

    #Input image
    axs[0].imshow(to_im_shape(X.cpu().squeeze()))
    axs[0].set_title('input')

    # Ground truth
    # Overlay the sampled 0-value pixels (using blue markers 'o')
    axs[1].imshow(to_im_shape(X.cpu().squeeze()))
    axs[1].scatter(zero_x, zero_y, color='red', marker='o', s=50)
    # Overlay the sampled 1-value pixels (using red markers 'x')
    axs[1].scatter(one_x, one_y, color='green', marker='o', s=50)
    axs[1].set_title('Ground truth')

    # Ground truth full
    im = axs[2].imshow(to_im_shape(y_real), vmin=0, vmax=1)
    axs[2].set_title('Full Ground Truth')

    # Prediction
    im = axs[3].imshow(y_pred, vmin=0, vmax=1)
    axs[3].set_title('Prediction')

    # Colorbar
    fig.colorbar(im, cax=axs[4])

    plt.tight_layout()

    plt.savefig(f'graphics/predictions_{model_name}.png')
    



def plot_all_metrics(observed_eval_metrics_array, dataset_name=None, loss_labels=None, model_labels=None, split_labels=None, metric_labels=None, include_legend = True):
    # Extract dimensions
    num_losses, num_models, num_splits, num_metrics = observed_eval_metrics_array.shape
    
    num_not_losses = num_metrics - num_losses
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_splits, num_metrics, figsize=(num_metrics * 1.7, num_splits * 1.7))
    fig.suptitle(dataset_name if dataset_name else "")
    
    # Define labels
    loss_labels = [f'Loss {i+1}' for i in range(num_losses)] if loss_labels is None else loss_labels
    model_labels = [f'Model {i+1}' for i in range(num_models)] if model_labels is None else model_labels
    split_labels = [f'Split {i+1}' for i in range(num_splits)] if split_labels is None else split_labels
    metric_labels = [f'Metric {i+1}' for i in range(num_metrics)] if metric_labels is None else metric_labels
    
    # Colors for different models and losses
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Iterate through the array and plot the metrics
    handles, labels = [], []
    y_limits = [0, 0]  # To store the max y-limits for the top and bottom rows of the last columns
    
    for k in range(num_splits):
        for m in range(num_metrics):
            ax = axes[k, m]
            width = 0.2
            
            for j in range(num_models):
                for i in range(num_losses):
                    metric = observed_eval_metrics_array[i, j, k, m]
                    
                    label = f'{loss_labels[i]}' if j == num_models - 1 else None #only label once
                    
                    color = colors[i]
                    
                    is_loss_in_question = (m == i)
                    
                    alpha = 0.4 if (m < num_losses and not is_loss_in_question) else 0.9
                    placement = width*(num_losses + 0.5)*j + i * width + width/4
                    
                    # Highlight specific bars
                    edgecolor = 'black' if is_loss_in_question else 'none'
                    linewidth = 2 if is_loss_in_question else 0
                    
                    bar = ax.bar(placement, metric, width, color=color, alpha=alpha, label=label, edgecolor=edgecolor, linewidth=linewidth)
                    
                    if label and label not in labels and m == num_metrics - 1:
                        handles.append(bar)
                        labels.append(label)
            
            # Add a black line between Model 1 and Model 2
            if num_models > 1:
                separation_position = width * (num_losses + 0.5)
                ax.axvline(separation_position - width / 2, color='black', linewidth=.5, linestyle='--')
            
            if k == num_splits - 1: #in bottom row, put model labels on x axis
                # Model labels
                ax.set_xticks([width*(num_losses + 0.5)*j + width*(num_losses - 1)/2 for j in range(num_models)])
                ax.set_xticklabels(model_labels)
            else: #in any other row, remove x axis
                ax.set_xticks([])
                
            if k == 0: #in top row, put metric labels in title
                ax.set_title(metric_labels[m])
                
            if m == 0: #in first column or third-to-last column, write train/test labels
                ax.set_ylabel(split_labels[k])
            elif m > num_losses: #in all other columns, hide y-axis
                ax.yaxis.set_visible(False) 
            
            # Show only the bottom and left spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(True if m <= num_losses else False)
            ax.spines['bottom'].set_visible(True)
            
            # Set specific y-axis ticks
            if m >= num_losses:
                ax.set_yticks([0, 0.5, 1])
            else:
                pass
            
            
    
    # Set the same y-limits for the top and bottom rows of the last columns
    for m in range(num_losses):
        
        ylims_this_column = [axes[k, m].get_ylim() for k in range(num_splits)]
        smallest_y = min([y[0] for y in ylims_this_column])
        biggest_y = max([y[1] for y in ylims_this_column])
        
        
        for k in range(num_splits):
            axes[k, m].set_ylim(smallest_y, biggest_y)
    
    # Add legend underneath the plot
    if include_legend:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=num_losses, frameon=False)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    fig.canvas.draw()
    
    
    fig.savefig(f'graphics/all_metrics_{dataset_name}.png')
