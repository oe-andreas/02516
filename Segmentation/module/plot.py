import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_losses, test_losses, model_name):
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(test_losses, label='test')
    plt.legend()
    plt.savefig(f'graphics/loss_history_{model_name}.png')

def plot_metrics(observed_eval_metrics, model_name):
    observed_eval_metrics = np.array(observed_eval_metrics)
    plt.figure()
    plt.plot(observed_eval_metrics[:,0], label='dice')
    plt.plot(observed_eval_metrics[:,1], label='iou')
    plt.plot(observed_eval_metrics[:,2], label='accuracy')
    plt.plot(observed_eval_metrics[:,3], label='sensitivity')
    plt.plot(observed_eval_metrics[:,4], label='specificity')
    plt.legend()
    plt.savefig(f'graphics/score_history_{model_name}.png')

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


