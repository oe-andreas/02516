import torch
from .losses.losses import dice_loss, iou_loss, accuracy, sensitivity, specificity, bce_weighted, bce_loss, focal_loss

def train(model, device, opt, scheduler, loss_fn, epochs, train_loader, test_loader):
    
    eval_metrics = [dice_loss, iou_loss, accuracy, sensitivity, specificity, bce_weighted, bce_loss, focal_loss]
    
    observed_eval_metrics_train = []
    observed_eval_metrics_test = []
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        # Training
        avg_loss = 0
        avg_eval_metrics = torch.zeros(len(eval_metrics))
        
        for X_batch, Y_batch, Z_batch in train_loader:
            model.train()  # train mode
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
 
            # set parameter gradients to zero
            opt.zero_grad()
            
            # forward
            Y_pred = model(X_batch)

            # Extract only the pixels where Z_batch == 1
            Y_batch_masked = Y_batch[Z_batch > 0.5]
            Y_pred_masked = Y_pred[Z_batch > 0.5]

            # Calculate loss only on masked pixels
            loss = loss_fn(Y_batch_masked, Y_pred_masked)
            loss.backward()  # backward-pass
            opt.step()  # update weights
            
            # calculate metrics to show the user
            avg_loss += loss.detach().cpu() / len(train_loader)
            
            model.eval()
            with torch.no_grad():
                for i, eval_metric in enumerate(eval_metrics):
                    avg_eval_metrics[i] += eval_metric(Y_batch_masked, Y_pred_masked).cpu() / len(test_loader)
            
                
        observed_eval_metrics_train.append(avg_eval_metrics)
            
        print(' - loss: %f' % avg_loss)
        train_losses.append(avg_loss)

        # Testing
        avg_loss = 0
        avg_eval_metrics = torch.zeros(len(eval_metrics))
        model.eval()
        for X_batch, Y_batch, Z_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            with torch.no_grad():
                Y_pred = model(X_batch)
                # Extract only the pixels where Z_batch == 1
                Y_batch_masked = Y_batch[Z_batch > 0.5]
                Y_pred_masked = Y_pred[Z_batch > 0.5]
                loss = loss_fn(Y_batch_masked, Y_pred_masked)

            avg_loss += loss.detach().cpu() / len(test_loader)
            
            
            for i, eval_metric in enumerate(eval_metrics):
                avg_eval_metrics[i] += eval_metric(Y_batch_masked, Y_pred_masked).cpu() / len(test_loader)
                
        observed_eval_metrics_test.append(avg_eval_metrics)
        print(' - val_loss: %f' % avg_loss)
        test_losses.append(avg_loss)

        # Step the scheduler with the validation loss
        scheduler.step(avg_loss)
        
    
    return train_losses, test_losses, observed_eval_metrics_train, observed_eval_metrics_test
