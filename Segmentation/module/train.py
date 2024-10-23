import torch
from .losses.losses import dice_loss, iou_loss, accuracy, sensitivity, specificity

def train(model, device, opt, loss_fn, epochs, train_loader, test_loader):
    
    eval_metrics = [dice_loss, iou_loss, accuracy, sensitivity, specificity]
    
    observed_eval_metrics = []
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        # Training
        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_loss += loss.detach().cpu() / len(train_loader)
        print(' - loss: %f' % avg_loss)
        train_losses.append(avg_loss)

        # Testing
        avg_loss = 0
        avg_eval_metrics = []
        model.eval()
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            with torch.no_grad():
                Y_pred = model(X_batch)
                loss = loss_fn(Y_batch, Y_pred)

            avg_loss += loss.detach().cpu() / len(test_loader)
            
    
            for eval_metric in eval_metrics:
                avg_eval_metrics.append(eval_metric(Y_batch, Y_pred).cpu() / len(test_loader))
        
        observed_eval_metrics.append(avg_eval_metrics)
        print(' - val_loss: %f' % avg_loss)
        test_losses.append(avg_loss)
    
    return train_losses, test_losses, observed_eval_metrics