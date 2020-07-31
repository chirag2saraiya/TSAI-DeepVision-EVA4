import time
import copy
from tqdm import tqdm

def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          scheduler,
          max_epochs_stop,
          n_epochs,
          ):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs


    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """
    dataset_sizes = {'train':len(train_loader.dataset),'val':len(valid_loader.dataset)}
    since = time.time()
    train_loss_log = []
    test_loss_log = []
    train_acc_log = []
    test_acc_log = []
    history = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                current_dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                current_dataloader = valid_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for (inputs, labels) in tqdm(current_dataloader):
            #for inputs, labels in current_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            if phase == "train":
              epoch_loss = running_loss / len(train_loader.dataset)
              epoch_acc = running_corrects.double() / len(train_loader.dataset)
            else:
              epoch_loss = running_loss / len(valid_loader.dataset)
              epoch_acc = running_corrects.double() / len(valid_loader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
              train_loss_log.append(epoch_loss)
              train_acc_log.append(epoch_acc)
            else:
              test_loss_log.append(epoch_loss)
              test_acc_log.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_file_name)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    history = pd.DataFrame({  'train_loss' : train_loss_log,
                          'train_acc' : train_acc_log,
                          'valid_loss' : test_acc_log,
                          'valid_acc' : test_acc_log
                        })


    print('Best val Acc: {:4f}'.format(best_acc))
    return model, history
