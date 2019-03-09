import copy
import time
from tqdm import tqdm

import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

def train_model(model, dataloaders, criterion, optimizer, scheduler,
                epochs:int, sample_duration, date, use_tbx, model_name):
    since = time.time()
    
    if use_tbx:
        writer = SummaryWriter('runs/{}/{}'.format(model_name, date))

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # cudnn setting
    torch.backends.cudnn.benchmark=True
    try:
        for epoch in tqdm(range(epochs)):
            print('Epoch {}/{}'.format(epoch, epochs-1))
            print('-' * 10)
            epoch_since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iteration
                print('Iteration start on', phase)
                for inputs, labels in dataloaders[phase]:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())

                    # Zero the parameter gradient
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs) # Output: [BatchSize, Classes, TimeLength]
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, dim=1)

                    if phase == 'train':
                        # back propagation
                        loss.backward()
                        optimizer.step()

                    # statics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / (len(dataloaders[phase].dataset) * sample_duration)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                # Scheduler
                if phase == 'valid':
                    scheduler.step(epoch_loss)
                # Logs for tensorboardX
                if use_tbx:
                    if phase == 'train':
                        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                        writer.add_scalar('Loss/train', epoch_loss, epoch)
                        # For Summary
                        train_acc = epoch_acc
                        train_loss = epoch_loss
                    else:
                        writer.add_scalar('Accuracy/valid', epoch_acc, epoch)
                        writer.add_scalar('Loss/valid', epoch_loss, epoch)
                        # For Summary
                        valid_acc = epoch_acc
                        valid_loss = epoch_loss

                # Time Information
                epoch_elapsed = time.time() - epoch_since
                epoch_since = time.time()
                print('{} Epoch {} in {:.0f} m {:.0f}s'.format(phase, 
                                                               epoch, 
                                                               epoch_elapsed // 60, 
                                                               epoch_elapsed % 60))


                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'valid':
                    val_acc_history.append(epoch_acc)

            print()
            # Logs for Summary
            if use_tbx:
                writer.add_scalars('Accuracy/Both', {
                    'train': train_acc,
                    'valid': valid_acc,
                }, epoch)
                writer.add_scalars('Loss/Both', {
                    'train': train_loss,
                    'valid': valid_loss,
                }, epoch)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early.')
    
    
    # Time Info
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Valdation Accuracy: {:.4f}'.format(best_acc))
    
    # Last Acc & Loss
    last_val = {'acc': best_acc, 'loss': epoch_loss}
    # Close summarywriter
    if use_tbx:
        writer.close()
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, last_val
