from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset import MatchClips, heatmap_loader
from model import generate_model
from argments import parse_opts

# TimeZone
JST = timezone(timedelta(hours=+9), 'JST')
now = datetime.now(JST).strftime('%Y-%m-%d-%H')


def get_optimizer(trial, param_to_update):
    #optimizer_names = ['Adam']
    #optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    #if optimizer_name == optimizer_names[0]:
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    optimizer = optim.Adam(param_to_update, lr=adam_lr, weight_decay=weight_decay)
    #else:
        # momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        # optimizer = optim.SGD(param_to_update, lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)
    return optimizer


def train(model, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 1 - correct / (len(test_loader.dataset) * 64)


def objective(trial):
    # Load options
    args = parse_opts()
    
    # Model
    model, param_to_update = generate_model(args)
    # Parallel GPUs Process
    model = nn.DataParallel(model, device_ids=list(args.gpus))
    # GPUs
    torch.cuda.set_device(args.gpus[0])
    model.cuda()

    # Transforms
    if args.model == '3DResNeXt':
        sample_size = 112
    else:
        sample_size = 160

    # Transforms Compose
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
    ])
    pose_transform = transforms.Compose([transforms.ToTensor()])
    
    # Load Dataset
    frame_len = 64
    if args.data == 'rgb':
        POINT_DIR = Path('../Datasets/Points')
        train_video = MatchClips(POINT_DIR, match_list=args.train_list, sample_duration=frame_len, spatial_transform=rgb_transform)
        valid_video = MatchClips(POINT_DIR, match_list=args.valid_list, sample_duration=frame_len, spatial_transform=rgb_transform)
    elif args.data == 'pose':
        POINT_DIR = Path('../Datasets/Points_Heatmap')
        train_video = MatchClips(POINT_DIR, match_list=args.train_list, sample_duration=frame_len, spatial_transform=pose_transform, get_loader=heatmap_loader)
        valid_video = MatchClips(POINT_DIR, match_list=args.valid_list, sample_duration=frame_len, spatial_transform=pose_transform, get_loader=heatmap_loader)

    # DataLoader
    train_loader = DataLoader(train_video, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)
    valid_loader = DataLoader(valid_video, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)

    # Loss Function
    criterion = nn.CrossEntropyLoss().cuda()
    # Optimzer
    optimizer = get_optimizer(trial, param_to_update)

    for step in tqdm(range(args.epochs)):
        # print('Epoch {}/{}'.format(step, args.epochs-1))
        # print('-' * 20)
        train(model, train_loader, optimizer, criterion)
        error_rate = test(model, valid_loader)
        print("Error Rate:", error_rate)
        # Report error_rate(intermediate objective value)
        trial.report(error_rate, step)
        # Handle pruning based on the intermediate value(error_rate).
        if trial.should_prune(step):
            raise optuna.structs.TrialPruned()

    return error_rate

if __name__ == '__main__':
    import optuna
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    
    print('  User attrs:')
    for key, value in trial.user_attrs.items():
        print('    {}: {}'.format(key, value))
