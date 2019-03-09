from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from train import train_model
from dataset import MatchClips, heatmap_loader
from model import generate_model
from argments import parse_opts

# TimeZone
JST = timezone(timedelta(hours=+9), 'JST')
now = datetime.now(JST).strftime('%Y-%m-%d-%H') 

def main(*kargs):
    # Model Initializaion
    model, params_to_update = generate_model(args)
    # Parallel GPUs Process
    model = nn.DataParallel(model, device_ids=list(args.gpus))
    # GPUs Device settings
    torch.cuda.set_device(args.gpus[0])
    model.cuda()
    
    # Transforms for Global Model and Pose Local Model
    if args.model == "3DResNeXt":
        sample_size = 112
    else: #if args.model == "P3D":
        sample_size = 160

    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
    ])
    pose_transform = transforms.Compose([transforms.ToTensor()])
    
    print('Reading Dataset...')
    # Create Dataset
    frame_len = 64
    if args.data == 'rgb':
        POINT_DIR = Path('../Datasets/Points')
        train_video = MatchClips(POINT_DIR, match_list=args.train_list, sample_duration=frame_len, spatial_transform=rgb_transform)
        valid_video = MatchClips(POINT_DIR, match_list=args.valid_list, sample_duration=frame_len, spatial_transform=rgb_transform)
    elif args.data == 'pose':
        POINT_DIR = Path('../Datasets/Points_Heatmap')
        train_video = MatchClips(POINT_DIR, match_list=args.train_list, sample_duration=frame_len, spatial_transform=pose_transform, get_loader=heatmap_loader)
        valid_video = MatchClips(POINT_DIR, match_list=args.valid_list, sample_duration=frame_len, spatial_transform=pose_transform, get_loader=heatmap_loader)

    
    print('Creating DataLoader...')
    # Create DataLoader
    train_loader = DataLoader(train_video, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)
    valid_loader = DataLoader(valid_video, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)
    dataloaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    # Setting optimizer, loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # Optimizer
    optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=args.weight_decay)
    # Decay LR by a factor of 0.1 every 5 epochs
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    epochs = args.epochs
    
    print('Training Start...')
    model_ft, hist, last_val = train_model(model, dataloaders, criterion, optimizer,
                                           scheduler, epochs, frame_len, now,
                                           use_tbx=args.use_tbx, model_name=args.model)
    
    # Saving Model
    SAVE_DIR = Path('./experiments')
    SAVE_PATH = SAVE_DIR.joinpath(args.model, now)
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    SAVE_NAME = SAVE_PATH.joinpath(
        'date-{0}-Epoch{1}-Acc{2:.2f}-Loss{3:.2f}-Data-{4}.pth'.format(now,
                                                    epochs, 
                                                    last_val['acc']*10, 
                                                    last_val['loss'],
                                                    args.data))
    torch.save(model_ft.state_dict(), str(SAVE_NAME))
    
    torch.cuda.empty_cache()
    print("Model Saved!")


if __name__ == '__main__':
    args = parse_opts()
    main(args)
