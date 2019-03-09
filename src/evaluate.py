import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import generate_model
from argments import parse_opts
from dataset import MatchClips, heatmap_loader

# Time Zone
JST = timezone(timedelta(hours=+9), 'JST')
now = datetime.now(JST).strftime('%Y-%m-%d-%H')

def evaluate(*kargs):
   # GPU Device settings
    torch.cuda.set_device(args.gpus[0])
    # Model Instance
    model, params = generate_model(args)
    # Parallel GPUs Process
    model = nn.DataParallel(model, device_ids=args.gpus)
    # Load Pretrained Model
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval()

    if args.model == '3DResNeXt':
        sample_size = 112
    else:
        sample_size = 160

    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize((sample_size, sample_size)),
        transforms.ToTensor(),
        ])
    pose_transform = transforms.Compose([transforms.ToTensor()])

    # Create Dataset
    frame_len = 64
    if args.data == 'rgb':
        POINT_DIR = Path('../Datasets/Points')
        test_video = MatchClips(POINT_DIR, match_list=args.eval_list, sample_duration=frame_len, spatial_transform=rgb_transform, onEval=True)
    elif args.data == 'pose':
        POINT_DIR = Path('../Datasets/Points_Heatmap')
        test_video = MatchClips(POINT_DIR, match_list=args.eval_list, sample_duration=frame_len, spatial_transform=pose_transform, get_loader=heatmap_loader, onEval=True)

    # Create DataLoader
    test_loader = DataLoader(test_video, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)

    running_corrects = 0
    all_preds = []
    all_scores = []
    all_labels = []
    all_frames = []
    bar_prob = []
    bar_labels = []
    # Evaluate
    for inputs, (labels, frames) in test_loader:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        outputs = model(inputs) # Output: [Batch, Classes, TimeLength]
        outputs = F.softmax(outputs, dim=1) # Extract Probability
        # For Bar Score
        scores_sorted = outputs.cpu().data.numpy().transpose(0,2,1) #to[64,13]
        bar_prob.append(scores_sorted)

        scores, preds = outputs.topk(1, dim=1, largest=True, sorted=True)
        #running_corrects += preds.eq(labels.view(1, -1).expand_as(preds))
        running_corrects += torch.sum(preds == labels.data)
        # Save Predicted Result, Score and Label
        np_pred = preds.cpu().numpy().reshape(-1)
        np_score = scores.cpu().data.numpy().reshape(-1)
        np_label = labels.cpu().numpy().reshape(-1) 
        # Append for evaluate later
        all_preds = np.append(all_preds, np_pred)
        all_scores = np.append(all_scores, np_score)
        all_labels = np.append(all_labels, np_label)
        all_frames = np.append(all_frames, np.array(frames))

    acc = running_corrects.double() / (len(test_loader.dataset) * frame_len)
    print('Accuracy: {:.4f}'.format(acc))
    
    # Bar score
    bar_prob = np.array(bar_prob).reshape(-1,13)
    print('Save bar_score as shape of:', bar_prob.shape)
    # Correct labels, Frames
    pred_dict = {"Preds": all_preds, "Top1_score": all_scores, "Labels": all_labels, "Frames": all_frames}
    # To DataFrame
    score_df = pd.DataFrame(bar_prob)
    pred_df = pd.DataFrame(pred_dict)
    # Change number to str label for demo
    shot_labels = pd.read_csv('../Datasets/shot_labels.txt', header=None)
    f = lambda x: shot_labels.values[int(x)][0]
    pred_df['class_str_label'] = pred_df.Labels.apply(f)
    pred_df['class_str_top1'] = pred_df.Preds.apply(f)
    #pred_df = pred_df.drop(columns=['Preds', 'Labels'])
    # Change Columns name of score_df
    shot_dict = shot_labels.to_dict()[0]
    score_df = score_df.rename(columns=shot_dict)
    # Concat two DataFrames
    all_df = pd.concat([pred_df, score_df], axis=1)
    # Model Path
    model_path = Path(args.model_path)
    model_name = model_path.parts[-1]
    print(model_name)
    # Save Path
    save_path = Path('./results', args.model)
    save_path.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(save_path.joinpath(model_name[:-4] + "-match-{}.csv".format(args.eval_list[0])), index=False)

if __name__ == '__main__':
    args = parse_opts()
    evaluate(args)
