import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--start_epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--layers', type=int, default=0)
    parser.add_argument('--n_worker', type=int, default=0)
    parser.add_argument('--data', choices=['rgb', 'pose'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_list', type=int, nargs='+')
    parser.add_argument('--valid_list', type=int, nargs='+')
    parser.add_argument('--eval_list', type=int, nargs='+')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--model', choices=['P3D', '3DResNeXt','P3D_flatten', 'PoseNet'])
    parser.add_argument('--use_tbx', action='store_true')
    parser.add_argument('--use_crf', action='store_true')
    parser.add_argument('--train_layer', choices=['all', 'layer1', 'layer2', 'TCN'])

    args = parser.parse_args()

    return args
