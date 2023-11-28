import os
import argparse
import wandb
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import data
from util.distributed import init_distributed, is_main_process
from util.util import set_random_seed, save_videos_as_files, save_videos_to_images
from config.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    # Config path
    parser.add_argument('--config', type=str, default='./config/config_classifier.yaml', help='training config dir')
    # experiment specifics
    parser.add_argument('--exp_name', type=str, default='diffusion', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--model', type=str, default='diffusion', help='name of the model. [diffusion, VAE]')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--save_root', type=str, default='./results', help='models are saved here')
    parser.add_argument('--save_name', type=str, help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--test_fold', type=int, default=1, help='models are saved here')

    # etc.
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    # for DDP
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    parser.add_argument('--sample_algorithm', type=str, default='ddim')


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    local_rank = init_distributed()

    args = parse_args()
    set_random_seed(args.seed)

    # Save path setting


    opt = Config(args.config, args, is_train=True, verbose=False)
    opt.continue_train = not args.no_resume
    opt.data.test_fold = args.test_fold
    opt.save_root = './eval_results'
    opt.data.exp_name = args.exp_name

    if is_main_process() :
        os.makedirs(opt.save_root, exist_ok=True)

    if not args.single_gpu:
        opt.local_rank = local_rank
        opt.device = local_rank

    val_dataset, train_dataset = data.get_train_val_dataloader(opt.data)

    score_dict = {'filename': []}
    for i in range(opt.data.maxframe) :
        score_dict[f'frame_{i+1}'] = []
    score_dict['total_score'] = []
    score_dict['diff_score'] = []
    score_dict['label'] = []

    for i, data_i in enumerate(tqdm(val_dataset)):
        diff, seg, filename, label = data_i['diff'], data_i['seg'], data_i['filename'], data_i['label']
        diff = (diff * 0.5 + 0.5).squeeze()
        seg = (seg * 0.5 + 0.5).squeeze()

        for batch in range(seg.size(0)) :
            score_dict['filename'].append(filename[batch])
            score_dict['label'].append(label[batch].item())
            score_dict['diff_score'].append(diff[batch].sum().item())
            score_dict['total_score'].append(seg[batch].sum().item())
            for frame in range(opt.data.maxframe) :
                one_frame = seg[batch][frame].sum().item()
                score_dict[f'frame_{frame+1}'].append(one_frame)

    df = pd.DataFrame.from_dict(score_dict)
    df.to_csv(os.path.join(opt.save_root, args.exp_name + '.csv'), index=False)

