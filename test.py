import os
import argparse
import wandb
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import data
from trainer.base import image_frame_to_grid
from util.distributed import init_distributed, is_main_process
from util.util import set_random_seed, save_videos_as_files
from config.config import Config
from models.diffusion import make_beta_schedule, create_gaussian_diffusion
import trainer
def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    # Config path
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='training config dir')
    # experiment specifics
    parser.add_argument('--exp_name', type=str, default='diffusion', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--model', type=str, default='diffusion', help='name of the model. [diffusion, VAE]')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--save_root', type=str, default='./results', help='models are saved here')
    parser.add_argument('--save_name', type=str, help='name of the experiment. It decides where to store samples and models')

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
    save_root = os.path.join(args.save_root, args.save_name)
    if is_main_process() :
        os.makedirs(save_root, exist_ok=True)

    opt = Config(args.config, args, is_train=True, verbose=False)
    opt.continue_train = not args.no_resume
    opt.save_root = os.path.join('./checkpoints', args.exp_name)
    opt.model.param.maxframe = opt.data.maxframe
    opt.model.param.sub_frame = opt.data.sub_frame
    opt.diffusion.sample_algorithm = args.sample_algorithm

    if not args.single_gpu:
        opt.local_rank = local_rank
        opt.device = local_rank

    val_dataset, train_dataset = data.get_train_val_dataloader(opt.data)

    # UNet model settings
    model, model_ema, optimizer, scheduler = trainer.get_model_optimizer_and_scheduler(opt)
    # Diffusion settings
    beta_schedule = make_beta_schedule(**opt.diffusion.beta_schedule)
    diffusion = create_gaussian_diffusion(beta_schedule, predict_xstart=False)

    trainer = trainer.get_trainer(opt,
                                  diffusion, model,
                                  model_ema,
                                  optimizer,
                                  scheduler,
                                  train_dataset, val_dataset,
                                  wandb)
    trainer.load_checkpoint(opt, args.which_iter)

    score_dict = {'filename': [],
                  'score': [],
                  'label': []}

    for i, data_i in enumerate(tqdm(val_dataset)):
        samples, filenames, labels, scores = trainer.test(data_i)
        save_videos_as_files(samples, save_root, filenames, labels)

        score_dict['filename'].extend(filenames)
        score_dict['score'].extend(scores)
        score_dict['label'].extend(labels.tolist())

    score_df = pd.DataFrame.from_dict(score_dict)
    score_df.to_csv(os.path.join(save_root, 'scores.csv'), index=False)
    print('Test was successfully finished.')