from tqdm import tqdm
import os
import argparse
import data
from util.distributed import init_distributed, is_main_process
from util.util import set_random_seed
from config.config import Config
from models.diffusion import make_beta_schedule, create_gaussian_diffusion
from util.iter_counter import IterationCounter
import trainer
import wandb
def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    # Config path
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='training config dir')
    # experiment specifics
    parser.add_argument('--exp_name', type=str, default='diffusion', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--model', type=str, default='diffusion', help='name of the model. [diffusion, VAE]')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    # etc.
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--which_iter', type=int, default=None)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    # for DDP
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    local_rank = init_distributed()

    args = parse_args()
    set_random_seed(args.seed)

    # Save path setting
    save_root = os.path.join(args.checkpoints_dir, args.exp_name)
    if is_main_process() :
        os.makedirs(save_root, exist_ok=True)

    opt = Config(args.config, args, is_train=True, verbose=True)
    opt.continue_train = not args.no_resume
    opt.save_root = save_root
    opt.model.param.maxframe = opt.data.maxframe
    opt.model.param.sub_frame = opt.data.sub_frame
    # Debug setting
    if args.debug :
        opt.display_freq= 1
        # opt.print_freq = 1
        # opt.save_latest_freq= 1
        # opt.save_epoch_freq= 1
        opt.data.train.batch_size=2
        opt.image_to_wandb = False

    if not args.single_gpu:
        opt.local_rank = local_rank
        opt.device = local_rank

    val_dataset, train_dataset = data.get_train_val_dataloader(opt.data)

    # UNet model settings
    model, model_ema, optimizer, scheduler = trainer.get_model_optimizer_and_scheduler(opt)
    # Diffusion settings
    beta_schedule = make_beta_schedule(**opt.diffusion.beta_schedule)
    diffusion = create_gaussian_diffusion(beta_schedule, predict_xstart=False)

    if is_main_process() and opt.image_to_wandb:
        wandb.init(project="CAD", name=opt.exp_name, settings=wandb.Settings(code_dir="."), resume=False)

    trainer = trainer.get_trainer(opt,
                                  diffusion, model,
                                  model_ema,
                                  optimizer,
                                  scheduler,
                                  train_dataset, val_dataset,
                                  wandb)
    trainer.load_checkpoint(opt, args.which_iter)

    iter_counter = IterationCounter(opt, len(train_dataset))

    for epoch in iter_counter.training_epochs() :
        iter_counter.record_epoch_start(epoch)
        if not args.single_gpu:
            train_dataset.sampler.set_epoch(iter_counter.epoch_iter)
        for i, data_i in enumerate(tqdm(train_dataset), start=iter_counter.epoch_iter):
            iter_counter.record_one_iteration()
            trainer.optimize_parameters(data_i)
            # record current epoch/iter
            iter_counter.record_current_iter()
            # Check training process
            trainer.end_of_iteration(data_i, iter_counter)
            if args.debug :
                break

        iter_counter.record_epoch_end()
        # trainer.end_of_epoch(val_dataset, iter_counter)
    trainer.save_checkpoint('latest')
    print('Training was successfully finished.')
