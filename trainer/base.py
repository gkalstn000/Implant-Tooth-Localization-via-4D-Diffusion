from email.policy import strict
import os
import glob
import time

import torch
import numpy as np
import torchvision
from torch import nn
from util.distributed import is_master, master_only, is_main_process
from util.distributed import master_only_print as print
import torch.distributed as dist
from PIL import Image
class BaseTrainer(object):
    def __init__(self,
                 opt,
                 diffusion, model,
                 model_ema,
                 optimizer,
                 scheduler,
                 train_data_loader, val_data_loader=None,
                 wandb=None):
        super(BaseTrainer, self).__init__()
        print('Setup trainer.')

        # Initialize models and data loaders.
        self.opt = opt
        self.diffusion = diffusion
        self.model = model
        if opt.distributed:
            self.model_module = self.model.module
        else:
            self.model_module = self.model

        self.is_inference = train_data_loader is None
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.wandb = wandb

        if self.is_inference:
            return

        self.loss_dict = {'loss_total': 0,
                          'loss_vb': 0,
                          'loss_mse': 0,}

    def _pre_save_checkpoint(self):
        pass

    def save_checkpoint(self, total_iteration):
        self._pre_save_checkpoint()
        _save_checkpoint(self.opt,
                         self.model, self.model_ema,
                         self.optimizer,
                         self.scheduler,
                         total_iteration)

    def load_checkpoint(self, opt, which_iter=None):
        if which_iter is not None:
            model_path = os.path.join(opt.save_root, '{:09}_checkpoint.pt'.format(which_iter))
            latest_checkpoint_path = glob.glob(model_path)
            assert len(latest_checkpoint_path) <= 1, "please check the saved model {}".format(
                model_path)
            if len(latest_checkpoint_path) == 0:
                print('No checkpoint found at iteration {}.'.format(which_iter))
                return None
            checkpoint_path = latest_checkpoint_path[0]

        elif os.path.exists(os.path.join(opt.save_root, 'latest_checkpoint.pt')):
            checkpoint_path = os.path.join(opt.save_root, 'latest_checkpoint.pt')
        else:
            print('No checkpoint found.')
            return None
        resume = opt.phase == 'train' and opt.resume
        if resume :
            self._load_checkpoint(checkpoint_path, resume)


    def _load_checkpoint(self, checkpoint_path, resume=True):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.model_ema.load_state_dict(checkpoint['model_ema'])
        print('load [model_ema] from {}'.format(checkpoint_path))

        if self.opt.phase == 'train':
            if 'model' not in checkpoint:
                self.model_module.load_state_dict(checkpoint['model_ema'])
                print('load_from_model_ema')
            else:
                self.model.load_state_dict(checkpoint['model'])
            print('load [model] from {}'.format(checkpoint_path))
            if resume:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                print('load optimizers and schdules from {}'.format(checkpoint_path))

        print('Done with loading the checkpoint.')


    def end_of_epoch(self, val_dataset, iter_counter):
        # Update the learning rate policy for the generator if operating in the
        # Logging.
        current_epoch = iter_counter.current_epoch
        elapsed_epoch_time = iter_counter.time_per_epoch
        print('Epoch: {}, total time: {:6f}.'.format(current_epoch,
                                                     elapsed_epoch_time))
        self._end_of_epoch(val_dataset, iter_counter)

        if iter_counter.needs_saving() and is_main_process() :
            # Plot Validation dataset
            data_iter = iter(val_dataset)
            data = next(data_iter)
            self.save_image(data, True)
            self.save_checkpoint('latest')


    def end_of_iteration(self, data, iter_counter):
        current_epoch = iter_counter.current_epoch
        total_iteration = iter_counter.total_steps_so_far
        time_per_iter = iter_counter.time_per_iter
        # Logging.
        if iter_counter.needs_printing() and is_main_process():
            self._print_current_errors(current_epoch, total_iteration, time_per_iter)
            self._write_wandb(current_epoch, total_iteration)

        self._end_of_iteration(data, current_epoch, total_iteration)

        if iter_counter.needs_displaying() and is_main_process():
            self.save_image(data, False)
        # Save everything to the checkpoint.
        if iter_counter.needs_saving() and is_main_process():
            self.save_checkpoint(iter_counter.total_steps_so_far)
            self.save_checkpoint('latest')
    def _write_wandb(self, current_epoch, total_iteration):
        log_dict = {}
        for loss_name, losses in self.loss_dict.items():
            log_dict['model_update' + '/' + loss_name] = losses
        log_dict['epoch'] = current_epoch
        log_dict['iter']= total_iteration
        if self.opt.image_to_wandb :
            self.wandb.log(log_dict)
    def _print_current_errors(self, current_epoch, current_iteration, time_per_iter):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (current_epoch, current_iteration, time_per_iter)
        for loss_name, losses in self.loss_dict.items():
            full_loss_name = 'model_update' + '/' + loss_name
            message += '%s: %.3f ' % (full_loss_name, losses)

        print(message)
        log_name = os.path.join(self.opt.save_root, 'loss_log.txt')
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)


    def save_image(self, data, is_valid, num_grid=1):
        samples, filename = self._get_visualizations(data, is_valid)
        gt_grid = image_frame_to_grid(data['ct'][:num_grid]) * 255
        gen_grid = image_frame_to_grid(samples[:num_grid]) * 255
        diff = torch.abs(gt_grid-gen_grid)
        vis = torch.cat([gt_grid, gen_grid, diff], 0)

        gathered_samples = [torch.zeros_like(vis) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, vis)
        gathered_samples = torch.cat(gathered_samples, 1)

        tag = 'valid' if is_valid else 'train'

        if is_main_process():
            save_path = os.path.join(self.opt.save_root, 'images')
            os.makedirs(save_path, exist_ok=True)
            filename_cat = '_AND_'.join(filename) + '.png'
            image = Image.fromarray(gathered_samples.cpu().numpy().squeeze().astype(np.uint8)).convert('L')
            image.save(os.path.join(save_path, filename_cat))
            if self.opt.image_to_wandb :
                self.wandb.log({f'{tag} samples': self.wandb.Image(image)})

    def _get_save_path(self, subdir, ext):
        subdir_path = os.path.join(self.opt.logdir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
        return os.path.join(
            subdir_path, 'epoch_{:05}_iteration_{:09}.{}'.format(
                self.current_epoch, self.current_iteration, ext))

    def _start_of_epoch(self, current_epoch):
        pass

    def _start_of_iteration(self, data, current_iteration):
        return data

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        pass
    
    def _end_of_epoch(self, data, iter_counter):
        pass

    def _get_visualizations(self, data, is_valid):
        return None, None

    def _init_loss(self, opt):
        raise NotImplementedError

    def optimize_parameters(self, data):
        raise NotImplementedError

    def test(self, data_loader, output_dir, current_iteration):
        raise NotImplementedError

@master_only
def _save_checkpoint(opt,
                     model, model_ema,
                     optimizer,
                     scheduler,
                     total_iteration):

    if isinstance(total_iteration, str) :
        latest_checkpoint_path = 'latest_checkpoint.pt'
    else :
        latest_checkpoint_path = '{:09}_checkpoint.pt'.format(total_iteration)
    save_path = os.path.join(opt.save_root, latest_checkpoint_path)
    torch.save(
        {
            'model': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
        save_path,
    )
    print(f'Save Checkpoints {latest_checkpoint_path} Done')
    return save_path

def image_frame_to_grid(frame) :
    frame = frame.cuda()
    b, c, f, h, w = frame.size()
    # Normalize
    min_HW, max_HW = get_min_max_over_HW(frame)
    frame = (frame - min_HW) / (max_HW - min_HW)

    frame = frame.permute(0, 3, 2, 4, 1).contiguous().view(b, h, w*f, c)
    grid_image = frame.permute(1, 0, 2, 3).contiguous().view(h, w * f * b, c)
    return grid_image

def get_min_max_over_HW(frame) :
    # Find the minimum across height (dim=3), but not keeping the dimensions
    min_over_height, _ = frame.min(dim=3)

    # Now, min_over_height is of shape (b, c, f, w), and you can call min again if you want to reduce over width
    min_over_height_and_width, _ = min_over_height.min(dim=3)  # Note that now width is dim=3

    # If you wanted to keep the reduced dimensions you can do:
    min_over_height, _ = frame.min(dim=3, keepdim=True)
    min_over_height_and_width, _ = min_over_height.min(dim=4, keepdim=True)

    # Max
    max_over_height, _ = frame.max(dim=3)

    # Now, min_over_height is of shape (b, c, f, w), and you can call min again if you want to reduce over width
    max_over_height_and_width, _ = max_over_height.max(dim=3)  # Note that now width is dim=3

    # If you wanted to keep the reduced dimensions you can do:
    max_over_height, _ = frame.max(dim=3, keepdim=True)
    max_over_height_and_width, _ = max_over_height.max(dim=4, keepdim=True)

    return min_over_height_and_width, max_over_height_and_width