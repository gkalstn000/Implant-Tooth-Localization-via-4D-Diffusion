import os
import math
import importlib
from tqdm import tqdm
import random
import numpy as np
from torch import nn, optim
import torch
from torch import autograd
import torch.distributions as dist
import torch.nn.functional as F
from trainer import accumulate
from trainer.base import BaseTrainer
from models.diffusion import ddim_steps
from collections import defaultdict
from util.util import print_PILimg
class DiffusionTrainer(BaseTrainer):
    def __init__(self,
                 opt,
                 diffusion, model,
                 model_ema,
                 optimizer,
                 scheduler,
                 train_data_loader, val_data_loader=None,
                 wandb=None):
        super(DiffusionTrainer, self).__init__(opt,
                                               diffusion, model,
                                               model_ema,
                                               optimizer,
                                               scheduler,
                                               train_data_loader, val_data_loader,
                                               wandb)

        self.accum = 0.5 ** (32 / (10 * 1000))
        self.log_size = int(math.log(opt.data.resolution, 2))
        height = width = opt.data.resolution
        self.load_size = (int(height), int(width))


    def preprocess_input(self, data):
        img_frame, info, start,  filename = data['ct'], data['info'], data['start_frame'], data['filename']

        return img_frame.to(torch.device('cuda')).float(), info.to(torch.device('cuda')).float(), start.to(torch.device('cuda')), filename
    def optimize_parameters(self, data):
        r"""Training step of model with diffusion
        Args:
            data (dict): data used in the training step

        output_dict = {'ct': image_frame,
                'info': patient_info,
                'filename': filename}
        """
        self.model.train()
        img_frame, info, start, filename = self.preprocess_input(data)
        device = img_frame.device

        time_t = torch.randint(0,
                               self.opt.diffusion.beta_schedule.n_timestep,
                               (img_frame.shape[0],),
                               device=device)

        loss_dict = self.diffusion.training_losses(self.model,
                                                   x_start = [img_frame, start],
                                                   t = time_t,
                                                   cond_input = info,
                                                   prob = 1 - self.opt.diffusion.guidance_prob)

        loss = loss_dict['loss'].mean()
        loss_mse = loss_dict['mse'].mean()
        loss_vb = loss_dict['vb'].mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()

        accumulate(self.model_ema, self.model_module, 0.9999)

        self.loss_dict['loss_total'] = loss_dict['loss'].mean().detach().item()
        self.loss_dict['loss_vb'] = loss_vb.detach().item()
        self.loss_dict['loss_mse'] = loss_mse.detach().item()

    def _get_visualizations(self, data, is_valid):
        img_frame, info, start, filename = self.preprocess_input(data)
        self.model_ema.eval()
        with torch.no_grad():
            if self.opt.diffusion.sample_algorithm == 'ddpm':
                print ('Sampling algorithm used: DDPM')
                samples = self.diffusion.p_sample_loop(self.model_ema,
                                                       x_cond = [info, start, img_frame.shape],
                                                       progress = True,
                                                       cond_scale = self.opt.diffusion.cond_scale)
            elif self.opt.diffusion.sample_algorithm == 'ddim':
                print ('Sampling algorithm used: DDIM')
                nsteps = 150
                noise = torch.randn(img_frame.shape).cuda()
중                # q_sasmple
                corrupt_level = 300
                t = torch.tensor([corrupt_level] * img_frame.size(0))
                x_t = self.diffusion.q_sample(img_frame, t, noise=noise)

                seq = range(0, corrupt_level, corrupt_level//nsteps)
                betas = self.diffusion.betas
                xs, x0_preds = ddim_steps(x= [noise, start],
                                          seq= seq,
                                          model= self.model_ema,
                                          b= torch.tensor(betas).float().cuda(),
                                          cond_scale= self.opt.diffusion.cond_scale,
                                          x_cond= info)
                samples = xs[-1].cuda()

        return samples, filename

