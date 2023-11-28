import os
import math
import importlib
from tqdm import tqdm, trange
import random
import numpy as np
from torch import nn, optim
import torch
from torch import autograd
import torch.distributions as dist
import torch.nn.functional as F
from trainer import accumulate
from trainer.base import BaseTrainer
from models.diffusion import ddim_steps_no_cond as ddim_steps
from collections import defaultdict
from util.util import print_PILimg, video_minmax
class Trainer(BaseTrainer):
    def __init__(self,
                 opt,
                 diffusion, model,
                 model_ema,
                 optimizer,
                 scheduler,
                 train_data_loader, val_data_loader=None,
                 wandb=None):
        super(Trainer, self).__init__(opt,
                                               diffusion, model,
                                               model_ema,
                                               optimizer,
                                               scheduler,
                                               train_data_loader, val_data_loader,
                                               wandb)

        self.accum = 0.5 ** (32 / (10 * 1000))
        height = width = opt.data.resolution
        self.load_size = (int(height), int(width))


    def preprocess_input(self, data):
        img_frame, start,  filename = data['ct'], data['start_frame'], data['filename']

        return img_frame.to(torch.device('cuda')).float(), start.to(torch.device('cuda')), filename
    def optimize_parameters(self, data):
        r"""Training step of model with diffusion
        Args:
            data (dict): data used in the training step

        output_dict = {'ct': image_frame,
                'info': patient_info,
                'filename': filename}
        """
        self.model.train()
        img_frame, start, filename = self.preprocess_input(data)
        device = img_frame.device

        time_t = torch.randint(0,
                               self.opt.diffusion.beta_schedule.n_timestep,
                               (img_frame.shape[0],),
                               device=device)

        loss_dict = self.diffusion.training_losses(self.model,
                                                   x_start = [img_frame, start],
                                                   cond_input= None,
                                                   t = time_t,
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
        img_frame, start, filename = self.preprocess_input(data)
        self.model_ema.eval()
        with torch.no_grad():
            if self.opt.diffusion.sample_algorithm == 'ddpm':
                print ('Sampling algorithm used: DDPM')
                samples = self.diffusion.p_sample_loop(self.model_ema,
                                                       x_cond = [start, img_frame.shape],
                                                       progress = True,
                                                       cond_scale = self.opt.diffusion.cond_scale)
            elif self.opt.diffusion.sample_algorithm == 'ddim':
                # print ('Sampling algorithm used: DDIM')
                nsteps = 150
                noise = torch.randn(img_frame.shape).cuda()
                # q_sasmple
                corrupt_level = self.opt.args.corrupt_level
                t = torch.tensor([corrupt_level] * img_frame.size(0)).to(img_frame.device)
                x_t = self.diffusion.q_sample(img_frame, t, noise=noise)

                seq = range(0, corrupt_level, corrupt_level//nsteps)
                betas = self.diffusion.betas[:corrupt_level]
                xs, x0_preds = ddim_steps(x= [x_t, start],
                                          seq= seq,
                                          model= self.model_ema,
                                          b= torch.tensor(betas).float().cuda())
                samples = xs[-1].cuda()

        return samples, x_t

    def test(self, data):
        sub_frame = self.opt.data.sub_frame
        maxframe = self.opt.data.maxframe
        steps = maxframe // sub_frame

        filenames = data['filename']
        labels = data['label']

        frame_fake = []
        x_ts = []
        for step in range(steps) :
            start = step * sub_frame
            idx = list(range(start, start+sub_frame))
            data_sub = {key:val[:, :, idx] if key == 'ct' else val for key, val in data.items()}
            data_sub['start_frame'] += start
            samples, x_t = self._get_visualizations(data_sub, True)
            frame_fake.append(samples.cpu())
            x_ts.append(x_t.cpu())

        x_t = (torch.clamp(torch.cat(x_ts, 2), -1, 1) + 1) / 2
        frame_fake = (torch.clamp(torch.cat(frame_fake, 2), -1, 1) + 1) / 2
        frame_real = (data['ct'][:, :, :steps*sub_frame] + 1) / 2
        diff = frame_real - frame_fake
        diff = torch.where(diff > 0.1, diff, 0)
        diff = video_minmax(diff)
        diff_seg = torch.where(diff > 0.5, 1, 0)
        samples = torch.cat([x_t, frame_real, frame_fake, diff, diff_seg], -1)*255
        scores = diff_seg.sum((1, 2, 3, 4))**0.5
        return samples, filenames, labels, scores.tolist()