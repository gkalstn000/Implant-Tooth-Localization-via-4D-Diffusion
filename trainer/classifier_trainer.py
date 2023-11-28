import os
import math
import importlib
from tqdm import tqdm, trange
import random
import numpy as np
from torch import nn, optim
import torch
from trainer.base import BaseTrainer

from torch import autograd
import torch.distributions as dist
import torch.nn.functional as F
from trainer import accumulate
from trainer.base import BaseTrainer
from models.diffusion import ddim_steps_no_cond as ddim_steps
from collections import defaultdict
from util.util import print_PILimg
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
        self.BCE = nn.BCELoss()
        print('Setup trainer.')


    def preprocess_input(self, data):
        img_frame, filename, label = data['ct'], data['filename'], data['label']

        return img_frame.to(torch.device('cuda')).float(), filename, label.to(torch.device('cuda')).unsqueeze(-1).float()
    def optimize_parameters(self, data):
        r"""Training step of model with diffusion
        Args:
            data (dict): data used in the training step

        output_dict = {'ct': image_frame,
                'info': patient_info,
                'filename': filename}
        """
        self.model.train()
        img_frame, filename, label = self.preprocess_input(data)
        img_frame = img_frame * (1-0.02) + torch.rand_like(img_frame).to(img_frame.device) * 0.02

        output = self.model(img_frame)
        loss = self.BCE(output, label)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()

        accumulate(self.model_ema, self.model_module, 0.9999)

        self.loss_dict['BCE'] = loss * 20


    def test(self, data):
        img_frame, filename, label = self.preprocess_input(data)
        img_frame = img_frame * (1-0.02) + torch.rand_like(img_frame).to(img_frame.device) * 0.02

        self.model_ema.eval()
        with torch.no_grad():
            output = self.model_ema(img_frame)

        return label.squeeze().cpu().tolist(), output.squeeze().cpu().tolist()