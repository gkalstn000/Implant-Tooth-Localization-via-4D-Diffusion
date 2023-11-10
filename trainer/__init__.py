import importlib
import random

import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam, lr_scheduler
from util.distributed import master_only_print as print

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def get_trainer(opt,
                diffusion, model,
                model_ema,
                optimizer,
                scheduler,
                train_dataset, val_dataset,
                wandb):
    module, trainer_name = opt.trainer.type.split('::')

    trainer_lib = importlib.import_module(module)
    trainer_class = getattr(trainer_lib, trainer_name)
    trainer = trainer_class(opt,
                            diffusion, model,
                            model_ema,
                            optimizer,
                            scheduler,
                            train_dataset, val_dataset,
                            wandb)
    return trainer

def get_model_optimizer_and_scheduler(opt):
    model_module, model_name = opt.model.type.split('::')
    lib = importlib.import_module(model_module)
    network = getattr(lib, model_name)
    model = network(opt.model.param).to(opt.device)
    model_ema = network(opt.model.param).to(opt.device)
    model_ema.eval()
    accumulate(model_ema, model, 0)
    print('net [{}] parameter count: {:,}'.format(
        'model', _calculate_model_size(model)))


    optimizer = get_optimizer(opt.optimizer, model)

    if opt.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # Scheduler
    scheduler = get_scheduler(opt.scheduler, optimizer)
    return model, model_ema, optimizer, scheduler


def _calculate_model_size(model):
    r"""Calculate number of parameters in a PyTorch network.

    Args:
        model (obj): PyTorch network.

    Returns:
        (int): Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_scheduler(opt_opt, opt):
    """Return the scheduler object.

    Args:
        opt_opt (obj): Config for the specific optimization module (gen/dis).
        opt (obj): PyTorch optimizer object.

    Returns:
        (obj): Scheduler
    """
    if opt_opt.type == 'step':
        scheduler = lr_scheduler.StepLR(
            opt,
            step_size=opt_opt.step_size,
            gamma=opt_opt.gamma)
    elif opt_opt.type == 'constant':
        scheduler = lr_scheduler.LambdaLR(opt, lambda x: 1)

    else:
        return NotImplementedError('Learning rate policy {} not implemented.'.
                                   format(opt_opt.lr_policy.type))
    return scheduler


def get_optimizer(opt_opt, net):
    return get_optimizer_for_params(opt_opt, net.parameters())

def get_optimizer_for_params(opt_opt, params):
    r"""Return the scheduler object.

    Args:
        opt_opt (obj): Config for the specific optimization module (gen/dis).
        params (obj): Parameters to be trained by the parameters.

    Returns:
        (obj): Optimizer
    """
    # We will use fuse optimizers by default.
    if opt_opt.type == 'adam':
        opt = Adam(params,
                   lr=opt_opt.lr,
                   betas=(opt_opt.adam_beta1, opt_opt.adam_beta2))
    else:
        raise NotImplementedError(
            'Optimizer {} is not yet implemented.'.format(opt_opt.type))
    return opt



