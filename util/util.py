import matplotlib.pyplot as plt
import random
import numpy as np
import torch
def print_PILimg(img) :
    plt.imshow(img)
    plt.show()


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_network(epoch, opt):
    save_filename = '%s_net.pth' % (epoch)
    save_dir = os.path.join(opt.checkpoints_dir, opt.id)
    save_path = os.path.join(save_dir, save_filename)
    ckpt = torch.load(save_path, map_location=lambda storage, loc: storage)

    return ckpt
def save_network(G, D, optG, optD, epoch, opt):
    save_filename = '%s_net.pth' % (epoch)
    save_path = os.path.join(opt.checkpoints_dir, opt.id, save_filename)

    torch.save(
        {
            "netG": G.state_dict(),
            "netD": D.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict()
        },
        save_path
    )


