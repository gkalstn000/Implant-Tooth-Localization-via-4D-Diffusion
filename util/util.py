import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import cv2
import os
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


def save_videos_as_files(samples, save_root, filenames, labels, fps=30):
    for i in range(samples.shape[0]):  # 각 배치 아이템에 대해 반복
        filename = filenames[i]
        label = labels[i]
        save_path = os.path.join(save_root, f'{label}_{filename}.mp4')
        video = cv2.VideoWriter(save_path,
                                cv2.VideoWriter_fourcc(*'XVID'),
                                fps,  # FPS, 필요에 따라 조절 가능
                                (samples.shape[4], samples.shape[3]),
                                isColor=False)  # (Width, Height)

        for f in range(samples.shape[2]):  # 각 프레임에 대해 반복
            frame = samples[i, 0, f, :, :]  # 현재 프레임을 추출 (채널 차원 제거)
            video.write(frame.numpy().astype(np.uint8))

        video.release()  # 비디오 파일 작성 완료