import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np
import torch
import cv2
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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
        print(f'Video saved: {save_path}')

def save_videos_to_images(samples, save_root, filenames, labels) :


    for i in range(samples.shape[0]):
        _, input_, gen, diff, diff_seg = torch.chunk(samples[i], 5, -1)
        filename = filenames[i]

        input_path = os.path.join(save_root, 'original', filename)
        os.makedirs(input_path, exist_ok=True)
        diff_path = os.path.join(save_root, 'diff', filename)
        os.makedirs(diff_path, exist_ok=True)
        diff_seg_path = os.path.join(save_root, 'diff_seg', filename)
        os.makedirs(diff_seg_path, exist_ok=True)

        for idx, (origin, diff, diff_s) in enumerate(zip(input_[0], diff[0], diff_seg[0])) :
            origin_PIL = Image.fromarray(origin.numpy().astype(np.uint8))
            origin_PIL.save(os.path.join(input_path, f'{idx:02}.png'))
            diff_PIL = Image.fromarray(diff.numpy().astype(np.uint8))
            diff_PIL.save(os.path.join(diff_path, f'{idx:02}.png'))
            diff_seg_PIL = Image.fromarray(diff_s.numpy().astype(np.uint8))
            diff_seg_PIL.save(os.path.join(diff_seg_path, f'{idx:02}.png'))



def calculate_scores(true_list, pred_list, threshold=0.5):
    # Convert probabilities to class predictions based on the threshold
    pred_classes = [1 if prob > threshold else 0 for prob in pred_list]

    # Calculate various scores
    scores = {
        'valid/accuracy': accuracy_score(true_list, pred_classes),
        'valid/f1': f1_score(true_list, pred_classes),
        'valid/precision': precision_score(true_list, pred_classes),
        'valid/recall': recall_score(true_list, pred_classes)
    }

    return scores

def video_minmax(video) :
    b, c, f, h, w = video.size()

    for i in range(b):
        for j in range(f):
            frame = video[i, :, j, :, :]
            min_val = torch.min(frame)
            max_val = torch.max(frame)
            if max_val - min_val > 0:  # To avoid division by zero
                video[i, :, j, :, :] = (frame - min_val) / (max_val - min_val)
    return video
