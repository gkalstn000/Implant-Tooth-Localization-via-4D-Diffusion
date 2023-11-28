from data.base_dataset import BaseDataset, get_transform, get_random_params
import os
import pandas as pd
import random
import torch
from PIL import Image

class ClassifierDataset(BaseDataset) :
    def initialize(self, opt, is_inference):
        self.opt = opt
        self.load_size = (opt.resolution, opt.resolution) if isinstance(opt.resolution, int) else opt.resolution
        self.maxframe = opt.maxframe
        self.is_inference = is_inference
        self.df = self.get_paths(os.path.join(opt.path, 'xray'))
        self.preprocess_mode = opt.preprocess_mode
        self.scale_param = opt.scale_param if not is_inference else 0

    # def get_paths(self, root):
    #     if self.is_inference :
    #         df = pd.read_csv(os.path.join(root, 'test.csv'))
    #     else :
    #         df = pd.read_csv(os.path.join(root, 'train.csv'))
    #     return df

    def get_image_tensor(self, path):

        img = Image.open(path).convert('L').resize(self.load_size)
        param = get_random_params(img.size, self.scale_param)
        trans = get_transform(param, normalize=True, toTensor=True)
        img = trans(img)
        return img, param
    def __getitem__(self, index):
        pid, filename, lnum, mnum, rnum, tnum, label = self.df.iloc[index]

        image_frame = []
        diff_frame = []
        seg_frame = []

        for index in range(self.opt.maxframe) :
            image_path = os.path.join(self.opt.path, self.opt.name, self.opt.exp_name, 'original', filename, f'{index:02}.png')
            image, _ = self.get_image_tensor(image_path)
            image_frame.append(image)
        image_frame = torch.stack(image_frame, 1)

        for index in range(self.opt.maxframe) :
            image_path = os.path.join(self.opt.path, self.opt.name, self.opt.exp_name, 'diff', filename, f'{index:02}.png')
            image, _ = self.get_image_tensor(image_path)
            diff_frame.append(image)
        diff_frame = torch.stack(diff_frame, 1)

        for index in range(self.opt.maxframe) :
            image_path = os.path.join(self.opt.path, self.opt.name, self.opt.exp_name, 'diff_seg', filename, f'{index:02}.png')
            image, _ = self.get_image_tensor(image_path)
            seg_frame.append(image)
        seg_frame = torch.stack(seg_frame, 1)

        return {'ct': image_frame,
                'diff': diff_frame,
                'seg': seg_frame,
                'filename': filename,
                'label': label}

    def __len__(self):
        return len(self.df)

