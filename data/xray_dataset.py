from data.base_dataset import BaseDataset
import os
import pandas as pd
import random
import torch

class XrayDataset(BaseDataset) :

    def __getitem__(self, index):
        pid, filename, lnum, mnum, rnum, tnum, label = self.df.iloc[index]

        xray_path = os.path.join(self.opt.path, 'PeX-ray', filename, 'image_list.txt')
        fd = open(os.path.join(xray_path))
        xray_list = fd.readlines()
        fd.close()
        xray_list.sort()

        start = random.randint(0, len(xray_list) -self.opt.sub_frame)

        image_frame = []

        if self.is_inference :
            for file in xray_list:
                image, param = self.get_image_tensor(file[:-1])
                image_frame.append(image)
            start = 0
        else :
            for file in xray_list[start:start+self.opt.sub_frame] :
                image, param = self.get_image_tensor(file[:-1])
                image_frame.append(image)

        image_frame = torch.stack(image_frame, 1)

        return {'ct': image_frame,
                'start_frame': start,
                'filename': filename,
                'label': label}

    def __len__(self):
        return len(self.df)

