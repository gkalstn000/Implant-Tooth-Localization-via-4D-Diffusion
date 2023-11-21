from data.base_dataset import BaseDataset
import os
import pandas as pd
import random
import lmdb
import torch

class CTDataset(BaseDataset) :

    def __getitem__(self, index):
        pid, sex, age, xraytube, filename, label = self.df.iloc[index]

        ct_path = os.path.join(self.opt.path, 'CBCT', filename, 'image_list.txt')
        fd = open(os.path.join(ct_path))
        ct_list = fd.readlines()
        fd.close()
        ct_list.sort()

        clip_index = self.opt.maxframe
        ct_length = len(ct_list)
        ct_list = ct_list[(ct_length - clip_index) // 2 : (ct_length - clip_index) // 2 + clip_index]

        start = random.randint(0, len(ct_list) - self.opt.sub_frame)

        assert clip_index == len(ct_list), 'CT frame 개수가 100개보다 적음'
        image_frame = []

        if self.is_inference :
            for file in ct_list:
                image, param = self.get_image_tensor(file[:-1])
                image_frame.append(image)
            start = 0
        else :
            for file in ct_list[start:start+self.opt.sub_frame] :
                image, param = self.get_image_tensor(file[:-1])
                image_frame.append(image)

        image_frame = torch.stack(image_frame, 1)
        patient_info = torch.tensor([sex, age, xraytube])

        return {'ct': image_frame,
                'info': patient_info,
                'start_frame': start,
                'filename': filename,
                'label': label}

    def __len__(self):
        return len(self.df)

