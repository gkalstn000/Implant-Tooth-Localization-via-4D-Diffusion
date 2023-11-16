from data.base_dataset import BaseDataset
import os
import pandas as pd
import random
import lmdb
import torch

class CadDataset(BaseDataset) :

    def initialize(self, opt, is_inference):
        self.opt = opt
        self.load_size = (opt.resolution, opt.resolution)
        self.maxframe = opt.maxframe
        self.is_inference = is_inference
        self.df = self.get_paths(opt.path)
        self.preprocess_mode = opt.preprocess_mode
        self.scale_param = opt.scale_param if not is_inference else 0

        path = os.path.join(opt.path, f'256-256')
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get_paths(self, root):
        if self.is_inference :
            df1 = pd.read_csv(os.path.join(root, 'positive.csv'))
            df2 = pd.read_csv(os.path.join(root, f'negative_{self.opt.test_fold}.csv'))
            df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
        else :
            fold_list = [1, 2, 3]
            fold_list.pop(fold_list.index(self.opt.test_fold))
            df1 = pd.read_csv(os.path.join(root, f'negative_{fold_list[0]}.csv'))
            df2 = pd.read_csv(os.path.join(root, f'negative_{fold_list[1]}.csv'))
            df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

        return df

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

