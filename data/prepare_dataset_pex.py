import os
import numpy as np
import pandas as pd
from PIL import Image
from util.util import print_PILimg
import pydicom
import lmdb
import glob
from torchvision.transforms import functional as trans_fn
from io import BytesIO
import multiprocessing
from tqdm import tqdm

size = (160, 120)
def gen_image_fullXray(filepath) :
    pre, post = filepath[:-1].split('/')[:3], filepath[:-1].split('/')[3]
    order_idx, tail = post.split('-')
    order_idx = int(order_idx)
    tail = tail.replace('png', 'tif')
    up_file = os.path.join(dataroot, '/'.join(pre), f'{1}_{order_idx}-{tail}')
    down_file= os.path.join(dataroot, '/'.join(pre), f'{0}_{order_idx}-{tail}')

    zero = Image.fromarray(np.zeros((120, 160)).astype(np.uint8))

    if os.path.exists(up_file) :
        up_img = Image.open(up_file).convert('L').resize(size)
        assert up_img.size == size, f'{up_file} size가 {size}가 아니라 {up_img.size}'
        up_img = up_img.transpose(Image.FLIP_TOP_BOTTOM)
    else :
        up_img = zero

    if os.path.exists(down_file) :
        down_img = Image.open(down_file).convert('L').resize(size)
        assert down_img.size == size, f'{down_file} size가 {size}가 아니라 {down_img.size}'
    else :
        down_img = zero

    image_data = np.concatenate([up_img, down_img], axis=0)
    return Image.fromarray(image_data)


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(5)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')

class Resizer:
    def __init__(self, *, size, root):
        self.size = size
        self.root = root

    def get_resized_bytes(self, img):
        # img = trans_fn.resize(img, self.size, Image.BILINEAR)
        buf = BytesIO()
        img.save(buf, format='png')
        img_bytes = buf.getvalue()
        return img_bytes

    def prepare(self, filename):
        # filename = os.path.join(self.root, filename)
        img = gen_image_fullXray(filename)
        img_bytes = self.get_resized_bytes(img)
        return img_bytes

    def __call__(self, index_filename):
        index, filename = index_filename
        result_img = self.prepare(filename)
        return index, result_img, filename

def get_file_list_from_PatientID(df) :

    df_new = {column:[] for column in df.columns}

    for idx, columns in df.iterrows() :
        filename = columns['FileName']
        folder_path = os.path.join(dataroot, 'PeX-ray', filename, 'middle')
        if not os.path.exists(folder_path): continue

        for key, val in columns.items() :
            df_new[key].append(val)

        filelist = []

        for index in range(16) :
            formatted_index = '{:02}'.format(index)
            save_path = os.path.join('cad', 'PeX-ray', filename, 'middle', f'{formatted_index}-drr.png')
            filelist.append(save_path)
        filelist.sort()
        with open(os.path.join(dataroot, 'PeX-ray', filename, 'image_list.txt'), "w") as f:
            for path in filelist:
                f.write(path.replace('cad/', '')+ "\n" )

    return pd.DataFrame.from_dict(df_new)

dataroot = '/datasets/msha/cad'

Implant_marking_df = pd.read_csv(os.path.join(dataroot, 'Implant_Marking.csv'))
patient_statistics_df = pd.read_csv(os.path.join(dataroot, 'patient_statistics_info.csv'))

CBCT_df = pd.read_csv(os.path.join(dataroot, 'CBCT_info.csv'))
PaX_df = pd.read_csv(os.path.join(dataroot, 'PaX_info.csv'))
PeX_df = pd.read_csv(os.path.join(dataroot, 'PeX_info.csv'))

# Implant_marking_df.shape = (574, 3)
# CBCT_df.shape = (574, 18)
df = pd.merge(PeX_df, Implant_marking_df, on=['FileName', 'PatientID'])

# DataSplit
positive_df = df[df.Label == 1]
negative_df = df[df.Label == 0]
negative_df = negative_df.sample(frac=1).reset_index(drop=True)
length = len(negative_df)
negative_fold_1_df = negative_df.iloc[:length//3]
negative_fold_2_df = negative_df.iloc[length//3:2*length//3]
negative_fold_3_df = negative_df.iloc[2*length//3:]

positive_df = get_file_list_from_PatientID(positive_df)
negative_fold_1_df = get_file_list_from_PatientID(negative_fold_1_df)
negative_fold_2_df = get_file_list_from_PatientID(negative_fold_2_df)
negative_fold_3_df = get_file_list_from_PatientID(negative_fold_3_df)

# Save csv files
positive_df.to_csv(os.path.join(dataroot, 'xray',  'positive.csv'), index=False)
negative_fold_1_df.to_csv(os.path.join(dataroot, 'xray', 'negative_1.csv'), index=False)
negative_fold_2_df.to_csv(os.path.join(dataroot, 'xray', 'negative_2.csv'), index=False)
negative_fold_3_df.to_csv(os.path.join(dataroot, 'xray', 'negative_3.csv'), index=False)

#Save images
xray_root = os.path.join(dataroot, 'PeX-ray')
out = dataroot
lmdb_save_path = os.path.join(out, 'xray')
os.makedirs(lmdb_save_path, exist_ok=True)
n_worker = 8
chunksize = 10


xray_file_paths = []
for filename in pd.concat([positive_df, negative_fold_1_df, negative_fold_2_df, negative_fold_3_df], axis=0).FileName :
    fd = open(os.path.join(dataroot, 'PeX-ray', filename, 'image_list.txt'))
    xray_list = fd.readlines()
    fd.close()
    xray_file_paths.extend(xray_list)
total = len(xray_file_paths)
xray_file_paths.sort()

with lmdb.open(lmdb_save_path, map_size=1024 ** 4, readahead=False) as env:
    with env.begin(write=True) as txn:
        txn.put(format_for_lmdb('length'), format_for_lmdb(total))
        resizer = Resizer(size=size, root=xray_root)
        with multiprocessing.Pool(n_worker) as pool:
            for idx, result_img, filename in tqdm(
                    pool.imap_unordered(resizer, enumerate(xray_file_paths), chunksize=chunksize),
                    total=total):

                filename = os.path.splitext(filename)[0] + '.png'
                # filename = '/'.join(filename.split('/')[4:])
                txn.put(format_for_lmdb(filename), result_img)



