import os
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


def gen_image_from_ct(filepath) :
    dicom_file = pydicom.dcmread(filepath)
    image_data = dicom_file.pixel_array
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
        img = trans_fn.resize(img, self.size, Image.BICUBIC)
        buf = BytesIO()
        img.save(buf, format='png')
        img_bytes = buf.getvalue()
        return img_bytes

    def prepare(self, filename):
        # filename = os.path.join(self.root, filename)
        img = gen_image_from_ct(filename)
        img_bytes = self.get_resized_bytes(img)
        return img_bytes

    def __call__(self, index_filename):
        index, filename = index_filename
        result_img = self.prepare(filename)
        return index, result_img, filename

def get_file_list_from_PatientID(df) :

    df_new = {'PatientID':[],
              'PatientSex': [],
              'PatientAge': [],
              'XRayTubeCurrent': [],
              'FileName': [],
              'Label': []
              }
    for idx, (PatientID, PatientSex, PatientAge, XRayTubeCurrent, FileName, Label) in df.iterrows() :
        folder_path = os.path.join(dataroot, 'CBCT', FileName)
        if not os.path.exists(folder_path) : continue
        df_new['PatientID'].append(PatientID)
        df_new['PatientSex'].append(PatientSex)
        df_new['PatientAge'].append(PatientAge)
        df_new['XRayTubeCurrent'].append(XRayTubeCurrent)
        df_new['FileName'].append(FileName)
        df_new['Label'].append(Label)

        filelist = []
        for filename in os.listdir(folder_path) :
            if 'dcm' not in filename : continue
            filename = filename.replace('dcm', 'png')
            filelist.append(os.path.join('cad', 'CBCT', FileName, filename))

        with open(os.path.join(dataroot, 'CBCT', FileName, 'image_list.txt'), "w") as f:
            for path in filelist:
                f.write(path.replace('cad/', '') + "\n")

    return pd.DataFrame.from_dict(df_new)

dataroot = '/datasets/msha/cad'

Implant_marking_df = pd.read_csv(os.path.join(dataroot, 'Implant_Marking.csv'))
patient_statistics_df = pd.read_csv(os.path.join(dataroot, 'patient_statistics_info.csv'))

CBCT_df = pd.read_csv(os.path.join(dataroot, 'CBCT_info.csv'))
PaX_df = pd.read_csv(os.path.join(dataroot, 'PaX_info.csv'))
PeX_df = pd.read_csv(os.path.join(dataroot, 'PeX_info.csv'))

# Implant_marking_df.shape = (574, 3)
# CBCT_df.shape = (574, 18)
df = pd.merge(CBCT_df, Implant_marking_df, on=['FileName', 'PatientID'])
df['PatientSex'] = df['PatientSex'].map({'M': 1, 'F': 0})
df['PatientAge'] = df['PatientAge'].str.rstrip('Y').astype(int)

columns_to_drop = ['Modality', 'StudyDate', 'StudyTime', 'KVP', 'Rows', 'Columns',
                   'BitsAllocated', 'BitsStored', 'HighBit', 'WindowCenter', 'WindowWidth', 'SliceThickness', 'PixelSpacing']

df = df.drop(columns=columns_to_drop)

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
positive_df.to_csv(os.path.join(dataroot, 'positive.csv'), index=False)
negative_fold_1_df.to_csv(os.path.join(dataroot, 'negative_1.csv'), index=False)
negative_fold_2_df.to_csv(os.path.join(dataroot, 'negative_2.csv'), index=False)
negative_fold_3_df.to_csv(os.path.join(dataroot, 'negative_3.csv'), index=False)

#Save images
CT_root = os.path.join(dataroot, 'CBCT')
out = dataroot
size = (256, 256)
lmdb_save_path = os.path.join(out, '256-256')
os.makedirs(lmdb_save_path, exist_ok=True)
n_worker = 8
chunksize = 10

dcm_file_paths = glob.glob(os.path.join(CT_root, '**', '*.dcm'), recursive=True)
total = len(dcm_file_paths)
#
#
#
# with lmdb.open(lmdb_save_path, map_size=1024 ** 4, readahead=False) as env:
#     with env.begin(write=True) as txn:
#         txn.put(format_for_lmdb('length'), format_for_lmdb(total))
#         resizer = Resizer(size=size, root=CT_root)
#         with multiprocessing.Pool(n_worker) as pool:
#             for idx, result_img, filename in tqdm(
#                     pool.imap_unordered(resizer, enumerate(dcm_file_paths), chunksize=chunksize),
#                     total=total):
#
#                 filename = os.path.splitext(filename)[0] + '.png'
#                 filename = '/'.join(filename.split('/')[4:])
#                 txn.put(format_for_lmdb(filename), result_img)



