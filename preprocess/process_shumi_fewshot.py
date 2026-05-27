import scipy
from scipy import signal
import os
import glob
import pickle
import tqdm
from utils import create_folder_eeg, save_eeg

## Reference:
## https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/preprocessing_shu.py

## For the 1st version

## Dataset Usage: Subject-Level Few-Shot Fine-Tune
channel_order_shumi=["FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "CZ", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6", "PZ", "P3", "P4", "T5", "T6", "PO3", "PO4", "OZ", "O1", "O2"]

# 1. get files & create dataset dirs
src_root = "/inspire/hdd/global_user/wangwei-240107010004/data/shumi"
dst_root = "/inspire/hdd/global_user/wangwei-240107010004/BSR/data/eeg/shumi_fewshot"
mat_files = glob.glob(os.path.join(src_root, "*.mat"))
mat_files = sorted(mat_files)
print(len(mat_files))
create_folder_eeg(dst_root, subjects=25)

# # 2. set splits, channel order and label information
# splits = {
#     'train': mat_files[:75],
#     'val': mat_files[75:100],
#     'test': mat_files[100:],
# }
# channel_order = ["FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "CZ", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6", "PZ", "P3", "P4", "T5", "T6", "PO3", "PO4", "OZ", "O1", "O2"]

# # 3. processing

for mat_file in mat_files:
    file_name = mat_file.split('/')[-1]
    sub = int(file_name.split('_')[0].split('-')[-1], 10)
    ses = int(file_name.split('_')[1].split('-')[-1], 10)
    # print(file_name, sub, ses)
    if ses < 3:
        split = "train"
    elif ses < 4:
        split = "val"
    else:
        split = "test"
    
    data = scipy.io.loadmat(mat_file)
    eeg = data['data']
    labels = data['labels'][0]
    eeg_resample = signal.resample(eeg, 800, axis=2)
    n_sample, _, _ = eeg.shape
    for i in range(n_sample):
        X = eeg_resample[i, :, :]
        y = labels[i]
        save_path = os.path.join(dst_root, "subject"+str(sub), split, file_name+"_"+str(i)+".pkl")
        save_eeg(X=X, label="left" if y==1 else "right", channel_order=channel_order_shumi, save_path=save_path)
    
# for split_key in splits.keys():
#     print("process:", split_key)
#     for file in tqdm.tqdm(splits[split_key]):
#         file_name = file.split('/')[-1][:-4]
#         data = scipy.io.loadmat(file)
#         eeg = data['data']
#         labels = data['labels'][0]
#         eeg_resample = signal.resample(eeg, 800, axis=2)
#         n_sample, _, _ = eeg.shape

#         for i in range(n_sample):
#             X = eeg_resample[i, :, :]
#             y = labels[i]
#             save_path = os.path.join(dst_root, split_key, file_name+"_"+str(i)+".pkl")
#             save_eeg(X=X, label="left" if y==1 else "right", channel_order=channel_order_shumi, save_path=save_path)
