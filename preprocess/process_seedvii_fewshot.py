from pathlib import Path
import numpy as np
from pathlib import Path
# import mne
from scipy.io import loadmat
import os
import pickle
import tqdm
from utils import save_eeg, create_folder_eeg

# Use official preprocessed eeg segments
# https://bcmi.sjtu.edu.cn/home/seed/seed.html

src_root = "/inspire/hdd/global_user/wangwei-240107010004/data/SEED-VII/EEG_preprocessed"
dst_root = "/inspire/hdd/global_user/wangwei-240107010004/BSR/data/eeg/seedvii_fewshot"
channel_order_seed = ["FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8","FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ","C2","C4","C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7","P5","P3","P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ","PO4","PO6","PO8","CB1","O1","OZ","O2","CB2"]

create_folder_eeg(root=dst_root, subjects=20, splits=["train", "test"])

# group = Path(src_root).glob('*.mat')
# group = sorted(group)
# print(group)

label = {
    1: "Happy",
    2: "Neutral",
    3: "Disgust",
    4: "Sad",
    5: "Anger",
    6: "Anger",
    7: "Sad",
    8: "Disgust",
    9: "Neutral",
    10: "Happy",
    11: "Happy",
    12: "Neutral",
    13: "Disgust",
    14: "Sad",
    15: "Anger",
    16: "Anger",
    17: "Sad",
    18: "Disgust",
    19: "Neutral",
    20: "Happy",
    21: "Anger",
    22: "Sad",
    23: "Fear",
    24: "Neutral",
    25: "Surprise",
    26: "Surprise",
    27: "Neutral",
    28: "Fear",
    29: "Sad",
    30: "Anger",
    31: "Anger",
    32: "Sad",
    33: "Fear",
    34: "Neutral",
    35: "Surprise",
    36: "Surprise",
    37: "Neutral",
    38: "Fear",
    39: "Sad",
    40: "Anger",
    41: "Happy",
    42: "Surprise",
    43: "Disgust",
    44: "Fear",
    45: "Anger",
    46: "Anger",
    47: "Fear",
    48: "Disgust",
    49: "Surprise",
    50: "Happy",
    51: "Happy",
    52: "Surprise",
    53: "Disgust",
    54: "Fear",
    55: "Anger",
    56: "Anger",
    57: "Fear",
    58: "Disgust",
    59: "Surprise",
    60: "Happy",
    61: "Disgust",
    62: "Sad",
    63: "Fear",
    64: "Surprise",
    65: "Happy",
    66: "Happy",
    67: "Surprise",
    68: "Fear",
    69: "Sad",
    70: "Disgust",
    71: "Disgust",
    72: "Sad",
    73: "Fear",
    74: "Surprise",
    75: "Happy",
    76: "Happy",
    77: "Surprise",
    78: "Fear",
    79: "Sad",
    80: "Disgust"
}

label2id = {
    "Happy": 0,
    "Surprise": 1,
    "Neutral": 2,
    "Disgust": 3,
    "Fear":4,
    "Sad":5,
    "Anger": 6,
}

subjects = range(1, 21)

# for matFile in tqdm.tqdm(group):
for subject in subjects:

    matFile = os.path.join(src_root, str(subject)+".mat")
    print(f'processing {matFile}')
    data = loadmat(matFile)
    
    for trial in range(1, 81):
        trial_data = data[str(trial)]
        if trial<21:
            split = "train"
        elif trial<41:
            split = "test"
        elif trial<61:
            split = "train"
        else:
            split = "test"
        
        trial_label = label[trial]
        label_id = label2id[trial_label]
        # print(eeg.shape, eeg_label)
        eeg_t = trial_data.shape[1]
        n_patches = eeg_t//2000
        for j in range(n_patches):
            patch = trial_data[:, j*2000:(j+1)*2000]
            out_path = os.path.join(dst_root, "subject{}".format(str(subject)), split, str(trial)+"_"+str(j)+".pkl")
            save_eeg(X=patch, label=label_id, channel_order=channel_order_seed, save_path=out_path)
                # print(out_path)
                # pickle.dump(
                #     {"X": patch, "y": eeg_label+1},
                #     open(out_path, "wb"),
                # )
                
        
        