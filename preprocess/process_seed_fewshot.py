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

src_root = "/inspire/hdd/global_user/wangwei-240107010004/data/SEED/SEED/SEED_EEG/Preprocessed_EEG"
dst_root = "/inspire/hdd/global_user/wangwei-240107010004/BSR/data/eeg/seed_fewshot"
channel_order_seed = ["FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8","FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8","T7","C5","C3","C1","CZ","C2","C4","C6","T8","TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8","P7","P5","P3","P1","PZ","P2","P4","P6","P8","PO7","PO5","PO3","POZ","PO4","PO6","PO8","CB1","O1","OZ","O2","CB2"]
subject_splits = 3

create_folder_eeg(root=dst_root, subjects=15, splits=["train", "val", "test"])

group = Path(src_root).glob('*.mat')
group = sorted(group)

# 预处理参数
l_freq = 0.1
h_freq = 75.0
rsfreq = 200

#chunks参数, 1s 数据
chunks = (62, rsfreq)

visit = {i+1: [0, 0, 0] for i in range(15)}

# -1 for negative, 0 for neutral and +1 for positive
label = loadmat("/inspire/hdd/global_user/wangwei-240107010004/data/SEED/SEED/SEED_EEG/Preprocessed_EEG/label.mat")["label"][0]
print(label)

for matFile in tqdm.tqdm(group):
    if matFile.name != "label.mat":
        subject = int(matFile.name[0:-13])
        if visit[subject][0] == 0:
            split = "train"
            visit[subject][0] = 1
        elif visit[subject][1] == 0:
            split = "val"
            visit[subject][1] = 1
        else:
            split = "test"
            visit[subject][2] = 1
            
        print(f'processing {matFile.name}')
        data = loadmat(matFile)
        # print(data.keys())
        for i in range(15):
            name = None
            for key in data.keys():
                if f'eeg{i+1}' in key:
                    name = key
            if name is None:
                raise ValueError("data not found")
            # print(name)
            eeg = data[name]
            eeg_label = label[i]
            # print(eeg.shape, eeg_label)
            eeg_t = eeg.shape[1]
            n_patches = eeg_t//2000
            for j in range(n_patches):
                patch = eeg[:, j*2000:(j+1)*2000]
                out_path = os.path.join(dst_root, "subject{}".format(str(subject)), split, matFile.name[0:-4]+"_"+name+"_"+str(j)+".pkl")
                save_eeg(X=patch, label=eeg_label+1, channel_order=channel_order_seed, save_path=out_path)
                # print(out_path)
                # pickle.dump(
                #     {"X": patch, "y": eeg_label+1},
                #     open(out_path, "wb"),
                # )
                
        
        