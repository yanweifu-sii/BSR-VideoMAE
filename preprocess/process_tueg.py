from utils import create_folder_eeg, save_eeg
import mne
from pathlib import Path
import glob
from multiprocessing import Pool
import os
import pickle
from tqdm import tqdm
import numpy as np

global split
n_prcess = 32

def preprocessing_edf(edfFilePath, notch_freq=60, l_freq=0.1, h_freq=75.0, sfreq:int=200, n_jobs=5, channel_order=None):
    # reading edf
    raw = mne.io.read_raw_edf(edfFilePath, preload=False)
    if raw.ch_names[0].split('-')[-1] == 'LE':
        return None, raw.ch_names

    channel_order = channel_order if channel_order is not None else ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
    useless_chs = []
    for ch in raw.ch_names:
        if ch.split(' ')[-1].split('-')[0] not in channel_order:
            useless_chs.append(ch)
            
    raw.drop_channels(useless_chs)
    raw.load_data()
    # filtering
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)
    # data collected in North America use 60Hz notch filter
    raw = raw.notch_filter(notch_freq, n_jobs=n_jobs)
    # downsampling
    raw = raw.resample(sfreq, n_jobs=n_jobs)
    eegData = raw.get_data(units='uV')
    return eegData, raw.ch_names

def process(file, rsfreq = 200):
    # print(f'processing {file.name}')
    filename = file.split("/")[-1]
    eegData, chOrder = preprocessing_edf(file)
    print(chOrder)
    if chOrder[0].split('-')[-1] == 'LE':
        print("pass")
        return

    chOrder = [s.split(' ')[-1].split('-')[0] for s in chOrder]
    eegData = eegData[:, :-10*rsfreq]
    time_length = 2000

    for i in range(eegData.shape[1] // time_length):
        save_path = os.path.join(dst_root, split, filename.split('.')[0] + "_" + str(i) + ".pkl")
        X = eegData[:, i * time_length : (i + 1) * time_length]
        # "" means no label for unsupervised pretrain
        save_eeg(X, label="", channel_order=chOrder, save_path=save_path)

# 1. get files & create dataset dirs
src_root = "/tuh_eeg/v2.0.1/edf"
dst_root = "/processed_eeg/tueg"
all_files = glob.glob(src_root+"/**/*.edf", recursive=True)

create_folder_eeg(dst_root, splits=["train", "val"])

# 2. set splits
val_files = np.random.choice(all_files, size=int(len(all_files) * 0.2), replace=False)
train_files = list(set(all_files) - set(val_files))

# 3. multi-thread processing
split = "train"
pool = Pool(n_prcess)
for _ in tqdm(pool.imap_unordered(process, train_files), total=len(train_files)):
    pass
pool.close()
pool.join()

split = "val"
pool = Pool(n_prcess)
for _ in tqdm(pool.imap_unordered(process, val_files), total=len(val_files)):
    pass
pool.close()
pool.join()
