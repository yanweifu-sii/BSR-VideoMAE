import pickle
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample
import cv2
import glob
from model import BSRender
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils import read_eeg, create_folder_video

class EEGRender:
    def __init__(self, model, do_clahe, system, clahe_size=32, device="cuda") -> None:
        self.model = model.to(device)
        self.do_clahe = do_clahe
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(clahe_size, clahe_size))
        self.system = system
        self.device = device
        
    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    
    def clahe_process(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l_enhanced = self.clahe.apply(l)

        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
    
    def render(self, eeg):
        video = self.model(eeg, system=self.system)[0].permute(1, 2, 3, 0).cpu().detach().numpy()
        if not self.do_clahe:
            video = self.normalization(video)*255
        video = video.astype(np.uint8)
        return video
    
    def save_video(self, path, video, frame_size=(160, 160), fps=2):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        for frame in video:
            if self.do_clahe:
                frame = self.clahe_process(frame)
            video_writer.write(frame)
        video_writer.release()
        
    def process(self, eeg, save_path, frame_size=(160, 160), fps=2):
        video = self.render(eeg)
        self.save_video(save_path, video, frame_size, fps)

device = "cuda"
n_process = 15
renderer_ckpt = "/checkpoints/BSR_fft_abs_TUEG_RS2240000.ckpt"

# TUAB
video_root = "/video_fft_abs/seed_fewshot_clahe" # rs2240000 := renderer pretrain step=2240000
eeg_path_template = "/eeg/seed_fewshot/*/*/*.pkl"
frame_size=(288, 288)
system="1010"
labels = ["0", "1", "2"]
do_clahe = True
clahe_size = 32

create_folder_video(video_root, subjects=15, labels=labels, splits=["train", "val", "test"])

def process(eeg_list):
    model = BSRender().to(device)
    model.load_eeg_pretrain(renderer_ckpt)
    renderer = EEGRender(model, do_clahe=do_clahe, system=system, clahe_size=clahe_size)
    for eeg_path in tqdm(eeg_list, miniters=1000):
        subject = eeg_path.split("/")[-3]
        subset = eeg_path.split("/")[-2]
        eeg_name = eeg_path.split("/")[-1]
        video_name = eeg_name[:-3]+"mp4"
        eeg, label = read_eeg(eeg_path)
        video_path = os.path.join(video_root, subject, subset, str(label), video_name)
        eeg = {i:torch.tensor(eeg[i]).float().unsqueeze(0).to(device) for i in eeg.keys()}
        renderer.process(eeg, save_path=video_path, frame_size=frame_size)

eeg_list = glob.glob(eeg_path_template)
# process(eeg_list[:2])
k = len(eeg_list)//n_process if len(eeg_list)%n_process==0 else len(eeg_list)//n_process+1
eeg_sub_list = [eeg_list[i*k:(i+1)*k] for i in range(n_process)]
print(len(eeg_list))
print(sum([len(i) for i in eeg_sub_list]))
assert len(eeg_list) == sum([len(i) for i in eeg_sub_list])

threadPool = ThreadPoolExecutor(max_workers=n_process)
for sub_list in eeg_sub_list:
    threadPool.submit(process, sub_list)
threadPool.shutdown(wait=True)
