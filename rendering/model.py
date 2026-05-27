import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

channel_order_1020 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]
channel_order_1010 = ["FP1", "FPZ", "FP2", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "O1", "OZ", "O2"]


pix_order_1020 = [
    ["",  "FP1",  "", "FP2",   ""], 
    ["F7", "F3", "FZ", "F4", "F8"], 
    ["T3", "C3", "CZ", "C4", "T4"], 
    ["T5", "P3", "PZ", "P4", "T6"], 
    ["",   "O1",  "",  "O2",   ""]
]

pix_order_1010 = [
    ["",    "",    "FP1",    "", "FPZ",    "", "FP2",    "",    ""],
    ["AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8"],
    ["F7",  "F5",  "F3",  "F1",  "FZ",  "F2",  "F4",  "F6",   "F8"],
    ["FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"],
    ["T7",  "C5",  "C3",  "C1",  "CZ",  "C2",  "C4",  "C6",   "T8"],
    ["TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"],
    ["P7",  "P5",  "P3",  "P1",  "PZ",  "P2",  "P4",  "P6",   "P8"],
    ["PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8"],
    ["",    "",    "O1",    "",  "OZ",  "",    "O2",     "",    ""],
]

pix_order_1020_1010 = [
    ["",    "",    "FP1",    "", "FPZ",    "", "FP2",    "",    ""],
    ["AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8"],
    ["F7",  "F5",  "F3",  "F1",  "FZ",  "F2",  "F4",  "F6",   "F8"],
    ["FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"],
    ["T3",  "C5",  "C3",  "C1",  "CZ",  "C2",  "C4",  "C6",   "T4"],
    ["TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"],
    ["T5",  "P5",  "P3",  "P1",  "PZ",  "P2",  "P4",  "P6",   "T6"],
    ["PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8"],
    ["",    "",    "O1",    "",  "OZ",  "",    "O2",     "",    ""],
]


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


def init_eeg2pix(n, fft_channels, pix_channels=3):
    if n==5:
        return nn.Sequential(
            Up(fft_channels, fft_channels//2), # 101 -> 50
            Up(fft_channels//2, fft_channels//4), # 50 -> 25
            Up(fft_channels//4, fft_channels//8), # 25 -> 12
            Up(fft_channels//8, fft_channels//16), # 12 -> 6
            Up(fft_channels//16, pix_channels), # 6 -> 3
        )
    elif n==4:
        return nn.Sequential(
            Up(fft_channels, fft_channels//2), # 101 -> 50
            Up(fft_channels//2, fft_channels//4), # 50 -> 25
            Up(fft_channels//4, fft_channels//8), # 25 -> 12
            Up(fft_channels//8, pix_channels), # 12 -> 3
        )
    elif n==3:
        return nn.Sequential(
            Up(fft_channels, fft_channels//2), # 101 -> 50
            Up(fft_channels//2, fft_channels//8), # 50 -> 12
            Up(fft_channels//8, pix_channels), # 12 -> 3
        )
        
def init_pix2eeg(n, fft_channels, pix_channels=3):
    if n==5:
        return nn.Sequential(
            Down(pix_channels, fft_channels//16), # 3 -> 6
            Down(fft_channels//16, fft_channels//8), # 6 -> 12
            Down(fft_channels//8, fft_channels//4), # 12 -> 24
            Down(fft_channels//4, fft_channels//2), # 24 -> 48
            Down(fft_channels//2, fft_channels), # 48 -> 101
        )
    elif n==4:
        return nn.Sequential(
            Down(pix_channels, fft_channels//8), # 3 -> 12
            Down(fft_channels//8, fft_channels//4), # 12 -> 24
            Down(fft_channels//4, fft_channels//2), # 24 -> 48
            Down(fft_channels//2, fft_channels), # 48 -> 101
        )
    elif n==3:
        return nn.Sequential(
            Down(pix_channels, fft_channels//8), # 3 -> 12
            Down(fft_channels//8, fft_channels//2), # 12 -> 48
            Down(fft_channels//2, fft_channels), # 48 -> 101
        )
    
    
class BSReconstruct(nn.Module):
    def __init__(self, n=5, n_fft=200, hop_length=100, pix_channels=3):
        super(BSReconstruct, self).__init__()
        self.n_fft = n_fft
        self.fft_channels = n_fft//2+1
        self.hop_length = hop_length
        self.pix_channels = pix_channels
        
        self.eeg2pix = init_eeg2pix(n, self.fft_channels, self.pix_channels)
        self.pix2eeg = init_pix2eeg(n, self.fft_channels, self.pix_channels)
        
    def stft(self, eeg, channel):
        return torch.abs(torch.stft(eeg[channel], n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True).unsqueeze(0).unsqueeze(0))
        
    def spatialization(self, eeg, system="1020"):
        if system=="1020":
            pix_order = pix_order_1020
            size = 5
        elif system=="1010":
            pix_order = pix_order_1010
            size = 9
        ## put other systems here ... ##
        
        ## put other systems here ... ##
        else:
            raise NotImplementedError
        
        eeg_list = []
        for i in range(size):
            tmp_list = []
            for j in range(size):
                channel = pix_order[i][j]
                if channel in eeg.keys():
                    tmp_list.append(self.stft(eeg, channel))
                else:
                    # if this position does not match any channel in eeg data, fill all 0 instead
                    tmp_tensor = self.stft(eeg, list(eeg.keys())[0]).clone()
                    tmp_tensor[:] = 0
                    tmp_list.append(tmp_tensor)
            eeg_list.append(torch.cat(tmp_list, dim=1))
            
        return torch.cat(eeg_list, dim=0).permute(2, 4, 0, 1, 3)

    def forward(self, x, system="1020"):
        # do stft for each channel
        eeg = self.spatialization(x, system=system)
        n_frames = eeg.shape[1]
        frames = [self.eeg2pix(eeg[:, i, :, : ,:].permute(0, 3, 1, 2)) for i in range(n_frames)]
        eeg_reconstruct = [self.pix2eeg(frames[i]).unsqueeze(1) for i in range(n_frames)]
        eeg_reconstruct = torch.cat(eeg_reconstruct, dim=1).permute(0, 1, 3, 4, 2)
        return eeg, eeg_reconstruct


class BSRender(nn.Module):
    def __init__(self, n_fft=200, hop_length=100, pix_channels=3):
        super(BSRender, self).__init__()
        self.n_fft = n_fft
        self.fft_channels = n_fft//2+1
        self.hop_length = hop_length
        self.pix_channels = pix_channels
        
        self.eeg2pix = nn.Sequential(
            Up(self.fft_channels, self.fft_channels//2), # 101 -> 50
            Up(self.fft_channels//2, self.fft_channels//4), # 50 -> 25
            Up(self.fft_channels//4, self.fft_channels//8), # 25 -> 12
            Up(self.fft_channels//8, self.fft_channels//16), # 12 -> 6
            Up(self.fft_channels//16, self.pix_channels), # 6 -> 3
        )
        
    def stft(self, eeg, channel):
        return torch.abs(torch.stft(eeg[channel], n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True).unsqueeze(0).unsqueeze(0))
        
    def spatialization(self, eeg, system="1020"):
        if system=="1020":
            pix_order = pix_order_1020
            size = 5
        elif system=="1010":
            pix_order = pix_order_1010
            size = 9
        elif system=="1020_1010":
            pix_order = pix_order_1020_1010
            size = 9
        ## put other systems here ... ##
        
        ## put other systems here ... ##
        else:
            raise NotImplementedError
        
        eeg_list = []
        for i in range(size):
            tmp_list = []
            for j in range(size):
                channel = pix_order[i][j]
                if channel in eeg.keys():
                    tmp_list.append(self.stft(eeg, channel))
                else:
                    # if this position does not match any channel in eeg data, fill all 0 instead
                    tmp_tensor = self.stft(eeg, list(eeg.keys())[0]).clone()
                    tmp_tensor[:] = 0
                    tmp_list.append(tmp_tensor)
            eeg_list.append(torch.cat(tmp_list, dim=1))
            
        return torch.cat(eeg_list, dim=0).permute(2, 4, 0, 1, 3)
    
    def load_eeg_pretrain(self, pretrain_eeg_path):
        print("load pretrained eeg2vid pretrained checkpoint:", pretrain_eeg_path)
        checkpoint = torch.load(pretrain_eeg_path)['state_dict']
        pretrain_eeg_dict = {
            k[len('model.eeg2pix.'):]: v 
            for k, v in checkpoint.items() 
            if k.startswith('model.eeg2pix')
        }
        self.eeg2pix.load_state_dict(pretrain_eeg_dict, strict=False)
    
    def forward(self, x, system="1020"):
        # do stft for each channel
        eeg = self.spatialization(x, system=system)
        n_frames = eeg.shape[1]
        frames = [self.eeg2pix(eeg[:, i, :, : ,:].permute(0, 3, 1, 2)).unsqueeze(1) for i in range(n_frames)]
        video = torch.cat(frames, dim=1).permute(0, 2, 1, 3, 4)
        return video

# test
if __name__ == "__main__":
    device = "cuda"
    renderer = BSRender()
    data = [pickle.load(open("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangwei-240107010004/BSR/data/eeg/tueg/val/aaaaanut_s003_t003_9.pkl", "rb")),
            pickle.load(open("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangwei-240107010004/BSR/data/eeg/tueg/val/aaaaanut_s003_t003_8.pkl", "rb"))]
    eeg = [data[i]["eeg"] for i in range(len(data))]
    eeg = {i: torch.stack([torch.tensor(eeg[j][i]).float() for j in range(len(eeg))]) for i in eeg[0].keys()}
    print(eeg)
    vid = renderer(eeg)
    # print(vid.shape)
    
    