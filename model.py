import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from eval_metrics import get_metrics_binary, get_metrics_multiclass
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, balanced_accuracy_score, f1_score, average_precision_score

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
    
    def de_spatialization(self, x, eeg, system):
        if system=="1020":
            pix_order = pix_order_1020
            size = 5
        elif system=="1010":
            pix_order = pix_order_1010
            size = 9
        elif system=="1020_1010":
            pix_order = pix_order_1020_1010
            size = 9

        f_reconstruct = {}

        for i in range(size):
            for j in range(size):
                channel = pix_order[i][j]
                if channel in x.keys():
                    f_reconstruct[channel] = eeg[:, :, i, j, :]
                    
        return f_reconstruct

    def check(self, x, system="1020"):
        # do stft for each channel
        eeg = self.spatialization(x, system=system)
        n_frames = eeg.shape[1]
        frames = [self.eeg2pix(eeg[:, i, :, : ,:].permute(0, 3, 1, 2)) for i in range(n_frames)]
        eeg_reconstruct = [self.pix2eeg(frames[i]).unsqueeze(1) for i in range(n_frames)]
        eeg_reconstruct = torch.cat(eeg_reconstruct, dim=1).permute(0, 1, 3, 4, 2)
        f = self.de_spatialization(x, eeg, system)
        f_reconstruct = self.de_spatialization(x, eeg_reconstruct, system)
        return f, f_reconstruct
        

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
    
class MultiHeadModel(nn.Module):
    def __init__(self, model, hidden_size=768):
        super().__init__()
        self.model = model

        # abnormal detection
        task_ab_n_cls = 2
        self.ab_classifier = nn.Linear(hidden_size, task_ab_n_cls)
        
        # event clasification
        task_ev_n_cls = 6
        self.ev_classifier = nn.Linear(hidden_size, task_ev_n_cls)
        
        # task_mi_n_cls = 4
        # self.mi_classifier = nn.Linear(hidden_size, task_mi_n_cls)
        
        # # emotion classification
        # task_em_n_cls = 3
        # self.em_classifier = nn.Linear(hidden_size, task_em_n_cls)
        
        # # slow event
        # task_sl_n_cls = 3
        # self.sl_classifier = nn.Linear(hidden_size, task_sl_n_cls)
        
        # # sleep stage
        # task_ss_n_cls = 5
        # self.ss_classifier = nn.Linear(hidden_size, task_ss_n_cls)
        
        # # work load
        # task_wl_n_cls = 2
        # self.wl_classifier = nn.Linear(hidden_size, task_wl_n_cls)
        

    def forward(self, inputs):
        outputs = self.model(**inputs)
        sequence_output = outputs[0]
        
        ab_logits = self.ab_classifier(sequence_output[:, 0])
        ev_logits = self.ev_classifier(sequence_output[:, 0])
        # mi_logits = self.mi_classifier(sequence_output[:, 0])
        # em_logits = self.em_classifier(sequence_output[:, 0])
        # sl_logits = self.sl_classifier(sequence_output[:, 0])
        # ss_logits = self.ss_classifier(sequence_output[:, 0])
        # wl_logits = self.wl_classifier(sequence_output[:, 0])

        return ab_logits, ev_logits
    
    
class BinaryModel(nn.Module):
    def __init__(self, model, hidden_size=768, head=None):
        super().__init__()
        self.model = model
        n_cls = 2
        if head is None:
            self.classifier = nn.Linear(hidden_size, n_cls)
        else:
            self.classifier = head

    def forward(self, inputs):
        outputs = self.model(**inputs)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0])
        return logits
    
    
class MultiClassModel(nn.Module):
    def __init__(self, model, hidden_size=768, n_cls=6, head=None):
        super().__init__()
        self.model = model
        self.n_cls = n_cls
        if head is None:
            self.classifier = nn.Linear(hidden_size, n_cls)
        else:
            self.classifier = head
        
    def forward(self, inputs):
        outputs = self.model(**inputs)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0])
        return logits


class MultiTaskTraining(pl.LightningModule):
    def __init__(self, model, lr, weight_decay=1e-4, epochs=20, weight_dict=None):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        
        if weight_dict is None:
            self.ab_weight = 0.6
            self.ev_weight = 0.4
        else:
            self.ab_weight = weight_dict["ab"]
            self.ev_weight = weight_dict["ev"]
            
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        batch_ab = batch['ab']
        batch_ev = batch['ev']
        
        if batch_ab is not None:
            pixel_values = batch_ab["video"].permute(0, 2, 1, 3, 4)
            inputs = {"pixel_values": pixel_values.to(self.device)}
            y = batch_ab["label"]
            ab_logits, ev_logits = self.model(inputs)
            ab_loss = self.loss(ab_logits, y)
            self.log("train_ab_loss", ab_loss, on_epoch=True, sync_dist=True)
        else:
            ab_loss = torch.tensor(0.0, device=self.device)
                
        if batch_ev is not None:
            pixel_values = batch_ev["video"].permute(0, 2, 1, 3, 4)
            inputs = {"pixel_values": pixel_values.to(self.device)}
            y = batch_ev["label"]
            ab_logits, ev_logits = self.model(inputs)
            ev_loss = self.loss(ev_logits, y)
            self.log("train_ev_loss", ev_loss, on_epoch=True, sync_dist=True)
        else:
            ev_loss = torch.tensor(0.0, device=self.device)

        loss = ab_loss*self.ab_weight + ev_loss*self.ev_weight 
        self.log("train_total_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_ab = batch['ab']
        batch_ev = batch['ev']
        
        if batch_ab is not None:
            pixel_values = batch_ab["video"].permute(0, 2, 1, 3, 4)
            inputs = {"pixel_values": pixel_values.to(self.device)}
            y = batch_ab["label"]
            with torch.no_grad():
                step_gt_ab = y.cpu().numpy().copy()
                ab_logits, ev_logits = self.model(inputs)
                ab_loss = self.loss(ab_logits, y)
                self.log("val_ab_loss", ab_loss, on_epoch=True, sync_dist=True, logger=True)
                
        if batch_ev is not None:
            pixel_values = batch_ev["video"].permute(0, 2, 1, 3, 4)
            inputs = {"pixel_values": pixel_values.to(self.device)}
            y = batch_ev["label"]
            with torch.no_grad():
                ab_logits, ev_logits = self.model(inputs)        
                ev_loss = self.loss(ev_logits, y)
                self.log("val_ev_loss", ev_loss, on_epoch=True, sync_dist=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer,
                              T_max=self.epochs,
                              eta_min=1e-7)
        
        return [optimizer], [scheduler]


class BinaryTraining(pl.LightningModule):
    def __init__(self, model, lr, weight_decay=1e-4, epochs=20):
        super().__init__()
        self.model = model
        self.binary_metrics = get_metrics_binary
        self.multiclass_metrics = get_metrics_multiclass
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        
        self.test_preds = []
        self.test_labels = []
        
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values = batch["video"].permute(0, 2, 1, 3, 4)
        inputs = {"pixel_values": pixel_values.to(self.device)}
        y = batch["label"]
        logits = self.model(inputs)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values = batch["video"].permute(0, 2, 1, 3, 4)
        inputs = {"pixel_values": pixel_values.to(self.device)}
        y = batch["label"]
        with torch.no_grad():
            step_gt = y.cpu().numpy().copy()
            logits = self.model(inputs)
            step_result = logits.softmax(dim=1)[:, 1].cpu()
            # result = self.binary_metrics(step_gt_ab, step_result_ab)
            # self.log("val_ab_bacc", result["balanced_accuracy"], sync_dist=True, on_epoch=True, logger=True)
            ab_loss = self.loss(logits, y)
            self.log("val_loss", ab_loss, on_epoch=True, sync_dist=True, logger=True)
            
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values = batch["video"].permute(0, 2, 1, 3, 4)
        inputs = {"pixel_values": pixel_values.to(self.device)}
        y = batch["label"]
        with torch.no_grad():
            step_gt = y.cpu()
            logits = self.model(inputs)
            step_result = logits.softmax(dim=1)[:, 1].cpu()
            # result = self.binary_metrics(step_gt_ab, step_result_ab)
            # self.log("val_ab_bacc", result["balanced_accuracy"], sync_dist=True, on_epoch=True, logger=True)
            self.test_preds.append(step_result)
            self.test_labels.append(y)

            return {'test_preds': step_result, 'test_labels': step_gt}
        
    def on_test_epoch_end(self):
        # 将所有批次的预测和标签拼接起来
        all_preds = torch.cat(self.test_preds, dim=0)
        all_labels = torch.cat(self.test_labels, dim=0)
        
        # 使用gather进行分布式同步
        # sync_dist=True 会自动处理多GPU/多节点的同步
        gathered_preds = self.all_gather(all_preds)
        gathered_labels = self.all_gather(all_labels)
        
        # 如果是主进程，计算指标
        if self.trainer.is_global_zero:
            # 将gathered数据转换为numpy数组
            # gathered_preds 的形状为 [world_size, num_samples]
            # 我们需要将其展平
            all_preds_np = gathered_preds.cpu().numpy().flatten()
            all_labels_np = gathered_labels.cpu().numpy().flatten()
            
            # 使用sklearn计算全局指标
            # print(all_labels_np, all_preds_np)
            threshold = 0.5
            all_probs_np = all_preds_np.copy()
            all_preds_np[all_preds_np >= threshold] = 1
            all_preds_np[all_preds_np < threshold] = 0
            test_bacc = balanced_accuracy_score(all_labels_np, all_preds_np)
            test_pr_auc = average_precision_score(all_labels_np, all_probs_np)
            test_roc_auc = roc_auc_score(all_labels_np, all_probs_np)
            
            # 记录指标
            self.log('test_bacc', test_bacc, on_epoch=True, sync_dist=True, logger=True)
            self.log('test_pr_auc', test_pr_auc, on_epoch=True, sync_dist=True, logger=True)
            self.log('test_roc_auc', test_roc_auc, on_epoch=True, sync_dist=True, logger=True)
            
        # 清空列表
        self.test_preds.clear()
        self.test_labels.clear()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer,
                              T_max=self.epochs,
                              eta_min=1e-7)
        
        return [optimizer], [scheduler]


class MultiClassTraining(pl.LightningModule):
    def __init__(self, model, lr, n_cls, weight_decay=1e-5, epochs=20):
        super().__init__()
        self.model = model
        self.metrics = get_metrics_multiclass
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.n_cls = n_cls
        
        self.test_preds = []
        self.test_labels = []
        
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values = batch["video"].permute(0, 2, 1, 3, 4)
        inputs = {"pixel_values": pixel_values.to(self.device)}
        y = batch["label"]
        logits = self.model(inputs)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values = batch["video"].permute(0, 2, 1, 3, 4)
        inputs = {"pixel_values": pixel_values.to(self.device)}
        y = batch["label"]
        with torch.no_grad():
            step_gt = y.cpu().numpy().copy()
            logits = self.model(inputs)
            # step_result = logits.softmax(dim=1)[:, 1].cpu()
            # result = self.binary_metrics(step_gt_ab, step_result_ab)
            # self.log("val_ab_bacc", result["balanced_accuracy"], sync_dist=True, on_epoch=True, logger=True)
            loss = self.loss(logits, y)
            self.log("val_loss", loss, on_epoch=True, sync_dist=True, logger=True)
            
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values = batch["video"].permute(0, 2, 1, 3, 4)
        inputs = {"pixel_values": pixel_values.to(self.device)}
        y = batch["label"]
        with torch.no_grad():
            step_gt = y.cpu()
            logits = self.model(inputs)
            step_result = logits.softmax(dim=1).cpu()
            step_result = torch.argmax(step_result, dim=-1)
            # result = self.binary_metrics(step_gt_ab, step_result_ab)
            # self.log("val_ab_bacc", result["balanced_accuracy"], sync_dist=True, on_epoch=True, logger=True)
            self.test_preds.append(step_result)
            self.test_labels.append(y)

            return {'test_preds': step_result, 'test_labels': step_gt}
        
    def on_test_epoch_end(self):
        # 将所有批次的预测和标签拼接起来
        all_preds = torch.cat(self.test_preds, dim=0)
        all_labels = torch.cat(self.test_labels, dim=0)
        
        # 使用gather进行分布式同步
        # sync_dist=True 会自动处理多GPU/多节点的同步
        gathered_preds = self.all_gather(all_preds)
        gathered_labels = self.all_gather(all_labels)
        
        # 如果是主进程，计算指标
        if self.trainer.is_global_zero:
            # 将gathered数据转换为numpy数组
            # gathered_preds 的形状为 [world_size, num_samples]
            # 我们需要将其展平
            all_preds_np = gathered_preds.cpu().numpy().flatten()
            all_labels_np = gathered_labels.cpu().numpy().flatten()
            
            # 使用sklearn计算全局指标
            test_bacc = balanced_accuracy_score(all_labels_np, all_preds_np)
            test_cohen = cohen_kappa_score(all_labels_np, all_preds_np, labels=range(self.n_cls))
            test_f1 = f1_score(all_labels_np, all_preds_np, average="weighted", labels=range(self.n_cls))
            
            # 记录指标
            self.log('test_bacc', test_bacc, on_epoch=True, sync_dist=True, logger=True)
            self.log('test_cohen', test_cohen, on_epoch=True, sync_dist=True, logger=True)
            self.log('test_f1', test_f1, on_epoch=True, sync_dist=True, logger=True)
            
        # 清空列表
        self.test_preds.clear()
        self.test_labels.clear()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer,
                              T_max=self.epochs,
                              eta_min=1e-7)
        
        return [optimizer], [scheduler]


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
    
    