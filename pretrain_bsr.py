import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os
import argparse
from pytorch_lightning.strategies import DDPStrategy

from model import BSReconstruct
from utils import prepare_TUEG_dataloader


class BSR_pretrain(pl.LightningModule):
    def __init__(self, model, args, save_path):
        super().__init__()
        self.model = model
        self.criterion = nn.L1Loss()
        self.save_path = save_path
        
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        # store the checkpoint every 20000 steps
        if self.global_step % 20000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )
            
        X, _ = batch
        eeg, eeg_reconstruct = self.model(X)
        loss = self.criterion(eeg, eeg_reconstruct)
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        X, _ = batch
        with torch.no_grad():
            eeg, eeg_reconstruct = self.model(X)
            loss = self.criterion(eeg, eeg_reconstruct)
            self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-5,
        )
        return [optimizer]


def pretrain(args):
    train_loader, val_loader = prepare_TUEG_dataloader(args)
    
    # define the version
    N_version = (
        len(os.listdir("/inspire/hdd/global_user/wangwei-240107010004/BSR-VideoMAE/logs/pretrain_bsr/lightning_logs"))
    )
    print("N_version:", N_version)
    # define the model
    save_path = f"/inspire/hdd/global_user/wangwei-240107010004/BSR-VideoMAE/checkpoints/pretrain_bsr/{N_version}-unsupervised"
    
    pretrain_model = BSReconstruct(n=args.num_layers, n_fft=args.sampling_rate, hop_length=args.hop_length)
    model = BSR_pretrain(pretrain_model, args, save_path)

    trainer = pl.Trainer(
        default_root_dir="/inspire/hdd/global_user/wangwei-240107010004/BSR-VideoMAE/logs/pretrain_bsr/lightning_logs",
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        benchmark=True,
        enable_checkpointing=True,
        max_epochs=args.epochs,
    )

    # train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="number of workers")
    parser.add_argument("--dataset_path", type=str, default="/inspire/hdd/project/brainsignals/public/BSR-Data/processed_eeg/tueg", help="location of the dataset")
    parser.add_argument("--sampling_rate", type=int, default=200, help="sampling rate (r)")
    parser.add_argument("--hop_length", type=int, default=100, help="token hop length (t - p)")
    parser.add_argument("--num_layers", type=int, default=5, help="number of BSR layers")
    args = parser.parse_args()
    print(args)
    pretrain(args)
    