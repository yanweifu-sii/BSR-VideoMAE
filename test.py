import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from transformers import VideoMAEModel, VideoMAEImageProcessor, TimesformerModel, VivitModel, VideoMAEConfig
import argparse
import pathlib
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities import CombinedLoader
import os
from eval_metrics import get_metrics_binary, get_metrics_multiclass
from utils import get_loader
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import MultiHeadModel, MultiTaskTraining, MultiClassModel, BinaryModel, BinaryTraining, MultiClassTraining
import warnings
from pytorch_lightning import seed_everything

# warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="The video decoding and encoding capabilities of torchvision are deprecated from version 0.22 and will be removed in version 0.24.")


tuab_root_path = "/inspire/hdd/global_user/wangwei-240107010004/BSR/data/video_fft_abs/tuab_clahe"
tuev_root_path = "/inspire/hdd/global_user/wangwei-240107010004/BSR/data/video_fft_abs/tuev_clahe"
# shumi_root_path = "/inspire/hdd/global_user/wangwei-240107010004/BSR/data/video_fft_abs/shumi@tueg_clahe32"
# seed_root_path = "/inspire/hdd/global_user/wangwei-240107010004/BSR/data/video_fft_abs/seed_clahe"
# tusl_root_path = "/inspire/hdd/global_user/wangwei-240107010004/BSR/data/video_fft_abs/tusl_clahe"
# hmc_root_path = "/inspire/hdd/global_user/wangwei-240107010004/BSR/data/video_fft_abs/hmc_clahe"
# wl_root_path = "/inspire/hdd/global_user/wangwei-240107010004/BSR/data/video_fft_abs/Workload_clahe"


def finetune(args):
    seed_everything(42)

    pix_resize = 224
    image_processor = VideoMAEImageProcessor.from_pretrained("/inspire/hdd/global_user/wangwei-240107010004/models/videomae-base")

    ckpt = "/inspire/hdd/global_user/wangwei-240107010004/models/videomae-base"
    model = VideoMAEModel.from_pretrained(ckpt)
    hidden_size=768

    pretrain_ckpt = "/inspire/hdd/global_user/wangwei-240107010004/BSR-VideoMAE/checkpoints/vmae-onlytu2-e9.ckpt"
    print("load pretrain ckpt:", pretrain_ckpt)
    pretrain_ckpt = torch.load(pretrain_ckpt)
    weights = {k[12:]: v for k, v in pretrain_ckpt["state_dict"].items() if k.startswith("model.model.")}
    loadresult = model.load_state_dict(weights, strict=False)
    print("model load result:", loadresult)
    
    
    
    trainer = pl.Trainer(
        default_root_dir="/inspire/hdd/global_user/wangwei-240107010004/BSR-Video-MH/logs/six-task-custom",
        # callbacks=[EarlyStopping(monitor="val_bacc", patience=args.patience, mode="max")],
        accelerator="gpu",
        # strategy=DDPStrategy(find_unused_parameters=False),
        # benchmark=True,
        # enable_checkpointing=True,
        # max_epochs=args.epochs,
        # limit_train_batches=10,
        # accumulate_grad_batches=8,
        # log_every_n_steps=20,
        # limit_val_batches=10,
        # limit_test_batches=5
    )
    
    batch_size = 128
    tuab_train, tuab_val, tuab_test = get_loader(tuab_root_path, image_processor, batch_size=batch_size, pix_resize=pix_resize)
    tuev_train, tuev_val, tuev_test = get_loader(tuev_root_path, image_processor, batch_size=batch_size, pix_resize=pix_resize)

    head = nn.Linear(hidden_size, 2)
    weights = {k[20:]: v for k, v in pretrain_ckpt["state_dict"].items() if k.startswith("model.ab_classifier.")}
    loadresult = head.load_state_dict(weights, strict=False)
    print("tuab head load result:", loadresult)
    model_binary = BinaryModel(model=model, head=head)
    lightning_model = BinaryTraining(model_binary, lr=args.lr, epochs=args.epochs)

    result = trainer.test(
        model=lightning_model, dataloaders=tuab_test
    )[0]
    print("tuab result:")
    print(result)
    
    head = nn.Linear(hidden_size, 6)
    weights = {k[20:]: v for k, v in pretrain_ckpt["state_dict"].items() if k.startswith("model.ev_classifier.")}
    loadresult = head.load_state_dict(weights, strict=False)
    print("tuev head load result:", loadresult)
    model_multiclass = MultiClassModel(model=model, n_cls=6, head=head)
    lightning_model = MultiClassTraining(model_multiclass, n_cls=6, lr=args.lr, epochs=args.epochs)
    result = trainer.test(
        model=lightning_model, dataloaders=tuev_test
    )[0]
    print("tuev result:")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--dataset", type=str, default="shumi", help="finetune dataset")
    parser.add_argument("--dataset_path", type=str, default="/inspire/hdd/global_user/wangwei-240107010004/BSR/data/video_fft_abs/shumi@shumi_clahe36", help="location of the dataset")
    parser.add_argument("--model", type=str, default="videomae-base", help="finetune model")
    parser.add_argument("--log_name", type=str, default="shumi-vmae-base", help="log name")
    parser.add_argument("--patience", type=int, default=10, help="early stop bacc callback patience")
    parser.add_argument("--pretrain_ckpt", type=str, default=None, help="pretrain checkpoints")
    parser.add_argument("--drop_rate", type=float, default=0, help="dropout rate for specific model")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    args = parser.parse_args()
    print(args)
    finetune(args)

