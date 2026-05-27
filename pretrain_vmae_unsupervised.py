import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor, VideoMAEConfig
from eval_metrics import get_metrics_binary, get_metrics_multiclass
import pathlib
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities import CombinedLoader

from pytorchvideo.data import labeled_video_dataset
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
from torchvision.transforms import Compose
import os
import pytorchvideo
from torch.utils.data import DataLoader

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    Resize,
)
from utils import generate_masked_pos


# set hyper-paras
max_epochs = 10
log_root_dir="/inspire/hdd/global_user/wangwei-240107010004/BSR-VideoMAE/logs/unsp_tuab-ev"

# # vmae-small model
# model_name = "videomae-small"
# model_ckpt = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangwei-240107010004/models/models--MCG-NJU--videomae-small-finetuned-kinetics/snapshots/1bf2b82c809609f5df1ea7648aee577e2e486b63"
# batch_size = 80
# device = "cuda"

# vmae-base model
# model_name = "videomae-base"

# model_ckpt = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangwei-240107010004/models/models--MCG-NJU--videomae-base-finetuned-kinetics"
batch_size = 64
device = "cuda"

# set dataset
## dataset
dataset_name = "tuab"
dataset_root_path = "/inspire/hdd/project/roboticarm/wangwei-240107010004/data/tuab-ev"

# set random seed
seed = 4523

class VMAE_pretrain(pl.LightningModule):
    def __init__(self, model, save_path, batch_size):
        super().__init__()
        self.model = model
        self.save_path = save_path
        self.batch_size = batch_size
        self.num_patches_per_frame = (self.model.config.image_size // self.model.config.patch_size) ** 2
        self.seq_length = (self.model.config.num_frames // self.model.config.tubelet_size) * self.num_patches_per_frame
        
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        # store the checkpoint every 2000 steps
        if self.global_step % 2000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )
            
        pixel_values = batch["video"].permute(0, 2, 1, 3, 4)
        bool_masked_pos = generate_masked_pos(batch_size=pixel_values.shape[0], num_patches=self.seq_length, mask_ratio=0.9)
        outputs = self.model(pixel_values.to(device), bool_masked_pos=bool_masked_pos.to(device))
        loss = outputs.loss
        self.log("train_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            pixel_values = batch["video"].permute(0, 2, 1, 3, 4)
            bool_masked_pos = generate_masked_pos(batch_size=pixel_values.shape[0], num_patches=self.seq_length, mask_ratio=0.9)
            outputs = self.model(pixel_values.to(device), bool_masked_pos=bool_masked_pos.to(device))
            loss = outputs.loss
            self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-5,
        )
        return [optimizer]


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

model_ckpt = "/inspire/hdd/global_user/wangwei-240107010004/models/videomae-base"

# # train from scratch
# config_file_path = "/inspire/hdd/global_user/wangwei-240107010004/models/videomae-base/config.json"
# with open(config_file_path, "r") as f:
#     import json
#     config_dict = json.load(f)
# config = VideoMAEConfig.from_dict(config_dict)
# vmae = VideoMAEForPreTraining(config)

# load kinetics ckpt
vmae = VideoMAEForPreTraining.from_pretrained(model_ckpt)

image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)

def get_loader(root_path, num_workers=8):
    print("read data from:", root_path)
    dataset_root_path = pathlib.Path(root_path)
    video_count_train = len(list(dataset_root_path.glob("train/*/*.mp4")))
    video_count_val = len(list(dataset_root_path.glob("val/*/*.mp4")))
    video_count_test = len(list(dataset_root_path.glob("test/*/*.mp4")))
    video_total = video_count_train + video_count_val + video_count_test
    all_video_file_paths = (
        list(dataset_root_path.glob("train/*/*.mp4"))
        + list(dataset_root_path.glob("val/*/*.mp4"))
        + list(dataset_root_path.glob("test/*/*.mp4"))
    )
    class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    print(f"Unique classes: {list(label2id.keys())}.")
    
    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    num_frames_to_sample = vmae.config.num_frames
    # num_frames_to_sample = 16
    sample_rate = 2
    fps = 2
    clip_duration = num_frames_to_sample * sample_rate / fps

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                        # RandomShortSideScale(min_size=256, max_size=320),
                        # RandomCrop(resize_to),
                        # RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    train_dataset = labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    val_dataset = labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    
    return train_loader, val_loader

train_loader, val_loader = get_loader(dataset_root_path)

# define the version
N_version = (
    len(os.listdir(os.path.join(log_root_dir, "lightning_logs")))
)
print("N_version:", N_version)
# define the model
save_path = f"/inspire/hdd/global_user/wangwei-240107010004/BSR-VideoMAE/checkpoints/unsp_tuab-ev/{N_version}-unsupervised"

# vmae = vmae.to(device)
model = VMAE_pretrain(model=vmae.to(device), save_path=save_path, batch_size=batch_size)
trainer = pl.Trainer(
    default_root_dir=log_root_dir,
    accelerator="gpu",
    strategy=DDPStrategy(find_unused_parameters=False),
    benchmark=True,
    enable_checkpointing=True,
    max_epochs=max_epochs,
)

# train the model
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
# seq_length = (model.config.num_frames // model.config.tubelet_size) * num_patches_per_frame

# print(bool_masked_pos)
# inputs = {"pixel_values": pixel_values.to(device), "bool_masked_pos":bool_masked_pos.to(device)}
# outputs = model(pixel_values.to(device), bool_masked_pos=bool_masked_pos.to(device))
# print(outputs.logits.shape)
# loss = outputs.loss
# print(loss)
