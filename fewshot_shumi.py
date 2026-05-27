import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from transformers import VideoMAEModel, VideoMAEImageProcessor, VideoMAEConfig
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
from model import BinaryModel, BinaryTraining

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
from pytorch_lightning import seed_everything

seed_everything(42)

## load vmae model
ckpt = "/inspire/hdd/global_user/wangwei-240107010004/models/videomae-base"
hidden_size=768

pretrain_ckpt = "/inspire/hdd/global_user/wangwei-240107010004/BSR-VideoMAE/checkpoints/vmae-tu2-e9.ckpt"

batch_size = 128

# ## dataset
shumi_splits = 25
shumi_root_path = "/inspire/hdd/global_user/wangwei-240107010004/BSR/data/video_fft_abs/shumi_fewshot"

image_processor = VideoMAEImageProcessor.from_pretrained("/inspire/hdd/global_user/wangwei-240107010004/models/videomae-base")

def get_loader(root_path, splits=25, num_workers=8):
    print("read data from:", root_path)
    train_loaders = []
    val_loaders = []
    test_loaders = []
    
    for i in range(splits):
        subject = "subject{}".format(str(i+1))
        dataset_root_path = pathlib.Path(os.path.join(root_path, subject))
        print(dataset_root_path)
        video_count_train = len(list(dataset_root_path.glob(f"train/*/*.mp4")))
        video_count_val = len(list(dataset_root_path.glob(f"val/*/*.mp4")))
        video_count_test = len(list(dataset_root_path.glob(f"test/*/*.mp4")))
        video_total = video_count_train + video_count_val + video_count_test
        all_video_file_paths = (
            list(dataset_root_path.glob(f"train/*/*.mp4"))
            + list(dataset_root_path.glob(f"val/*/*.mp4"))
            + list(dataset_root_path.glob(f"test/*/*.mp4"))
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

        # num_frames_to_sample = vmae.config.num_frames
        num_frames_to_sample = 16
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
        test_dataset = labeled_video_dataset(
            data_path=os.path.join(dataset_root_path, "test"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)
    return train_loaders, val_loaders, test_loaders


train_loaders, val_loaders, test_loaders = get_loader(shumi_root_path)

n_epochs = 25

bacc_list = []
roc_auc_list = []
pr_auc_list = []

for split in range(25):
    train_loader = train_loaders[split]
    val_loader = val_loaders[split]
    test_loader = test_loaders[split]
    
    # ## train from scratch
    # config_file_path = "/inspire/hdd/global_user/wangwei-240107010004/models/videomae-base/config.json"
    # with open(config_file_path, "r") as f:
    #     import json
    #     config_dict = json.load(f)
    # config = VideoMAEConfig.from_dict(config_dict)
    # model = VideoMAEModel(config)
    
    # ## train from kinetics checkpoint
    # model = VideoMAEModel.from_pretrained(ckpt)
    
    # ## load tuab+ev checkpoint
    # model = VideoMAEModel.from_pretrained(ckpt)
    # pretrain_ckpt = "/inspire/hdd/global_user/wangwei-240107010004/BSR-VideoMAE/checkpoints/vmae-onlytuev-e9.ckpt"
    # print("load pretrain ckpt:", pretrain_ckpt)
    # pretrain_ckpt = torch.load(pretrain_ckpt)
    # weights = {k[12:]: v for k, v in pretrain_ckpt["state_dict"].items() if k.startswith("model.model.")}
    # loadresult = model.load_state_dict(weights, strict=False)
    # print("model load result:", loadresult)
    
    ## load unsupervised pretrain checkpoints
    model = VideoMAEModel.from_pretrained(ckpt)
    pretrain_ckpt = "/inspire/hdd/global_user/wangwei-240107010004/BSR-VideoMAE/checkpoints/vmae-unsp-tu2-e9-tfs.ckpt"
    print("load unspervised pretrain ckpt:", pretrain_ckpt)
    pretrain_ckpt = torch.load(pretrain_ckpt)
    weights = {k[15:]: v for k, v in pretrain_ckpt["state_dict"].items() if k.startswith("model.videomae.")}
    loadresult = model.load_state_dict(weights, strict=False)
    print("model load result:", loadresult)
    
    model_multiclass = BinaryModel(model=model)
    lightning_model = BinaryTraining(model_multiclass, lr=1e-5, epochs=n_epochs)

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        benchmark=True,
        enable_checkpointing=False,
        logger=False,
        max_epochs=n_epochs,
        # accumulate_grad_batches=4,
        # limit_train_batches=10,
        # limit_val_batches=10,
        # limit_test_batches=10
    )
    
    trainer.fit(
        lightning_model, train_dataloaders=train_loader
    )
    split_result = trainer.test(
        model=lightning_model, dataloaders=test_loader
    )[0]
    print(split_result)
    bacc_list.append(split_result["test_bacc"])
    roc_auc_list.append(split_result["test_roc_auc"])
    pr_auc_list.append(split_result["test_pr_auc"])
    
print("avg_bacc:", np.mean(bacc_list))
print("std_bacc:", np.std(bacc_list))
print("avg_roc_auc:", np.mean(roc_auc_list))
print("std_roc_auc:", np.std(roc_auc_list))
print("avg_pr_auc:", np.mean(pr_auc_list))
print("std_pr_auc:", np.std(pr_auc_list))