import torch
import os
import numpy as np
import pickle
import pathlib
import pytorchvideo
from torch.utils.data import DataLoader
from pytorchvideo.data import labeled_video_dataset
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
from torchvision.transforms import Compose
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

channel_order_1020 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ", "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]

def create_folder_video(root, labels, subjects=0, splits=["train", "val", "test"]):
    if subjects > 0: 
        # set each subject as independent subset
        # dataset - subject ID - train/val/test
        for subject in range(subjects):
            for split in splits:
                for label in labels:
                    split_path = os.path.join(root, "subject{}".format(str(subject+1)), split, label)
                    if not os.path.isdir(split_path):
                        os.makedirs(split_path)
    else:
        # dataset - train/val/test
        for split in splits:
            for label in labels:
                split_path = os.path.join(root, split, label)
                if not os.path.isdir(split_path):
                    os.makedirs(split_path)

def read_eeg(save_path):
    sample = pickle.load(open(save_path, "rb"))
    eeg = sample["eeg"]
    label = sample["label"]
    return eeg, label


class Dataset_TUEG(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.length = 2000
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        eeg, label = read_eeg(os.path.join(self.root, self.files[index]))
        X = {}
        y = 0
        for channel in channel_order_1020:
            if channel in eeg.keys():
                X[channel] = torch.tensor(eeg[channel]).float()
            else:
                X[channel] = torch.zeros(2000).float()
        return X, y

def prepare_TUEG_dataloader(args):
    # set random seed
    seed = 12345
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    root = args.dataset_path

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))

    print("pretrain files:", len(train_files))

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        Dataset_TUEG(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset_TUEG(os.path.join(root, "val"),
                   val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    return train_loader, val_loader


def get_loader(root_path, image_processor, batch_size, num_workers=8, num_frames=16, pix_resize=224):
    print("read data from:", root_path)
    dataset_root_path = pathlib.Path(root_path)
    print(dataset_root_path)
    video_count_train = len(list(dataset_root_path.glob(f"train/*/*.mp4")))
    video_count_val = len(list(dataset_root_path.glob(f"val/*/*.mp4")))
    video_count_test = len(list(dataset_root_path.glob(f"test/*/*.mp4")))
    video_total = video_count_train + video_count_val + video_count_test
    print("total video:", video_total)
    all_video_file_paths = (
        list(dataset_root_path.glob(f"train/*/*.mp4"))
        + list(dataset_root_path.glob(f"val/*/*.mp4"))
        + list(dataset_root_path.glob(f"test/*/*.mp4"))
    )
    class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
    label2id = {label: i for i, label in enumerate(class_labels)}
    print(label2id)
    id2label = {i: label for label, i in label2id.items()}
    print(f"Unique classes: {list(label2id.keys())}.")
    
    mean = image_processor.image_mean
    std = image_processor.image_std
    # if "shortest_edge" in image_processor.size:
    #     height = width = image_processor.size["shortest_edge"]
    # else:
    #     height = image_processor.size["height"]
    #     width = image_processor.size["width"]
    resize_to = (pix_resize, pix_resize)
    print("resize to ", pix_resize)

    num_frames_to_sample = num_frames
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
    test_dataset = labeled_video_dataset(
        data_path=os.path.join(dataset_root_path, "test"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def generate_masked_pos(batch_size, num_patches, mask_ratio=0.5):
        # provided by deepseek
        num_masked = int(num_patches * mask_ratio)
        bool_masked_pos = torch.zeros((batch_size, num_patches), dtype=torch.bool)
        for i in range(batch_size):
            indices = torch.randperm(num_patches)
            masked_indices = indices[:num_masked]
            bool_masked_pos[i, masked_indices] = True
        return bool_masked_pos
    
    