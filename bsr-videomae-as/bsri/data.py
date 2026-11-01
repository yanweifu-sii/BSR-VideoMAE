"""Direct EEG pickle datasets; no intermediate MP4 files are created."""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .model import CHANNELS_1010, CHANNELS_1020, CHANNELS_1020_1010, SYSTEM_CHANNELS, canonical_channel


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _normalize_label(label: Any) -> str:
    if isinstance(label, np.generic):
        label = label.item()
    return str(label)


def unpack_sample(sample: Any) -> tuple[Any, Any, list[str] | None]:
    if not isinstance(sample, dict):
        raise TypeError("Each pickle must contain a mapping")
    if "eeg" in sample:
        return sample["eeg"], sample.get("label", ""), sample.get("channel_order")
    if "X" in sample:
        names = sample.get("channel_order", sample.get("ch_names", sample.get("channels")))
        return sample["X"], sample.get("y", sample.get("label", "")), names
    raise KeyError("Pickle must use {'eeg','label'} or {'X','y'} format")


def find_split_files(root: str | Path, split: str) -> list[Path]:
    directory = Path(root) / split
    if not directory.is_dir():
        return []
    return sorted((*directory.rglob("*.pkl"), *directory.rglob("*.pickle")))


def infer_label_mapping(paths: list[Path]) -> dict[str, int]:
    labels = sorted({_normalize_label(unpack_sample(_read_pickle(path))[1]) for path in paths})
    labels = [label for label in labels if label != ""]
    return {label: index for index, label in enumerate(labels)}


def parse_channel_names(value: str | None) -> list[str] | None:
    if not value:
        return None
    path = Path(value)
    if path.is_file():
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, list):
            raise ValueError("Channel-name JSON must be a list")
        return [canonical_channel(name) for name in loaded]
    return [canonical_channel(name) for name in value.split(",") if name.strip()]


class EEGPickleDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        paths: list[Path],
        system: str = "1010",
        label_to_id: dict[str, int] | None = None,
        segment_length: int = 2000,
        training: bool = False,
        source_channel_names: list[str] | None = None,
        fill: str = "zero",
        fill_hop: int = 128,
    ) -> None:
        if fill not in ("zero", "repeat"):
            raise ValueError("fill must be 'zero' or 'repeat'")
        if not paths:
            raise ValueError("Dataset contains no pickle files")
        self.paths = paths
        self.system = system
        self.target_names = SYSTEM_CHANNELS[system]
        self.label_to_id = label_to_id or {}
        self.segment_length = segment_length
        self.training = training
        self.source_channel_names = source_channel_names
        self.fill = fill
        self.fill_hop = int(fill_hop)

    def __len__(self) -> int:
        return len(self.paths)

    def _as_mapping(self, eeg: Any, names: list[str] | None) -> dict[str, Tensor]:
        if isinstance(eeg, dict):
            return {canonical_channel(key): torch.as_tensor(value, dtype=torch.float32).flatten() for key, value in eeg.items()}
        array = torch.as_tensor(eeg, dtype=torch.float32)
        if array.ndim != 2:
            raise ValueError(f"Dense EEG must be 2-D, received {tuple(array.shape)}")
        supplied = names or self.source_channel_names
        known_counts = {
            len(CHANNELS_1020): CHANNELS_1020,
            len(CHANNELS_1010): CHANNELS_1010,
            len(CHANNELS_1020_1010): CHANNELS_1020_1010,
        }
        if supplied is None:
            supplied = list(known_counts.get(array.shape[0], ()))
        if not supplied and array.shape[1] in known_counts:
            array = array.T
            supplied = list(known_counts[array.shape[0]])
        if not supplied:
            raise ValueError(
                f"Cannot infer names for a {tuple(array.shape)} EEG array; pass --channel-names"
            )
        if array.shape[0] != len(supplied) and array.shape[1] == len(supplied):
            array = array.T
        if array.shape[0] != len(supplied):
            raise ValueError(f"EEG has {array.shape[0]} rows but {len(supplied)} channel names")
        return {canonical_channel(name): array[index].flatten() for index, name in enumerate(supplied)}

    def _crop_or_pad(self, signal: Tensor) -> Tensor:
        length = signal.numel()
        if length >= self.segment_length:
            maximum = length - self.segment_length
            start = random.randint(0, maximum) if self.training and maximum else maximum // 2
            return signal[start:start + self.segment_length]
        if self.fill == "repeat":
            return fill_canvas(signal, self.segment_length, self.fill_hop)
        return torch.nn.functional.pad(signal, (0, self.segment_length - length))

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        eeg, label, names = unpack_sample(_read_pickle(path))
        mapping = self._as_mapping(eeg, names)
        dense = torch.stack([
            self._crop_or_pad(mapping[name]) if name in mapping else torch.zeros(self.segment_length)
            for name in self.target_names
        ])
        label_key = _normalize_label(label)
        label_id = self.label_to_id.get(label_key, -1)
        return {"eeg": dense, "label": torch.tensor(label_id), "path": str(path)}


def fill_canvas(eeg: Tensor, canvas: int, hop: int) -> Tensor:
    """Fill ``[..., samples]`` out to ``canvas`` by tiling the last ``hop`` samples.

    This is the exact rule the training loop applies with ``--pad-mode repeat``
    (see bsri/pretrain.py): with ``hop=128`` and ``canvas=2000`` the STFT grid
    holds exactly the encoder's 16 frames, so :func:`bsri.model.temporal_subsample`
    becomes the identity instead of repeating frames.  A zero-filled tail is a
    *constant* signal, and the frame repetition it forces is what this replaces.

    A no-op when the input is already at least ``canvas`` long, or ``canvas <= 0``.
    """
    length = int(eeg.shape[-1])
    if canvas <= 0 or length >= canvas:
        return eeg
    if length < hop:
        raise ValueError(
            f"fill_canvas: {length} samples is shorter than one hop ({hop}); cannot tile the last frame"
        )
    fill = canvas - length
    tail = torch.arange(length - hop, length, device=eeg.device)
    reps = -(-fill // hop)  # ceiling division
    return torch.cat([eeg, eeg.index_select(-1, tail.repeat(reps)[:fill])], dim=-1)

def make_loader(
    root: str | Path,
    split: str,
    system: str,
    label_to_id: dict[str, int] | None,
    batch_size: int,
    num_workers: int,
    segment_length: int,
    channel_names: list[str] | None,
    shuffle: bool | None = None,
) -> DataLoader:
    paths = find_split_files(root, split)
    dataset = EEGPickleDataset(
        paths,
        system=system,
        label_to_id=label_to_id,
        segment_length=segment_length,
        training=split == "train",
        source_channel_names=channel_names,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
