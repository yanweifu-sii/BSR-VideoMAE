"""Consolidated BSR-I model definitions.

The BSR-I pipeline: a frozen :class:`BSRenderer` renders a dense EEG batch into
a ``[B,3,T,H,W]`` magnitude-STFT video; an adapter (``ResizeCLAHEAdapter``,
``ResizeOnlyAdapter`` or ``RenderToVideoAdapter``) maps it to RGB frames in
``[0,1]`` at the encoder's input size; the frames are ImageNet-normalized and
relaid out from ``[B,C,T,H,W]`` to Hugging Face's ``[B,T,C,H,W]``; and a
VideoMAE encoder (optionally Q/V LoRA-adapted) consumes them.  On top of that
stack this module also holds the readout and training heads: the MAE pretraining
wrapper (:class:`BSRIPretrainingModel`), JEPA utilities, the PSD band-power
heads (:class:`PSDHead` / :class:`ChannelPSDHead`), the multi-task
:class:`TaskHeads`, the few-shot classifiers, and the pipeline builders
(:func:`build_pretraining`, :func:`build_psd_model`).

``bsri.checkpoints`` imports this module, so :func:`build_pretraining` imports
its checkpoint loaders lazily (function-local import) to avoid a circular
import; this module never imports ``bsri.checkpoints`` at module level.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Iterator, TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import VideoMAEConfig, VideoMAEForPreTraining, VideoMAEModel

if TYPE_CHECKING:
    from .ref_data import RefCachePSDDataset


# --- BSR renderer (merged from bsri/renderer.py): channel grids, STFT helpers,
# Up/Down blocks, and the differentiable EEG -> spectral-video renderer. ---

CHANNELS_1020 = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ",
    "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2",
)
CHANNELS_1010 = (
    "FP1", "FPZ", "FP2", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2",
    "AF4", "AF6", "AF8", "F7", "F5", "F3", "F1", "FZ", "F2", "F4",
    "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4",
    "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8",
    "O1", "OZ", "O2",
)
CHANNELS_1020_1010 = tuple(dict.fromkeys(CHANNELS_1010 + CHANNELS_1020))

GRID_1020 = (
    ("", "FP1", "", "FP2", ""),
    ("F7", "F3", "FZ", "F4", "F8"),
    ("T3", "C3", "CZ", "C4", "T4"),
    ("T5", "P3", "PZ", "P4", "T6"),
    ("", "O1", "", "O2", ""),
)
GRID_1010 = (
    ("", "", "FP1", "", "FPZ", "", "FP2", "", ""),
    ("AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8"),
    ("F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"),
    ("FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"),
    ("T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"),
    ("TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"),
    ("P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"),
    ("PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8"),
    ("", "", "O1", "", "OZ", "", "O2", "", ""),
)
GRID_1020_1010 = (
    GRID_1010[0], GRID_1010[1], GRID_1010[2], GRID_1010[3],
    ("T3", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T4"),
    GRID_1010[5],
    ("T5", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "T6"),
    GRID_1010[7], GRID_1010[8],
)

SYSTEM_GRIDS = {"1020": GRID_1020, "1010": GRID_1010, "1020_1010": GRID_1020_1010}
SYSTEM_CHANNELS = {
    "1020": CHANNELS_1020,
    "1010": CHANNELS_1010,
    "1020_1010": CHANNELS_1020_1010,
}


def canonical_channel(name: str) -> str:
    """Normalize common EEG channel spellings (e.g. ``EEG Fp1-REF``)."""
    value = str(name).upper().strip()
    if value.startswith("EEG "):
        value = value[4:]
    value = value.split("-")[0].split(" ")[0].rstrip(".")
    return value


def stft_hop_for_frames(length: int, num_frames: int = 16, n_fft: int = 200) -> int:
    """Largest STFT hop for which ``length`` samples give exactly ``num_frames`` frames.

    ``torch.stft(center=True)`` returns ``1 + floor(length / hop)`` frames for a signal of
    ``length`` samples (the ``n_fft`` padding cancels for even ``n_fft``), so a hop that
    lands on exactly ``num_frames`` must satisfy ``length // hop == num_frames - 1``.  The
    valid hops form a narrow band; we take the largest, which keeps the grid as sparse as
    possible and leaves the last frame window reaching past ``length``, so no sample is left
    uncovered.

    This is the A2 input rule: every window length gets its own hop, so the STFT grid holds
    the encoder frames natively and nothing is filled, truncated or interpolated.  The
    frequency axis is unaffected (``n_fft`` is fixed), so band-power targets stay comparable
    across hops.
    """
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    if int(num_frames) < 2:
        raise ValueError("num_frames must be at least 2")
    lo = max(1, length // int(num_frames))
    hi = max(lo, length // (int(num_frames) - 1) + 2)
    valid = [hop for hop in range(lo, hi + 1) if 1 + length // hop == int(num_frames)]
    if not valid:
        raise ValueError(
            f"no hop gives exactly {num_frames} frames from {length} samples; "
            f"checked {lo}..{hi}"
        )
    return max(valid)


class Up(nn.Module):
    """Name and nesting intentionally match ``rendering/model.py``."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(x)


class Down(nn.Module):
    """Mirror of :class:`Up`; matches the checkpoint's ``pix2eeg`` stack."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)


class BSRenderer(nn.Module):
    """Render a dense EEG batch to ``[B, 3, T, H, W]`` without disk I/O.

    A 10-10 grid produces 288x288 frames; a 10-20 grid produces 160x160.
    ``forward`` accepts either a dense tensor plus channel names or a mapping of
    channel name to ``[B, samples]`` tensors.
    """

    def __init__(
        self,
        n_fft: int = 200,
        # 128, not 100: with n_fft=200 that makes a 2000-sample window land on
        # exactly 16 STFT frames (the encoder's num_frames) with every sample covered,
        # so the temporal axis never has to be resampled (no repeated/dropped frames).
        hop_length: int = 128,
        pix_channels: int = 3,
        system: str = "1010",
    ) -> None:
        super().__init__()
        if system not in SYSTEM_GRIDS:
            raise ValueError(f"Unknown EEG system {system!r}; choose from {tuple(SYSTEM_GRIDS)}")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pix_channels = pix_channels
        self.system = system
        fft_channels = n_fft // 2 + 1
        self.eeg2pix = nn.Sequential(
            Up(fft_channels, fft_channels // 2),
            Up(fft_channels // 2, fft_channels // 4),
            Up(fft_channels // 4, fft_channels // 8),
            Up(fft_channels // 8, fft_channels // 16),
            Up(fft_channels // 16, pix_channels),
        )
        # eeg2pix and pix2eeg are a matched pair in the released renderer checkpoint:
        # the first renders the |STFT| grid to RGB, the second inverts it.  The inverse is
        # not used for rendering, only to turn a reconstructed frame back into a spectrum.
        self.pix2eeg = nn.Sequential(
            Down(3, 6),
            Down(6, 12),
            Down(12, 25),
            Down(25, 50),
            Down(50, 101),
        )
        # Explicit rectangular window reproduces torch.stft(window=None) without warnings.
        self.register_buffer("stft_window", torch.ones(n_fft), persistent=False)
        # Flat grid positions of the real electrodes, reordered into ``channel_names``
        # order.  The GxG grid has padding slots and its row-major order differs from the
        # 10-10 channel list, so a naive flatten would misalign the spectrum channel axis
        # against the PSD head's channel axis.
        self.register_buffer("electrode_index", self._electrode_indices(), persistent=False)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return SYSTEM_CHANNELS[self.system]

    @property
    def output_size(self) -> int:
        return len(SYSTEM_GRIDS[self.system]) * 32

    def _dense_input(
        self,
        eeg: Tensor | Mapping[str, Tensor],
        channel_names: Sequence[str] | None,
    ) -> tuple[Tensor, tuple[str, ...]]:
        if isinstance(eeg, Tensor):
            names = tuple(canonical_channel(x) for x in (channel_names or self.channel_names))
            if eeg.ndim != 3:
                raise ValueError(f"Dense EEG must be [B,C,L], received {tuple(eeg.shape)}")
            if eeg.shape[1] != len(names):
                raise ValueError(f"EEG has {eeg.shape[1]} channels but {len(names)} names were supplied")
            return eeg, names
        if not eeg:
            raise ValueError("EEG mapping is empty")
        normalized = {canonical_channel(k): v for k, v in eeg.items()}
        first = next(iter(normalized.values()))
        if first.ndim == 1:
            normalized = {k: v.unsqueeze(0) for k, v in normalized.items()}
            first = next(iter(normalized.values()))
        names = tuple(normalized)
        dense = torch.stack([normalized[k] for k in names], dim=1)
        return dense, names

    def spatialize(
        self,
        eeg: Tensor | Mapping[str, Tensor],
        channel_names: Sequence[str] | None = None,
    ) -> Tensor:
        """Return magnitude STFT tokens as ``[B,T,F,G,G]``."""
        dense, names = self._dense_input(eeg, channel_names)
        lookup = {name: index for index, name in enumerate(names)}
        grid = SYSTEM_GRIDS[self.system]
        batch, _, samples = dense.shape
        zero = dense.new_zeros((batch, 1, samples))
        padded = torch.cat((dense, zero), dim=1)
        zero_index = dense.shape[1]
        indices = [lookup.get(channel, zero_index) if channel else zero_index for row in grid for channel in row]
        index = torch.tensor(indices, device=dense.device)
        spatial = padded.index_select(1, index).reshape(batch * len(indices), samples)
        spectrum = torch.stft(
            spatial,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.stft_window.to(dtype=dense.dtype),
            return_complex=True,
        ).abs()
        frames = spectrum.shape[-1]
        size = len(grid)
        return spectrum.reshape(batch, size, size, -1, frames).permute(0, 4, 3, 1, 2)

    def _electrode_indices(self) -> Tensor:
        """Flat grid slots of the real electrodes, in ``channel_names`` order."""
        grid = SYSTEM_GRIDS[self.system]
        flat = [name for row in grid for name in row]
        slots = [index for index, name in enumerate(flat) if name]
        names = [flat[index] for index in slots]
        order = [names.index(name) for name in self.channel_names]
        return torch.tensor([slots[index] for index in order], dtype=torch.long)


    def spectrum(self, eeg, channel_names=None) -> Tensor:
        """Magnitude STFT tokens ``[B, T, K, G, G]`` -- the renderer's own input."""
        return self.spatialize(eeg, channel_names)


    def render_tokens(self, tokens: Tensor) -> Tensor:
        """Render pre-computed ``[B, T, K, G, G]`` tokens, bypassing the STFT.

        Used to render a *masked* spectrum: the STFT is computed once, frames are
        zeroed, and only then is ``eeg2pix`` applied.
        """
        batch, frames, frequencies, height, width = tokens.shape
        pixels = self.eeg2pix(tokens.reshape(batch * frames, frequencies, height, width))
        return pixels.reshape(batch, frames, self.pix_channels, pixels.shape[-2], pixels.shape[-1]).permute(0, 2, 1, 3, 4)


    def pix2eeg_spectrum(self, pixels: Tensor) -> Tensor:
        """Invert :meth:`render_tokens`: ``[B, 3, T, H, W] -> [B, T, K, G, G]``."""
        batch, channels, frames, height, width = pixels.shape
        flat = pixels.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        out = self.pix2eeg(flat)
        _, bins, grid, _ = out.shape
        return out.reshape(batch, frames, bins, grid, grid)


    def forward(
        self,
        eeg: Tensor | Mapping[str, Tensor],
        channel_names: Sequence[str] | None = None,
    ) -> Tensor:
        tokens = self.spatialize(eeg, channel_names)
        batch, frames, frequencies, height, width = tokens.shape
        pixels = self.eeg2pix(tokens.reshape(batch * frames, frequencies, height, width))
        return pixels.reshape(batch, frames, self.pix_channels, pixels.shape[-2], pixels.shape[-1]).permute(0, 2, 1, 3, 4)


# --- Adapters (merged from bsri/adapter.py): renderer -> VideoMAE frame
# conversion (learned CLAHE-style, parameter-free resize, differentiable). ---

class ResidualContrastBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class ResizeCLAHEAdapter(nn.Module):
    """Framewise local-contrast adapter with a learned deformable resampler.

    Input is an unbounded BSR/BGR video ``[B,3,T,H,W]``. Output is an RGB-like
    float video in ``[0,1]`` with exactly ``output_size`` spatial dimensions.
    The offset and output heads are zero initialized, so the initial geometric
    behavior is ordinary bilinear resizing while all refinements are learned.
    """

    def __init__(
        self,
        output_size: int = 224,
        hidden_channels: int = 32,
        max_offset_pixels: float = 4.0,
        input_mode: str = "legacy_uint8",
        eps: float = 1e-6,
        flip_channels: bool = False,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.max_offset_pixels = max_offset_pixels
        if input_mode not in {"legacy_uint8", "minmax"}:
            raise ValueError("input_mode must be 'legacy_uint8' or 'minmax'")
        self.input_mode = input_mode
        self.eps = eps
        # Historically this adapter flipped the channel axis to undo the legacy
        # cv2 BGR write.  The proven-good baseline for this project is however the
        # parameter-free ResizeOnlyAdapter, which does NOT flip, so enabling the
        # flip here would hand a frozen VideoMAE a channel-permuted input at
        # initialisation -- a silent confound.  Measured: with the flip the two
        # adapters differ by max|d| = 0.926 at init, without it by 2.1e-06.
        # Default off so that ``learned`` starts bit-equivalent to ``resize``.
        self.flip_channels = flip_channels
        self.log_gain = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.stem = nn.Conv2d(9, hidden_channels, 3, padding=1)
        self.body = nn.Sequential(
            ResidualContrastBlock(hidden_channels, 1),
            ResidualContrastBlock(hidden_channels, 2),
            ResidualContrastBlock(hidden_channels, 4),
        )
        self.offset_head = nn.Conv2d(hidden_channels, 2, 3, padding=1)
        self.residual_head = nn.Sequential(
            ResidualContrastBlock(hidden_channels, 1),
            nn.Conv2d(hidden_channels, 3, 3, padding=1),
        )
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def _to_unit_range(self, x: Tensor) -> Tensor:
        if self.input_mode == "legacy_uint8":
            # Original rendering called numpy.astype(uint8) before OpenCV. Match
            # its truncate-and-wrap forward values, but use a straight-through
            # gradient so future renderer unfreezing remains possible.
            wrapped = torch.remainder(torch.trunc(x), 256.0) / 255.0
            surrogate = x / 255.0
            unit = surrogate + (wrapped - surrogate).detach()
        else:
            minimum = x.amin(dim=(-2, -1), keepdim=True)
            maximum = x.amax(dim=(-2, -1), keepdim=True)
            unit = (x - minimum) / (maximum - minimum).clamp_min(self.eps)
        gain = self.log_gain.exp().clamp(max=20.0)
        return torch.sigmoid(torch.logit(unit.clamp(self.eps, 1 - self.eps)) * gain + self.bias)

    def _base_grid(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        # Pixel-center grid equivalent to interpolate(..., align_corners=False).
        axis = (torch.arange(self.output_size, device=device, dtype=dtype) + 0.5) * (2.0 / self.output_size) - 1.0
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Adapter expects [B,3,T,H,W], received {tuple(video.shape)}")
        batch, channels, frames, height, width = video.shape
        x = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        # Legacy cv2.VideoWriter receives BGR and VideoMAE later decodes RGB; some
        # checkpoints rely on that flip.  See ``flip_channels`` in __init__.
        unit = self._to_unit_range(x.float())
        if self.flip_channels:
            unit = unit.flip(1)
        local_mean = F.avg_pool2d(unit, kernel_size=17, stride=1, padding=8)
        local_small = F.avg_pool2d(unit, kernel_size=33, stride=1, padding=16)
        features = self.body(self.stem(torch.cat((unit, unit - local_mean, unit - local_small), dim=1)))

        coarse = F.interpolate(features, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        offsets = torch.tanh(self.offset_head(coarse))
        scale = offsets.new_tensor((2.0 * self.max_offset_pixels / width, 2.0 * self.max_offset_pixels / height))
        offsets = offsets.permute(0, 2, 3, 1) * scale
        grid = self._base_grid(batch * frames, unit.device, unit.dtype) + offsets
        resized = F.grid_sample(unit, grid, mode="bilinear", padding_mode="border", align_corners=False)
        enhanced = (resized + 0.25 * torch.tanh(self.residual_head(coarse))).clamp(0.0, 1.0)
        return enhanced.reshape(batch, frames, channels, self.output_size, self.output_size).permute(0, 2, 1, 3, 4)


class ResizeOnlyAdapter(nn.Module):
    """Parameter-free adapter: a fixed value mapping followed by bilinear resizing.

    It owns no parameters or buffers, so ``state_dict()`` is empty and no
    optimizer state exists for it. ``value_mode`` selects the framewise mapping
    applied before the resize:

    ``minmax``
        Per frame and channel min-max normalization to ``[0, 1]`` with the
        denominator clamped at ``1e-6``; deterministic and CLAHE-free.
    ``legacy_uint8``
        Reproduce the old ``remainder(trunc(x), 256) / 255`` forward values of
        :class:`ResizeCLAHEAdapter` (without its learned refinement).
    ``legacy_uint8_ste``
        The same forward values as ``legacy_uint8``, but with a straight-through
        gradient.  ``torch.trunc`` is flat almost everywhere, so ``legacy_uint8``
        passes **no** gradient back to whatever produced ``x`` -- which silently
        makes an upstream module untrainable (in particular the renderer, when
        ``pretrain_psd --tuning-renderer`` unfreezes it).  This mode keeps the
        forward bit-identical (so the encoder's input distribution is unchanged)
        while letting the gradient through, which is what makes renderer
        fine-tuning a well-posed experiment rather than a no-op.
    ``clamp``
        Plain ``clamp(0, 1)``.
    ``raw``
        No value mapping at all; only the resize (the trailing clamp still
        bounds the output to ``[0, 1]``).
    """

    VALUE_MODES = ("minmax", "legacy_uint8", "legacy_uint8_ste", "clamp", "raw")

    def __init__(self, output_size: int = 224, value_mode: str = "minmax") -> None:
        super().__init__()
        if value_mode not in self.VALUE_MODES:
            raise ValueError(f"value_mode must be one of {self.VALUE_MODES}, received {value_mode!r}")
        self.output_size = output_size
        self.value_mode = value_mode

    def _to_unit_range(self, x: Tensor) -> Tensor:
        if self.value_mode == "minmax":
            minimum = x.amin(dim=(-2, -1), keepdim=True)
            maximum = x.amax(dim=(-2, -1), keepdim=True)
            return (x - minimum) / (maximum - minimum).clamp_min(1e-6)
        if self.value_mode == "legacy_uint8":
            return torch.remainder(torch.trunc(x), 256.0) / 255.0
        if self.value_mode == "legacy_uint8_ste":
            truncated = torch.trunc(x)
            # Forward is trunc(x); backward is the identity (straight-through).
            return torch.remainder(x + (truncated - x).detach(), 256.0) / 255.0
        # "clamp" and "raw" differ only through the trailing clamp in forward.
        return x

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Adapter expects [B,3,T,H,W], received {tuple(video.shape)}")
        batch, channels, frames, height, width = video.shape
        x = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).float()
        unit = self._to_unit_range(x).clamp(0.0, 1.0)
        resized = F.interpolate(
            unit, size=(self.output_size, self.output_size), mode="bilinear",
            align_corners=False, antialias=True,
        )
        return resized.reshape(batch, frames, channels, self.output_size, self.output_size).permute(0, 2, 1, 3, 4)


class RenderToVideoAdapter(nn.Module):
    """Own the whole renderer -> VideoMAE conversion, differentiably.

    Motivation.  The renderer emits a **zero-centred, bipolar** field whose scale
    spans ~6 orders of magnitude across sources (measured p99.5: +/-250 for
    ``seed_iv``, +/-3.1e3 for ``hinss``, +/-2.2e4 for ``shirazi_hbn``, tails to
    +/-1e6).  The historical conversion -- ``remainder(trunc(x), 256)/255`` --
    sends every *negative* value to the top of the range (``-100 -> 156/255 =
    0.61``), so it destroys the sign structure at x = 0 for **every** source, not
    just the large-amplitude ones.  That mapping is also non-differentiable
    (``trunc`` is flat), which is why upstream training needed a straight-through
    hack and a trust region to stay honest.

    This module replaces the whole conversion with ordinary differentiable ops, so
    renderer, adapter, encoder and head all train from a single ``loss.backward()``
    with no special handling:

    1. **sign-preserving log compression** ``z = sign(x) * log1p(|x| / c)``.
       Monotone in ``|x|`` (so the absolute magnitude that band power depends on is
       preserved, unlike any per-frame normalisation), antisymmetric about 0, and
       its gradient ``1/(c + |x|)`` is algebraic -- it does **not** underflow over
       the full dynamic range.  Any saturating map instead decays exponentially and
       measures exactly 0 gradient for most sources.
    2. **learnable per-channel affine**, bias initialised to 0 so that the
       antisymmetry ``u(-x) = 1 - u(x)`` (i.e. x = 0 renders as mid-grey) holds.
    3. **resize** of the logits to ``output_size``.
    4. **local-contrast refinement** in logit space with a **zero-initialised**
       residual head, so at init the module is exactly compression+affine+resize.
    5. **sigmoid** as the final bound into ``(0, 1)`` -- smooth, monotone, non-zero
       gradient everywhere, and no clamp dead zone.

    The convolutions run at ``output_size`` (after the resize) rather than at the
    renderer's 288x288, which is what keeps the memory sane.
    """

    def __init__(
        self,
        output_size: int = 224,
        hidden_channels: int = 16,
        init_scale: float = 255.0,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.residual_scale = residual_scale
        self.log_c = nn.Parameter(torch.full((1, 3, 1, 1), float(math.log(init_scale))))
        self.gain = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.stem = nn.Conv2d(9, hidden_channels, 3, padding=1)
        self.body = nn.Sequential(
            ResidualContrastBlock(hidden_channels, 1),
            ResidualContrastBlock(hidden_channels, 2),
            ResidualContrastBlock(hidden_channels, 4),
        )
        self.residual_head = nn.Sequential(
            ResidualContrastBlock(hidden_channels, 1),
            nn.Conv2d(hidden_channels, 3, 3, padding=1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def to_logits(self, x: Tensor) -> Tensor:
        """The differentiable, sign-preserving compression into logit space."""
        scale = self.log_c.exp().clamp_min(1e-3)
        compressed = torch.sign(x) * torch.log1p(x.abs() / scale)
        return compressed * self.gain + self.bias

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Adapter expects [B,3,T,H,W], received {tuple(video.shape)}")
        batch, channels, frames, height, width = video.shape
        x = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width).float()
        logits = F.interpolate(
            self.to_logits(x), size=(self.output_size, self.output_size), mode="bilinear",
            align_corners=False, antialias=True,
        )
        unit = torch.sigmoid(logits)
        local_mean = F.avg_pool2d(unit, kernel_size=17, stride=1, padding=8)
        local_small = F.avg_pool2d(unit, kernel_size=33, stride=1, padding=16)
        features = self.body(self.stem(torch.cat((unit, unit - local_mean, unit - local_small), dim=1)))
        refined = torch.sigmoid(logits + self.residual_scale * torch.tanh(self.residual_head(features)))
        return refined.reshape(batch, frames, channels, self.output_size, self.output_size).permute(0, 2, 1, 3, 4)


# --- LoRA (merged from bsri/lora.py): VideoMAE attention adapters. ---

class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual.

    Hugging Face VideoMAE calls ``functional.linear`` with the attention
    projection's ``weight`` instead of calling the projection module.  The
    dynamic properties below therefore expose the merged LoRA weight while
    retaining gradients only for A and B.  ``forward`` also makes the wrapper
    usable as an ordinary linear layer.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self) -> Tensor:
        # Weight dropout is shared over a batch.  This is compatible with the
        # direct-weight access used by VideoMAE and is disabled at evaluation.
        low_rank = F.dropout(self.lora_A.weight, p=self.dropout.p, training=self.training)
        return self.base.weight + (self.lora_B.weight @ low_rank) * self.scaling

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


def inject_videomae_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    targets: tuple[str, ...] = ("query", "value"),
) -> list[str]:
    """Replace every VideoMAE attention query/value projection in-place."""
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not name.endswith(targets) or not isinstance(module, nn.Linear):
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, rank, alpha, dropout))
        replaced.append(name)
    if not replaced:
        raise ValueError("No VideoMAE attention projections matched the LoRA targets")
    return replaced


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield from module.lora_A.parameters()
            yield from module.lora_B.parameters()


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    current = model.state_dict()
    unknown = sorted(set(state) - set(current))
    if unknown:
        raise ValueError(f"LoRA checkpoint contains unknown keys: {unknown[:5]}")
    missing = sorted(
        name for name in current
        if (".lora_A." in name or ".lora_B." in name) and name not in state
    )
    if missing:
        raise ValueError(f"LoRA checkpoint is missing keys: {missing[:5]}")
    model.load_state_dict(state, strict=False)


def trainable_parameter_summary(model: nn.Module) -> tuple[int, int, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total, 100.0 * trainable / total


# --- CLAHE enhancer (merged from bsri/clahe.py): the offline corpora's CLAHE step. ---

class CLAHEEnhancer(nn.Module):
    """Apply the offline corpora's CLAHE to ``[B,3,T,H,W]`` frames in ``[0,1]``."""

    def __init__(self, clip_limit: float = 2.0, tile_grid: int = 32) -> None:
        super().__init__()
        self.clip_limit = float(clip_limit)
        self.tile_grid = int(tile_grid)
        self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_grid, self.tile_grid))

    def _enhance(self, frames: np.ndarray) -> np.ndarray:
        """``(N,H,W,3)`` uint8 -> ``(N,H,W,3)`` uint8, CLAHE on L in LAB."""
        out = np.empty_like(frames)
        for index in range(frames.shape[0]):
            lab = cv2.cvtColor(frames[index], cv2.COLOR_BGR2LAB)
            lightness, a_channel, b_channel = cv2.split(lab)
            enhanced = cv2.merge((self._clahe.apply(lightness), a_channel, b_channel))
            out[index] = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return out

    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"CLAHEEnhancer expects [B,3,T,H,W], received {tuple(video.shape)}")
        # The same BGR-ordered memory the offline writer hands to cv2.
        as_uint8 = (
            video.detach().permute(0, 2, 3, 4, 1).flatten(0, 1).mul(255.0).round().clamp(0, 255).to(torch.uint8)
        )
        enhanced = torch.as_tensor(self._enhance(as_uint8.cpu().numpy()), dtype=video.dtype, device=video.device)
        restored = enhanced.reshape(video.shape[0], video.shape[2], video.shape[3], video.shape[4], 3)
        restored = restored.permute(0, 4, 1, 2, 3).div(255.0)
        # Straight-through: true CLAHE forward, identity gradient.
        return video + (restored - video).detach()


# --- Distillation teacher (merged from bsri/teacher.py): offline legacy
# CLAHE+resize operator used only as an adapter distillation target. ---

@torch.no_grad()
def clahe_resize_teacher(
    rendered: Tensor,
    output_size: int = 224,
    clip_limit: float = 2.0,
    tile_grid_size: int = 32,
) -> Tensor:
    """Apply legacy uint8 conversion, LAB-CLAHE and area resizing.

    This is deliberately non-differentiable and CPU based. It supplies targets
    during Adapter pretraining only; BSR-I inference has no OpenCV dependency in
    its forward graph.
    """
    if rendered.ndim != 5 or rendered.shape[1] != 3:
        raise ValueError(f"Teacher expects [B,3,T,H,W], received {tuple(rendered.shape)}")
    # Reproduce the reference code's direct numpy uint8 conversion, including
    # truncation and modulo wrapping outside [0,255].
    array = rendered.detach().float().cpu().permute(0, 2, 3, 4, 1).numpy()
    array = np.nan_to_num(array).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    result = np.empty_like(array, dtype=np.uint8)
    for batch in range(array.shape[0]):
        for frame in range(array.shape[1]):
            # Preserve the original renderer's OpenCV/BGR interpretation.
            lab = cv2.cvtColor(array[batch, frame], cv2.COLOR_BGR2LAB)
            luminance, a, b = cv2.split(lab)
            enhanced = cv2.cvtColor(cv2.merge((clahe.apply(luminance), a, b)), cv2.COLOR_LAB2BGR)
            # VideoWriter consumed BGR; pytorchvideo decoded the saved MP4 as RGB.
            result[batch, frame] = enhanced[..., ::-1]
    target = torch.from_numpy(result).permute(0, 1, 4, 2, 3).float().div_(255.0)
    batch, frames, channels, height, width = target.shape
    target = F.interpolate(
        target.reshape(batch * frames, channels, height, width),
        size=(output_size, output_size), mode="bilinear", align_corners=False, antialias=True,
    ).reshape(batch, frames, channels, output_size, output_size).permute(0, 2, 1, 3, 4)
    return target.to(device=rendered.device, non_blocking=True)


def gradient_loss(prediction: Tensor, target: Tensor) -> Tensor:
    pred_dx = prediction[..., 1:] - prediction[..., :-1]
    target_dx = target[..., 1:] - target[..., :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()


# --- JEPA utilities (merged from bsri/jepa.py): masking, predictor, EMA, loss. ---

def token_grid(config) -> tuple[int, int, int]:
    """Return (temporal patches, spatial grid, total tokens) for a VideoMAE config."""
    temporal = config.num_frames // config.tubelet_size
    grid = config.image_size // config.patch_size
    return temporal, grid, temporal * grid * grid


def sample_block_mask(
    temporal: int,
    grid: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample a shared multi-block mask over the (temporal, grid, grid) token grid."""
    total = temporal * grid * grid
    target = max(1, min(total - 1, int(round(total * mask_ratio))))
    mask = torch.zeros(temporal, grid, grid, dtype=torch.bool)
    tries = 0
    while int(mask.sum()) < target and tries < 256:
        block_t = int(torch.randint(1, min(4, temporal) + 1, (1,), generator=generator))
        block_h = int(torch.randint(2, min(5, grid) + 1, (1,), generator=generator))
        block_w = int(torch.randint(2, min(5, grid) + 1, (1,), generator=generator))
        t0 = int(torch.randint(0, temporal - block_t + 1, (1,), generator=generator))
        h0 = int(torch.randint(0, grid - block_h + 1, (1,), generator=generator))
        w0 = int(torch.randint(0, grid - block_w + 1, (1,), generator=generator))
        mask[t0:t0 + block_t, h0:h0 + block_h, w0:w0 + block_w] = True
        tries += 1
    flat = mask.reshape(-1)
    excess = int(flat.sum()) - target
    if excess > 0:
        selected = flat.nonzero(as_tuple=True)[0]
        drop = selected[torch.randperm(selected.numel(), generator=generator)[:excess]]
        flat[drop] = False
    return flat


class JEPAPredictor(nn.Module):
    """Narrow transformer predicting masked-token latents from visible latents."""

    def __init__(
        self,
        hidden_size: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        num_tokens: int = 1568,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.position_embeddings = nn.Parameter(torch.zeros(num_tokens, hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, context: Tensor, visible_index: Tensor, masked_index: Tensor) -> Tensor:
        """Predict target latents at ``masked_index`` from ``context`` [B, n_vis, D]."""
        batch = context.shape[0]
        visible = context + self.position_embeddings[visible_index].unsqueeze(0)
        masked = self.mask_token.unsqueeze(0).expand(batch, masked_index.numel(), -1)
        masked = masked + self.position_embeddings[masked_index].unsqueeze(0)
        hidden = self.blocks(torch.cat((visible, masked), dim=1))
        hidden = self.norm(hidden)[:, visible_index.numel():]
        return self.proj(hidden)


def ema_momentum(step: int, total_steps: int, base: float, final: float) -> float:
    if total_steps <= 0:
        return final
    progress = min(1.0, max(0.0, step / total_steps))
    return base + (final - base) * progress


@torch.no_grad()
def ema_update(student: nn.Module, target: nn.Module, momentum: float) -> None:
    for student_parameter, target_parameter in zip(student.parameters(), target.parameters()):
        target_parameter.mul_(momentum).add_(student_parameter.detach(), alpha=1.0 - momentum)


def copy_ema_target(student: nn.Module) -> nn.Module:
    target = copy.deepcopy(student)
    target.requires_grad_(False)
    return target


def jepa_loss(prediction: Tensor, target: Tensor, mode: str = "layernorm", beta: float = 1.0) -> Tensor:
    target = target.detach()
    if mode == "layernorm":
        target = F.layer_norm(target, (target.shape[-1],))
    elif mode == "l2":
        target = F.normalize(target, dim=-1)
    elif mode != "none":
        raise ValueError(f"Unknown target normalization {mode!r}")
    return F.smooth_l1_loss(prediction, target, beta=beta)


def trainable_parameters(*modules: nn.Module) -> Iterator[nn.Parameter]:
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                yield parameter


# --- End-to-end BSR-I models (former bsri/model.py): renderer -> adapter ->
# normalize/layout -> VideoMAE, plus the classification readouts. ---

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def temporal_subsample(video: Tensor, num_frames: int) -> Tensor:
    """Uniform temporal sampling on ``[B,C,T,H,W]`` (repeat if T is short)."""
    frames = video.shape[2]
    if frames == num_frames:
        return video
    indices = torch.linspace(0, frames - 1, num_frames, device=video.device).round().long()
    return video.index_select(2, indices)


class VideoPreparationMixin:
    renderer: BSRenderer
    adapter: ResizeCLAHEAdapter
    num_frames: int

    def render_and_adapt(self, eeg: Tensor, channel_names: Sequence[str] | None = None) -> tuple[Tensor, Tensor]:
        rendered = temporal_subsample(self.renderer(eeg, channel_names), self.num_frames)
        adapted = self.adapter(rendered)
        return rendered, adapted


class BSRIPretrainingModel(nn.Module, VideoPreparationMixin):
    def __init__(
        self,
        renderer: BSRenderer,
        adapter: ResizeCLAHEAdapter,
        videomae: VideoMAEForPreTraining | None = None,
        config: VideoMAEConfig | None = None,
    ) -> None:
        super().__init__()
        self.renderer = renderer
        self.adapter = adapter
        self.videomae = videomae or VideoMAEForPreTraining(config or VideoMAEConfig())
        self.num_frames = self.videomae.config.num_frames
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    def normalize(self, video: Tensor) -> Tensor:
        return (video - self.image_mean) / self.image_std

    def forward(self, eeg: Tensor, bool_masked_pos: Tensor, channel_names: Sequence[str] | None = None):
        rendered, adapted = self.render_and_adapt(eeg, channel_names)
        # Hugging Face VideoMAE consumes [B,T,C,H,W]. BSR/Adapter use the
        # conventional video layout [B,C,T,H,W] internally.
        pixel_values = self.normalize(adapted).permute(0, 2, 1, 3, 4)
        outputs = self.videomae(pixel_values, bool_masked_pos=bool_masked_pos)
        return outputs, rendered, adapted


class BSRIEncoder(nn.Module, VideoPreparationMixin):
    def __init__(
        self,
        renderer: BSRenderer,
        adapter: ResizeCLAHEAdapter,
        videomae: VideoMAEModel | None = None,
        config: VideoMAEConfig | None = None,
    ) -> None:
        super().__init__()
        self.renderer = renderer
        self.adapter = adapter
        self.videomae = videomae or VideoMAEModel(config or VideoMAEConfig())
        self.num_frames = self.videomae.config.num_frames
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    def forward(self, eeg: Tensor, channel_names: Sequence[str] | None = None) -> Tensor:
        _, adapted = self.render_and_adapt(eeg, channel_names)
        normalized = ((adapted - self.image_mean) / self.image_std).permute(0, 2, 1, 3, 4)
        hidden = self.videomae(normalized).last_hidden_state
        # VideoMAE was pretrained with use_mean_pooling=True; no synthetic CLS token.
        return hidden.mean(dim=1)


class BSRIClassifier(nn.Module):
    def __init__(self, encoder: BSRIEncoder, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(encoder.videomae.config.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(encoder.videomae.config.hidden_size, num_classes),
        )

    def forward(self, eeg: Tensor, channel_names: Sequence[str] | None = None) -> Tensor:
        return self.head(self.encoder(eeg, channel_names))


# --- Pipeline builder (moved from scripts/common.py): renderer + adapter +
# VideoMAE assembly, loading weights through bsri.checkpoints. ---

def build_pretraining(args: argparse.Namespace) -> BSRIPretrainingModel:
    # bsri.checkpoints imports this module (bsri.model), so its loaders must be
    # imported function-locally here; a module-level import would be circular.
    from bsri.checkpoints import load_legacy_videomae, load_renderer_checkpoint

    # The supplied checkpoint came from the original 4.22 config where this is
    # false; it owns videomae.layernorm rather than the newer mean-pooling norm.
    config = VideoMAEConfig(image_size=224, num_frames=16, use_mean_pooling=False)
    renderer = BSRenderer(system=args.system)
    adapter = ResizeCLAHEAdapter(hidden_channels=args.adapter_channels)
    videomae = VideoMAEForPreTraining(config)
    load_renderer_checkpoint(renderer, args.renderer_ckpt)
    load_legacy_videomae(videomae, args.vmae_ckpt)
    return BSRIPretrainingModel(renderer, adapter, videomae)


# --- PSD band-power heads (moved from scripts/pretrain_psd.py): global and
# per-channel readouts plus the PSD pretraining builder. ---

#: Band order of the five PSD targets; matches scripts.build_psd_cache.PSD_BANDS.
BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")
N_BANDS = len(BAND_NAMES)


class PSDHead(nn.Module):
    """Mean-token embedding -> per-channel band power, shaped ``[B, 69, 5]``."""

    def __init__(
        self,
        hidden_size: int,
        channels: int = 69,
        bands: int = N_BANDS,
        hidden: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.bands = int(bands)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.channels * self.bands),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(features.shape[0], self.channels, self.bands)


def channel_grid_coordinates(
    channel_names: tuple[str, ...] = SYSTEM_CHANNELS["1010"],
    grid: tuple[tuple[str, ...], ...] = GRID_1010,
) -> tuple[torch.Tensor, torch.Tensor]:
    """10-10 grid position of every montage channel, normalized to ``[0, 1]``.

    Returns ``(coords [C, 2], known [C])``.  A channel present in ``grid`` gets
    its ``(row, col)`` divided by ``grid_size - 1``; a channel the grid does not
    place (all 69 are placed today) is flagged ``known=False`` so the head can
    substitute a learnable default position.
    """
    lookup = {
        name: (row, col)
        for row, line in enumerate(grid)
        for col, name in enumerate(line)
        if name
    }
    size = len(grid)
    coords = torch.zeros(len(channel_names), 2)
    known = torch.zeros(len(channel_names), dtype=torch.bool)
    for index, name in enumerate(channel_names):
        position = lookup.get(name)
        if position is None:
            continue
        coords[index, 0] = position[0] / max(1, size - 1)
        coords[index, 1] = position[1] / max(1, size - 1)
        known[index] = True
    return coords, known


class ChannelPSDHead(nn.Module):
    """Per-channel readout: learnable queries cross-attend the encoder tokens.

    The pooled mean token averages all 1568 tokens (8 temporal blocks x 14x14
    patches) into one 768-vector, which the BCIC-IV-2a diagnosis showed washes
    out per-channel spectral detail.  This head keeps one learnable query per
    montage channel, seeds it with the channel's 10-10 grid position linearly
    mapped onto the 14x14 patch grid, refines it with ``layers`` cross-attention
    blocks over the full token sequence, and predicts that channel's five band
    powers::

        query[C, 768] + Linear(2 -> 768)(grid_position)
          -> N x (MultiheadAttention(query, tokens, tokens) + residual)
          -> LayerNorm -> Linear(768, 5) -> [B, C, 5]
    """

    def __init__(
        self,
        hidden_size: int,
        channels: int = 69,
        bands: int = N_BANDS,
        heads: int = 8,
        layers: int = 2,
        dropout: float = 0.0,
        channel_names: tuple[str, ...] = SYSTEM_CHANNELS["1010"],
        grid: tuple[tuple[str, ...], ...] = GRID_1010,
    ) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by --channel-heads {heads}")
        self.hidden_size = int(hidden_size)
        self.channels = int(channels)
        self.bands = int(bands)
        self.heads = int(heads)
        self.layers_count = int(layers)
        if len(channel_names) != self.channels:
            raise ValueError(f"{len(channel_names)} channel names for {self.channels} channels")
        coords, known = channel_grid_coordinates(channel_names, grid)
        self.register_buffer("coords", coords, persistent=False)
        self.register_buffer("known", known, persistent=False)
        self.query = nn.Parameter(torch.randn(self.channels, self.hidden_size) * 0.02)
        self.default_position = nn.Parameter(torch.zeros(2))
        self.position = nn.Linear(2, self.hidden_size)
        self.kv_norm = nn.LayerNorm(self.hidden_size)
        self.blocks = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_size, self.heads, dropout=dropout, batch_first=True)
            for _ in range(self.layers_count)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_size) for _ in range(self.layers_count)
        ])
        self.out_norm = nn.LayerNorm(self.hidden_size)
        self.proj = nn.Linear(self.hidden_size, self.bands)
        # Start from the "predict zero" optimum of the standardized target, so
        # the fresh attention head cannot blow the MSE up on its first steps.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def grid_positions(self) -> torch.Tensor:
        fallback = self.default_position.to(self.coords.dtype).unsqueeze(0).expand_as(self.coords)
        return torch.where(self.known.unsqueeze(-1), self.coords, fallback)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"ChannelPSDHead expects [B, S, H], received {tuple(tokens.shape)}")
        positions = self.position(self.grid_positions())
        queries = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1) + positions.unsqueeze(0)
        keys = self.kv_norm(tokens)
        for attention, norm in zip(self.blocks, self.norms):
            attended, _ = attention(norm(queries), keys, keys, need_weights=False)
            queries = queries + attended
        return self.proj(self.out_norm(queries))


class PSDNormalizer(nn.Module):
    """Per-dataset target standardization and validity mask, indexed by dataset id."""

    def __init__(self, dataset_ids: list[str], source: RefCachePSDDataset) -> None:
        super().__init__()
        mean = np.stack([source.psd_mean[name] for name in dataset_ids])
        std = np.stack([source.psd_std[name] for name in dataset_ids])
        valid = np.stack([source.psd_valid[name] for name in dataset_ids])
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("std", torch.from_numpy(std.astype(np.float32)))
        self.register_buffer("valid", torch.from_numpy(valid))
        self.dataset_ids = list(dataset_ids)

    def target(self, psd: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(normalized_target, mask)`` for one dataset id."""
        mean = self.mean[index].to(dtype=torch.float32)
        std = self.std[index].to(dtype=torch.float32)
        valid = self.valid[index]
        mask = (valid.unsqueeze(-1) & (std > 0)).unsqueeze(0).expand(psd.shape[0], -1, -1)
        normalized = (psd.float() - mean) / std.clamp_min(1e-6)
        return normalized, mask


def head_from_config(config: dict[str, Any], hidden_size: int) -> nn.Module:
    """Rebuild the PSD head recorded in a checkpoint's ``config``.

    Shared by :func:`build_psd_model` and the BCIC-IV-2a probe so evaluation
    always reconstructs the same readout the checkpoint was trained with.
    """
    channels = int(config.get("psd_channels", 69))
    if str(config.get("readout", "global")) == "channel":
        return ChannelPSDHead(
            hidden_size, channels=channels,
            heads=int(config.get("channel_heads", 8)),
            layers=int(config.get("channel_layers", 2)),
            dropout=float(config.get("channel_dropout", 0.0)),
        )
    return PSDHead(
        hidden_size, channels=channels, bands=N_BANDS,
        hidden=int(config.get("head_hidden", 512)),
        dropout=float(config.get("head_dropout", 0.1)),
    )


def build_psd_model(args: argparse.Namespace):
    """Renderer (frozen) + Adapter + VideoMAE-in-LoRA + PSD head."""
    model = build_pretraining(args)
    if args.adapter_mode == "render2video":
        # The fully differentiable conversion (sign-preserving log compression ->
        # affine -> resize -> refinement -> sigmoid).  No trunc / modulo / STE, so
        # renderer, adapter, encoder and head all train from one loss.backward().
        model.adapter = RenderToVideoAdapter(
            output_size=model.videomae.config.image_size,
            hidden_channels=getattr(args, "adapter_channels", 16),
        )
    elif args.adapter_mode == "learned":
        # ``learned`` historically first constructed a ResizeCLAHEAdapter and then
        # immediately overwrote it with the parameter-free ResizeOnlyAdapter, so
        # the mode has always resolved to ResizeOnlyAdapter; the dead
        # ResizeCLAHEAdapter construction was removed here (behavior preserved).
        model.adapter = ResizeOnlyAdapter(
            output_size=model.videomae.config.image_size, value_mode=args.value_mode
        )
    model.requires_grad_(False)
    replaced = inject_videomae_lora(
        model.videomae.videomae, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    if args.init is not None:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        if "lora_state" not in init:
            raise ValueError(f"--init {args.init} carries no lora_state")
        load_lora_state_dict(model.videomae.videomae, init["lora_state"])
    if args.tuning == "all":
        # Default: freeze nothing.  Sits after the blanket requires_grad_(False) above
        # and, unlike --tuning full, also covers the decoder-side tensors that live on
        # the ForPreTraining wrapper rather than inside .decoder: ``encoder_to_decoder``
        # and ``mask_token`` (295,296 params).  Unfreezing only .decoder leaves the mask
        # token -- the vector that actually fills masked patches -- fixed, which caps how
        # far the MAE term can adapt to rendered EEG.
        previously_frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        model.requires_grad_(True)
        print(json.dumps({"tuning": "all",
                          "newly_trainable_tensors": len(previously_frozen),
                          "examples": previously_frozen[:5]}), flush=True)
    if args.tuning == "full":
        model.videomae.videomae.requires_grad_(True)
    if getattr(args, "tuning_decoder", False):
        # The MAE decoder is frozen by default, which turns the auxiliary MAE term
        # into a constant penalty: once the fixed decoder hits its floor the encoder
        # cannot lower it further.  Unfreezing makes L_mae an actual converging
        # objective -- at the price of the decoder co-adapting, so read r2 next to
        # L_mae to tell "autoencoder improving" from "decoder absorbing the change".
        if args.tuning not in ("full", "all"):
            raise ValueError("--tuning-decoder requires --tuning full (or all, where all is already unfrozen)")
        decoder_module = getattr(model.videomae, "decoder", None)
        if decoder_module is None:
            raise ValueError("--tuning-decoder: model.videomae exposes no decoder")
        decoder_module.requires_grad_(True)
    if args.tuning == "adapter":
        # Freeze the encoder (LoRA included) but keep the inherited Stage-3 LoRA
        # weights in place, so the frozen encoder is the adapted one rather than
        # the raw VideoMAE.
        for parameter in lora_parameters(model.videomae.videomae):
            parameter.requires_grad_(False)
    if args.adapter_mode != "resize":
        model.adapter.requires_grad_(True)
    if getattr(args, "tuning_renderer", False):
        # The renderer is a 26,856-parameter purely linear stack (5 bare
        # ConvTranspose2d, no norm/activation/dropout).  Unfreezing it lets the
        # model choose the spectral -> 3-channel mix instead of inheriting the
        # frozen TUEG one.  Gradient only reaches it if the adapter's value mode
        # passes one through -- ``legacy_uint8`` does NOT (torch.trunc is flat),
        # which is why main() swaps in ``legacy_uint8_ste`` for this mode.
        model.renderer.requires_grad_(True)
    head = head_from_config(vars(args), model.videomae.config.hidden_size)
    return model, head, replaced


# --- Few-shot readouts (moved from scripts/fewshot_ref.py). ---

class Token0BSRIClassifier(BSRIClassifier):
    """Read out token 0 instead of the mean over all tokens.

    This is the BSR-VideoMAE reference's readout (``classifier(sequence_output[:, 0])``).
    VideoMAE here runs with ``use_mean_pooling=False``, so token 0 is *not* a CLS
    token -- it is patch 0 of the first frame -- and mean pooling over all 1568
    tokens is the readout every earlier number in this project used.  The
    difference is not cosmetic: mean pooling averages over a representation in
    which the reference's own probing found the class signal concentrated, so the
    two readouts are the top candidate for the 0.47-vs-0.58 gap.
    """

    def forward(self, eeg: Tensor, channel_names: Sequence[str] | None = None) -> Tensor:
        _, adapted = self.encoder.render_and_adapt(eeg, channel_names)
        normalized = ((adapted - self.encoder.image_mean) / self.encoder.image_std).permute(0, 2, 1, 3, 4)
        hidden = self.encoder.videomae(normalized).last_hidden_state
        return self.head(hidden[:, 0])


class FeatureStandardizer(nn.Module):
    """Per-dimension feature standardization that tolerates a batch of one.

    Drop-in replacement for ``BatchNorm1d(affine=False, track_running_stats=False)``
    -- for ``N >= 2`` it computes exactly the same ``(x - mean) / sqrt(var + eps)``
    -- but it also carries an exponential moving average of the statistics and
    falls back to it when only one sample is in the batch.

    That fallback is needed because per-channel batch statistics are *undefined*
    for a single sample: torch raises ``Expected more than 1 value per channel
    when training``.  Splits whose size is ``1 (mod batch_size)`` -- ShuMI's test
    split is 193 files against batch 8 -- end with such a batch, and the
    evaluation loaders deliberately do not ``drop_last`` (dropping a trial would
    silently shrink the test set).  The fallback affects at most one sample per
    split; every other batch takes the exact batch-statistics path.
    """

    def __init__(self, features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_var", torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] < 2:
            mean, var = self.running_mean, self.running_var
        else:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * var)
        return (x - mean) / torch.sqrt(var + self.eps)


# --- Multi-task heads (moved from scripts/pretrain_supervised.py). ---

class TaskHeads(nn.Module):
    """One ``Linear(768, num_classes)`` head per source dataset."""

    def __init__(self, hidden_size: int, task_classes: dict[str, int], dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, num_classes)
            for task, num_classes in sorted(task_classes.items())
        })

    def forward(self, features: torch.Tensor, task: str) -> torch.Tensor:
        return self.heads[task](self.dropout(features))


# --- Masking helper (moved from scripts/pretrain.py). ---

def masked_positions(model, batch_size: int, ratio: float, device: torch.device) -> torch.Tensor:
    config = model.videomae.config
    count = (config.num_frames // config.tubelet_size) * (config.image_size // config.patch_size) ** 2
    masked = int(round(count * ratio))
    noise = torch.rand(batch_size, count, device=device)
    indices = noise.argsort(dim=1)[:, :masked]
    result = torch.zeros(batch_size, count, dtype=torch.bool, device=device)
    return result.scatter_(1, indices, True)
