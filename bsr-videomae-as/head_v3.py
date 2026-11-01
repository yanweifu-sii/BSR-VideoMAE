"""v3 heads (v3-plan §5.5): classification readouts + spectral reconstruction.

Classification heads (feature -> logits; readout of the token sequence is a
separate choice, see :func:`readout_features`):
- ``mean``: LayerNorm -> Dropout -> Linear (v2 default) or MLP tail;
- norm slot in {none, running, batch} precedes LayerNorm (v2 semantics from
  ``ops/experiment.py::head``), selected by validation pre-registration later.

Reconstruction heads:
- ``SpectrumDecoderHead`` (decoder-head; alias ``SpectrumReconHead``): MAE-decoder
  patches -> per-**frame** ``[69, 5]``
  log10 band spectrum (G0 test target/loss definition, see module docstring
  of ``ops/pretrain_v3.py`` for the objective this serves).
- v1-native bypass uses ``ChannelPSDHead``/``PSDHead`` from bsri (imported in
  ``ops/pretrain_v3.py``), untouched.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import math

import torch
from torch import nn

from bsri.model import (FeatureStandardizer, N_BANDS, SYSTEM_CHANNELS,
                        channel_grid_coordinates)

CLASS_HEAD_KINDS = ('linear', 'mlp')
CLASS_NORMS = ('none', 'running', 'batch')
READOUTS = ('mean', 'token0')


class BatchStandardizer(nn.Module):
    """Per-dimension batch standardization (v2 ``ops/experiment.py`` copy)."""

    def forward(self, x):
        return (x - x.mean(0, keepdim=True)) / x.std(0, unbiased=False, keepdim=True).clamp_min(1e-5)


def classifier_head(hidden, classes, kind='linear', norm='none', dropout=0.1):
    """v2-compatible classification head with a pluggable normalization slot."""
    if kind not in CLASS_HEAD_KINDS:
        raise ValueError(f'unknown head kind {kind!r}; choose from {CLASS_HEAD_KINDS}')
    if norm not in CLASS_NORMS:
        raise ValueError(f'unknown norm {norm!r}; choose from {CLASS_NORMS}')
    modules = []
    if norm == 'running':
        # affine=False + stats tracked: v2's 'running' semantics.
        bn = nn.BatchNorm1d(hidden, affine=False)
        modules.append(bn)
    elif norm == 'batch':
        # DEV-4: v1's standardizer semantics with a running-stat fallback for
        # size-1 tail batches (shumi 193 % 24 == 1 would otherwise collapse
        # every feature of the batch to zero).
        modules.append(FeatureStandardizer(hidden))
    modules.append(nn.LayerNorm(hidden))
    if kind == 'mlp':
        modules.extend([nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, classes)])
    else:
        modules.extend([nn.Dropout(dropout), nn.Linear(hidden, classes)])
    return nn.Sequential(*modules)


def readout_features(sequence, readout='mean'):
    """Token-sequence readout: mean pooling (v2) or token-0 (v1 reference)."""
    if readout == 'mean':
        return sequence.mean(dim=1)
    if readout == 'token0':
        return sequence[:, 0]
    raise ValueError(f'unknown readout {readout!r}; choose from {READOUTS}')


class SpectrumDecoderHead(nn.Module):
    """Decoder patches -> per-frame ``[T, 69, 5]`` log10 band spectrum.

    Terminology (F-7): this is the **decoder-head** — it reads the MAE decoder's
    patch outputs (norm-pix un-standardized, restored with ``unfold_patches``),
    NOT encoder tokens and NOT the ``pix2eeg`` inverse (that is the
    ``v1native-inverse`` path, R1 only).

    Definition of target/loss (matches the Lead's T6 ruling and is asserted in
    ``tests/test_v3_pretrain_step.py``):

    - Input ``logits`` are the MAE decoder outputs for ALL patches in
      ``norm_pix`` space; they are un-standardized with the target patches'
      own ``patch_mean``/``patch_std`` and restored to a frame grid with
      ``bsri.pretrain.unfold_patches`` -> ``[B, C, T, H, W]``.
    - The grid is adaptively pooled back to the renderer's native G x G grid
      (pixel layout is preserved by the bilinear resize), the 69 electrode
      slots are gathered with the renderer's ``electrode_index`` (canonical
      channel order), and a shared MLP conditioned on a learnable per-channel
      embedding maps ``[3 + E] -> 5`` bands per frame.
    - Targets are ``bsri.pretrain.spectrum_band_target`` evaluated per FRAME
      on the UNMASKED parked ``renderer.spectrum()`` output (never on the
      masked model input): ``[B, T, 69, 5]`` in log10 space.
    - Loss = mean squared error over MASKED frames only.
    """

    def __init__(self, electrode_index, grid, channels_in=3, embed=16, hidden=128, bands=N_BANDS):
        super().__init__()
        # Registered buffer so `.to(device)` and checkpoints carry it along.
        self.register_buffer('electrode_index', electrode_index.detach().long().clone(),
                             persistent=True)
        self.grid = int(grid)
        self.channels = int(self.electrode_index.numel())
        self.bands = int(bands)
        in_features = int(channels_in) + int(embed)
        self.channel_embed = nn.Parameter(torch.randn(self.channels, embed) * 0.02)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.bands),
        )

    def frames_to_electrodes(self, frames):
        """``[B, C, T, H, W]`` -> per-cell ``[B, T, 69, C]`` on the G x G grid."""
        b, c, t, h, w = frames.shape
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), (self.grid, self.grid))
        pooled = pooled.reshape(b, t, c, self.grid * self.grid)          # [B,T,C,G*G]
        return pooled.index_select(3, self.electrode_index).permute(0, 1, 3, 2)  # [B,T,69,C]

    def forward(self, frames):
        """frames: denormalized decoder reconstruction ``[B, C, T, H, W]``."""
        cells = self.frames_to_electrodes(frames)                        # [B,T,69,3]
        batch, time = cells.shape[:2]
        embed = self.channel_embed.unsqueeze(0).unsqueeze(0).expand(batch, time, -1, -1)
        feats = torch.cat([cells, embed], dim=-1)                        # [B,T,69,3+E]
        return self.net(feats)                                           # [B,T,69,5]


def build_recon_head(renderer, hidden=128, embed=16):
    """Convenience factory bound to a renderer's electrode mapping."""
    return SpectrumDecoderHead(renderer.electrode_index,
                             grid=renderer.output_size // 32, hidden=hidden, embed=embed)


# Legacy alias (pre-F-7 name).
SpectrumReconHead = SpectrumDecoderHead

__all__ = [
    'CLASS_HEAD_KINDS', 'CLASS_NORMS', 'READOUTS',
    'BatchStandardizer', 'classifier_head', 'readout_features',
    'SpectrumDecoderHead', 'SpectrumReconHead', 'build_recon_head',
]
