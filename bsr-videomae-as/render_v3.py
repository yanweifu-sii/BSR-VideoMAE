"""v3 renderer & adapter factories (v3-plan §5.2).

Invariant: every renderer derives from ``bsri.model.BSRenderer`` and uses the
inherited ``spatialize()`` / ``spectrum()`` untouched (G0 asserts bit-identity
across R1/R2/R3 against a plain BSRenderer, i.e. the byte-identical v2 code).

- R1 legacy:  ``BSRenderer`` with its learned ``eeg2pix`` (v1 path; optional
  TUEG checkpoint load keeps ``pix2eeg`` for the v1-native recon bypass).
- R2 spectral: ported from v2 ``ops/spectral_components.py`` (read-only copy):
  ``eeg2pix=Identity``, per-sample log1p calibration, deterministic 1x1
  three-band projection (initialised exactly as v2 did), frozen ``pix2eeg``.
- R3 hybrid:   R2's calibration + deterministic split as a frozen buffer PLUS a
  learnable 1x1 Conv2d(K->3) band-mix delta that is ZERO-initialised, so at
  init R3 output == R2 output bit-for-bit while band mixing can be learned.

Adapters (v3-plan §5.2): resize / sharedscale / spectral / residual /
render2video / clahe - all pure renderer->video conversion, no spatial changes.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from bsri.model import (
    BAND_NAMES,
    GRID_1010,
    BSRenderer,
    RenderToVideoAdapter,
    ResizeCLAHEAdapter,
    ResizeOnlyAdapter,
    SYSTEM_GRIDS,
    SYSTEM_CHANNELS,
)

RENDERER_KINDS = ('r1', 'r2', 'r3')
ADAPTER_KINDS = ('resize', 'sharedscale', 'spectral', 'residual', 'render2video', 'clahe')


def grid_size(system='1010'):
    return len(SYSTEM_GRIDS[system])


class SpectralRenderer(BSRenderer):
    """R2: deterministic three-band projection over the unchanged STFT grid.

    Ported verbatim from v2 ``ops/spectral_components.py`` (read-only source).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Keep inherited inverse module for compatibility, but no old upsampler.
        self.eeg2pix = nn.Identity()
        bins = self.n_fft // 2 + 1
        self.projection = nn.Conv2d(bins, 3, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.zero_()
            # All non-DC bins are represented; no assumed source sampling rate.
            for band, idx in enumerate(torch.tensor_split(torch.arange(1, bins), 3)):
                self.projection.weight[band, idx, 0, 0] = 1 / len(idx)
        # D8.1: the deterministic three-band projection is FROZEN (never trained)
        # and excluded from every optimizer group by ops/pretrain_v3.py.
        self.projection.requires_grad_(False)
        self.pix2eeg.requires_grad_(False)

    def _calibrated(self, tokens):
        """Per-example log1p calibration shared by R2/R3 (v2 semantics)."""
        b, t, f, h, w = tokens.shape
        power = tokens.float().square()
        scale = power.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-12)
        return torch.log1p(power / scale).reshape(b * t, f, h, w)

    def render_tokens(self, tokens):
        z = self._calibrated(tokens)
        rgb = self.projection(z)
        rgb = F.interpolate(rgb, size=(self.output_size, self.output_size),
                            mode='bilinear', align_corners=False)
        b, t = tokens.shape[0], tokens.shape[1]
        return rgb.reshape(b, t, 3, self.output_size, self.output_size).permute(0, 2, 1, 3, 4)

    def forward(self, eeg, channel_names=None):
        return self.render_tokens(self.spatialize(eeg, channel_names))


class HybridRenderer(SpectralRenderer):
    """R3: R2 pipeline + learnable 1x1 band-mix delta, zero-init == R2.

    ``rgb = deterministic_split(z) + learned_mix(z)`` with ``learned_mix``
    zero-initialised: at init the output is bit-identical to R2, and the
    gradient that reaches ``learned_mix.weight`` is exactly the residual
    mixing v3 wants to learn on top of the deterministic projection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        bins = self.n_fft // 2 + 1
        # Frozen copy of v2's deterministic projection weights.
        self.register_buffer('base_mix', self.projection.weight.detach().clone(), persistent=True)
        self.projection.requires_grad_(False)  # superseded by base_mix + learned_mix
        self.learned_mix = nn.Conv2d(bins, 3, 1, bias=False)
        nn.init.zeros_(self.learned_mix.weight)

    @property
    def zero_initialised(self):
        return bool(torch.count_nonzero(self.learned_mix.weight).item() == 0)

    def render_tokens(self, tokens):
        z = self._calibrated(tokens)
        rgb = F.conv2d(z, self.base_mix) + self.learned_mix(z)
        rgb = F.interpolate(rgb, size=(self.output_size, self.output_size),
                            mode='bilinear', align_corners=False)
        b, t = tokens.shape[0], tokens.shape[1]
        return rgb.reshape(b, t, 3, self.output_size, self.output_size).permute(0, 2, 1, 3, 4)


# --- Adapters: faithful ports (v2 files are read-only references). ---

class SharedScaleAdapter(nn.Module):
    """Ported from v2 ``ops/spectral_sharedscale.py`` (one scale per clip)."""

    def __init__(self, output_size=224):
        super().__init__()
        self.output_size = output_size
        self.log_gain = nn.Parameter(torch.zeros(1, 3, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1, 1))

    def forward(self, video):
        # One scale across RGB preserves between-projection moments.
        x = video.float()
        mean = x.mean((1, 2, 3, 4), keepdim=True)
        std = x.var((1, 2, 3, 4), unbiased=False, keepdim=True).add(1e-6).sqrt()
        x = torch.sigmoid((x - mean) / std * self.log_gain.exp().clamp(max=10) + self.bias)
        b, c, t, h, w = x.shape
        z = F.interpolate(x.transpose(1, 2).reshape(b * t, c, h, w), size=(self.output_size, self.output_size),
                          mode='bilinear', align_corners=False, antialias=True)
        return z.reshape(b, t, c, self.output_size, self.output_size).transpose(1, 2)


class SpectralAdapter(nn.Module):
    """Ported from v2 ``ops/spectral_components.py`` (per-channel clip scale)."""

    def __init__(self, output_size=224):
        super().__init__()
        self.output_size = output_size
        self.log_gain = nn.Parameter(torch.zeros(1, 3, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1, 1))

    def forward(self, video):
        # Normalize a complete clip, never independent frames or examples.
        x = video.float()
        mean = x.mean((2, 3, 4), keepdim=True)
        std = x.var((2, 3, 4), unbiased=False, keepdim=True).add(1e-6).sqrt()
        x = torch.sigmoid((x - mean) / std * self.log_gain.exp().clamp(max=10) + self.bias)
        b, c, t, h, w = x.shape
        z = F.interpolate(x.transpose(1, 2).reshape(b * t, c, h, w), size=(self.output_size, self.output_size),
                          mode='bilinear', align_corners=False, antialias=True)
        return z.reshape(b, t, c, self.output_size, self.output_size).transpose(1, 2)


class ResidualAdapter(nn.Module):
    """Ported from v2 ``ops/experiment.py``: starts identical to ResizeOnly."""

    def __init__(self, output_size=224):
        super().__init__()
        self.base = ResizeOnlyAdapter(output_size=output_size, value_mode='minmax')
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 3, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        x = self.base(x)
        b, c, t, h, w = x.shape
        z = x.transpose(1, 2).reshape(b * t, c, h, w)
        z = (z + 0.1 * self.net(z)).clamp(0, 1)
        return z.reshape(b, t, c, h, w).transpose(1, 2)


def make_renderer(kind, system='1010', hop_length=128, renderer_ckpt=None):
    """Build an R1/R2/R3 renderer; ``renderer_ckpt`` only loads eeg2pix/pix2eeg weights."""
    if kind not in RENDERER_KINDS:
        raise ValueError(f'unknown renderer {kind!r}; choose from {RENDERER_KINDS}')
    if kind == 'r1':
        renderer = BSRenderer(system=system, hop_length=hop_length)
        if renderer_ckpt:
            from bsri.checkpoints import load_renderer_checkpoint
            load_renderer_checkpoint(renderer, renderer_ckpt)
        return renderer
    if kind == 'r2':
        renderer = SpectralRenderer(system=system, hop_length=hop_length)
        if renderer_ckpt:
            raise ValueError('R2/R3 do not consume legacy renderer checkpoints (eeg2pix is Identity)')
        return renderer
    renderer = HybridRenderer(system=system, hop_length=hop_length)
    if renderer_ckpt:
        raise ValueError('R2/R3 do not consume legacy renderer checkpoints (eeg2pix is Identity)')
    return renderer


def make_adapter(kind, output_size=224, value_mode='minmax'):
    if kind not in ADAPTER_KINDS:
        raise ValueError(f'unknown adapter {kind!r}; choose from {ADAPTER_KINDS}')
    if kind == 'resize':
        return ResizeOnlyAdapter(output_size=output_size, value_mode=value_mode)
    if kind == 'sharedscale':
        return SharedScaleAdapter(output_size=output_size)
    if kind == 'spectral':
        return SpectralAdapter(output_size=output_size)
    if kind == 'residual':
        return ResidualAdapter(output_size=output_size)
    if kind == 'render2video':
        return RenderToVideoAdapter(output_size=output_size, hidden_channels=16)
    return ResizeCLAHEAdapter(hidden_channels=16)


__all__ = [
    'RENDERER_KINDS', 'ADAPTER_KINDS', 'SpectralRenderer', 'HybridRenderer',
    'SharedScaleAdapter', 'SpectralAdapter', 'ResidualAdapter',
    'make_renderer', 'make_adapter', 'grid_size',
    'BSRenderer', 'SYSTEM_CHANNELS', 'GRID_1010', 'BAND_NAMES',
]
