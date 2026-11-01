"""Checkpoint adapters for the two legacy Lightning checkpoints and BSR-I files.

Two checkpoint *families* live here and they are not interchangeable:

* the **integrated** family (``stage3``/``finetune``) stores one ``model_state``
  for a whole ``BSRIEncoder``/``BSRIClassifier``, handled by
  :func:`load_bsri_backbone`;
* the **compact** family (``pretrain_mae_ref``/``pretrain_psd``) stores the
  renderer, adapter and encoder as separate keys with the LoRA matrices either
  merged into ``encoder_state`` or kept apart in ``lora_state``.  Rebuilding one
  requires knowing which adapter *and* which value mode the run used, which is
  what :func:`build_bsri_modules_from_compact_ckpt` owns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import VideoMAEConfig, VideoMAEModel

from .model import (
    BSRenderer,
    RenderToVideoAdapter,
    ResizeCLAHEAdapter,
    ResizeOnlyAdapter,
    inject_videomae_lora,
    load_lora_state_dict,
)
import json


def _state_dict(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} is not a mapping")
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path} does not contain a state dictionary")
    return state, checkpoint


def _load_checked(module: nn.Module, weights: dict[str, torch.Tensor], description: str, strict: bool = True) -> None:
    if not weights:
        raise ValueError(f"No {description} weights were found in the checkpoint")
    result = module.load_state_dict(weights, strict=strict)
    if not strict and len(result.missing_keys) == len(module.state_dict()):
        raise ValueError(f"No {description} tensor matched the target module")


def load_renderer_checkpoint(renderer: nn.Module, path: str | Path, strict: bool = True) -> None:
    """Load both halves of the released renderer: ``eeg2pix`` and ``pix2eeg``.

    The checkpoint carries a matched pair -- ``eeg2pix`` renders the |STFT| grid to
    RGB and ``pix2eeg`` inverts it -- so the inverse head must be restored too.
    Older renderer checkpoints predate ``pix2eeg``; then only ``eeg2pix`` loads.
    """
    state, _ = _state_dict(path)
    for name in ("eeg2pix", "pix2eeg"):
        target = getattr(renderer, name, None)
        if target is None:
            continue
        prefixes = (f"model.{name}.", f"renderer.{name}.", f"encoder.renderer.{name}.", f"{name}.")
        weights: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    weights[key[len(prefix):]] = value
                    break
        if not weights:
            continue
        _load_checked(target, weights, f"BSR renderer {name}", strict=strict)


def load_legacy_videomae(pretraining_model: nn.Module, path: str | Path, strict: bool = True) -> None:
    """Load ``vmae-unsp-mix-e49.ckpt`` into VideoMAEForPreTraining."""
    state, _ = _state_dict(path)
    weights = {key[len("model."):]: value for key, value in state.items() if key.startswith("model.")}
    if not weights and any(key.startswith("videomae.") for key in state):
        weights = dict(state)
    _load_checked(pretraining_model, weights, "VideoMAE pretraining", strict=strict)


def load_legacy_videomae_encoder(encoder: nn.Module, path: str | Path, strict: bool = True) -> None:
    state, _ = _state_dict(path)
    prefixes = ("model.videomae.", "videomae.videomae.", "encoder.videomae.")
    weights: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                weights[key[len(prefix):]] = value
                break
    _load_checked(encoder, weights, "VideoMAE encoder", strict=strict)


def load_bsri_backbone(backbone: nn.Module, path: str | Path, strict_adapter: bool = True) -> dict[str, Any]:
    """Load renderer, adapter and encoder from a BSR-I pretraining checkpoint."""
    state, checkpoint = _state_dict(path)
    renderer = {key[len("renderer."):]: value for key, value in state.items() if key.startswith("renderer.")}
    adapter = {key[len("adapter."):]: value for key, value in state.items() if key.startswith("adapter.")}
    videomae = {key[len("videomae.videomae."):]: value for key, value in state.items() if key.startswith("videomae.videomae.")}
    # Also accept checkpoints saved from BSRIEncoder or BSRIClassifier.
    if not renderer:
        renderer = {key[len("encoder.renderer."):]: value for key, value in state.items() if key.startswith("encoder.renderer.")}
    if not adapter:
        adapter = {key[len("encoder.adapter."):]: value for key, value in state.items() if key.startswith("encoder.adapter.")}
    if not videomae:
        videomae = {key[len("encoder.videomae."):]: value for key, value in state.items() if key.startswith("encoder.videomae.")}
    _load_checked(backbone.renderer, renderer, "BSR-I renderer", strict=False)
    _load_checked(backbone.adapter, adapter, "BSR-I adapter", strict=strict_adapter)
    _load_checked(backbone.videomae, videomae, "BSR-I VideoMAE encoder", strict=True)
    return checkpoint


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: dict[str, Any],
    **metadata: Any,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "epoch": epoch,
        "config": config,
        **metadata,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, destination)


#: Legacy BSR renderer (a frozen TUEG spectral -> RGB map) and the VideoMAE
#: encoder it was paired with.  Both are only the *initialisation* of a compact
#: checkpoint's backbone; the checkpoint itself overrides them.
REFERENCE_CHECKPOINTS = Path(__file__).resolve().parents[1].parent / "BSR-VideoMAE" / "checkpoints"
DEFAULT_RENDERER_CKPT = REFERENCE_CHECKPOINTS / "BSR_fft_abs_TUEG_RS2240000.ckpt"
DEFAULT_VMAE_CKPT = REFERENCE_CHECKPOINTS / "vmae-unsp-mix-e49.ckpt"


def adapter_from_compact_ckpt(checkpoint: dict[str, Any]) -> nn.Module:
    """Rebuild the adapter a compact checkpoint was trained with.

    ``adapter_mode`` decides the class and ``adapter_state`` its weights, so the
    two must be read together: building a ``ResizeCLAHEAdapter`` for a checkpoint
    whose ``adapter_mode`` was ``render2video`` does not fail loudly -- it loads
    nothing or the wrong tensors and silently hands the encoder a different
    input distribution than the run was trained on.  That is why every consumer
    builds the model through this function instead of hard-coding an adapter.

    An empty ``adapter_state`` (or ``adapter_mode == "resize"``) means the
    parameter-free :class:`~bsri.model.ResizeOnlyAdapter`, whose learned
    adaptation lives in the LoRA matrices instead.
    """
    config = checkpoint.get("config", {})
    adapter_state = checkpoint.get("adapter_state") or {}
    if hasattr(config, "to_dict"):
        # Newer checkpoints serialise the encoder config as a transformers VideoMAEConfig
        # object rather than a plain dict.  Every probe path rebuilds the model through
        # here, so accept both: without this, probing a freshly produced checkpoint dies
        # with "AttributeError: VideoMAEConfig object has no attribute get".
        config = config.to_dict()
    if "adapter_mode" not in config:
        # Newer checkpoints serialise the encoder config as a transformers VideoMAEConfig,
        # whose to_dict() carries only the HF fields, so the BSR keys that describe the
        # adapter are missing and every lookup falls back to its default -- which is how a
        # 16-channel RenderToVideoAdapter gets rebuilt as a 32-channel one and loading then
        # fails with size mismatches.  Recover them from the saved adapter state instead.
        _adapter_state = checkpoint.get("adapter_state") or {}
        _stem = _adapter_state.get("stem.weight")
        if _stem is not None:
            config["adapter_mode"] = "render2video"
            config["adapter_channels"] = int(_stem.shape[0])
        else:
            config["adapter_mode"] = "resize"
    adapter_mode = config.get("adapter_mode")
    if adapter_mode == "resize" or not adapter_state:
        return ResizeOnlyAdapter(output_size=224, value_mode=config.get("value_mode", "minmax"))
    if adapter_mode == "render2video":
        adapter = RenderToVideoAdapter(
            output_size=224, hidden_channels=int(config.get("adapter_channels", 32))
        )
        adapter.load_state_dict(adapter_state, strict=True)
        return adapter
    adapter = ResizeCLAHEAdapter(
        hidden_channels=int(config.get("adapter_channels", 32)),
        # Absent in pre-2026-09 checkpoints, which were built with the flip on;
        # every adapter in active use is flip-free, so default to False.
        flip_channels=bool(config.get("adapter_flip", False)),
    )
    adapter.load_state_dict(adapter_state, strict=True)
    return adapter


def build_bsri_modules_from_compact_ckpt(
    checkpoint: dict[str, Any],
    system: str = "1010",
    renderer_ckpt: str | Path = DEFAULT_RENDERER_CKPT,
    vmae_ckpt: str | Path = DEFAULT_VMAE_CKPT,
    renderer_from_checkpoint: bool = False,
) -> tuple[nn.Module, nn.Module, VideoMAEModel]:
    """``(renderer, adapter, videomae)`` for a compact ref/PSD checkpoint.

    The encoder arrives either fully merged (``encoder_state``) or as a pristine
    VideoMAE plus LoRA matrices to inject (``lora_state``); a checkpoint with
    neither is rejected rather than silently probed with random LoRA.

    Callers wrap the three in whatever they need (a bare encoder for the frozen
    probe, a :class:`~bsri.model.BSRIClassifier` for fine-tuning), so the frozen
    probe and a downstream fine-tune are guaranteed to see the same network.

    ``renderer_from_checkpoint`` reads the renderer out of ``renderer_state``
    instead of the fixed TUEG initialisation.  It defaults to ``False`` only to
    keep older results comparable; a checkpoint trained with
    ``tuning_renderer=True`` (every ``pretrain_psd`` run from
    ``psd_channel_23_4way_r2v20`` onward) carries a renderer that moved away from
    the TUEG map, so loading the initialisation instead evaluates a network that
    was never trained.  The two renderers differ by ~0.10 relative on the source
    images, which is far larger than the downstream effects being measured.
    """
    config = checkpoint.get("config", {})
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if "adapter_mode" not in config:
        _ad = (checkpoint.get("adapter_state") or {})
        _stem = _ad.get("stem.weight")
        if _stem is not None:
            config["adapter_mode"] = "render2video"
            config["adapter_channels"] = int(_stem.shape[0])
        else:
            config["adapter_mode"] = "resize"
    renderer = BSRenderer(system=system)
    if renderer_from_checkpoint and checkpoint.get("renderer_state"):
        state = dict(checkpoint["renderer_state"])
        missing = [key for key in renderer.state_dict() if key not in state]
        extra = [key for key in state if key not in renderer.state_dict()]
        if missing or extra:
            # Checkpoints written before the inverse head existed carry only eeg2pix;
            # the new pix2eeg weights come from --renderer-ckpt instead.
            print(json.dumps({"renderer_state_missing": missing[:8],
                              "missing_count": len(missing), "extra_count": len(extra)}),
                  flush=True)
        renderer.load_state_dict(state, strict=False)
    else:
        load_renderer_checkpoint(renderer, renderer_ckpt)
    adapter = adapter_from_compact_ckpt(checkpoint)
    videomae = VideoMAEModel(VideoMAEConfig(image_size=224, num_frames=16, use_mean_pooling=False))
    load_legacy_videomae_encoder(videomae, vmae_ckpt)
    if checkpoint.get("encoder_state") is not None:
        videomae.load_state_dict(checkpoint["encoder_state"], strict=True)
    elif checkpoint.get("lora_state"):
        inject_videomae_lora(
            videomae,
            rank=int(config.get("lora_rank", 8)),
            alpha=float(config.get("lora_alpha", 16.0)),
            dropout=float(config.get("lora_dropout", 0.0)),
        )
        load_lora_state_dict(videomae, checkpoint["lora_state"])
    else:
        raise ValueError("checkpoint carries neither encoder_state nor lora_state")
    return renderer, adapter, videomae
