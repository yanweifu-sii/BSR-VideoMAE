"""BSR-VideoMAE-AS 4-bench finetune runner (bcic2a/seed/seedvii/shumi).

Ported from the BSR-VideoMAE-AS internal `ops/experiment_v3.py`; supports the four
public protocols used in the paper: bcic2a/seedvii fixed (no official val) and
seed/shumi strict (official val early-stop/selection). Search/probe stages are kept
for compatibility but the delivered scripts only use `fixed`/`strict`.

Ported from clean-v2 ``ops/experiment_spectral.py`` (read-only reference) with:
- v3 compact checkpoints (renderer/adapter kinds from provenance.v3, pix2eeg
  already stripped for R2/R3); rebuilt through ``ops/render_v3.py`` factories;
- A2 input rule: renderer hop = ``stft_hop_for_frames(length, 16)`` (equals the
  historical ``window/15`` for 800/2000-sample corpora);
- protocols P-v1 / P-v2 (--protocol with preset enforcement, D8/G2);
- fail-closed test guard: only ``stage in ('final','strict','fixed')`` may touch
  ``split['test']``;
- strict (D54): train rows only, validation used for early-stop/selection only
  (no parameter updates), test once from best.pt;
- fixed (D56): train rows only, NO validation, fixed epoch count, select=last,
  test once - for corpora whose raw split has an empty val/ directory;
- final lock validation against ``manifests/final_v3_lock.json``.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from transformers import VideoMAEConfig, VideoMAEModel

from bsri.data import EEGPickleDataset
from bsri.model import BSRIEncoder, inject_videomae_lora, LoRALinear, stft_hop_for_frames
from head_v3 import classifier_head
from render_v3 import make_adapter, make_renderer

BASE = Path(__file__).resolve().parent
SPLITS_ROOT = None   # v17/D42: override via --splits-root (candidate dev-session splits)
CORPORA = {'bcic2a': 9, 'shumi': 25, 'seed': 15, 'seedvii': 20}
LENGTHS = {'bcic2a': 800, 'shumi': 800, 'seed': 2000, 'seedvii': 2000}

# P-v1: v1 frozen protocol (clean-v1 eval_four_bench.sh, dossier §3).
# v3.1: faithfully ported v2 recipe knobs (read-only port from
# bsr-clean-v2/ops/time_shift.py sha 09d7... and ops/balanced_loss.py; both tiny
# and semantically pinned below with unit tests).
def _splits_root():
    return SPLITS_ROOT if SPLITS_ROOT is not None else (BASE / 'manifests' / 'splits')


def time_shift(x, fraction=0.1):
    """Training-only circular temporal translation shared across channels."""
    assert x.ndim == 3 and 0 <= fraction <= 0.5
    limit = int(x.shape[-1] * fraction)
    if limit == 0:
        return x
    shifts = torch.randint(-limit, limit + 1, (x.shape[0], 1, 1), device=x.device)
    indices = (torch.arange(x.shape[-1], device=x.device).view(1, 1, -1) - shifts) % x.shape[-1]
    return x.gather(-1, indices.expand_as(x))


def mixup_pair(x, alpha=0.2, generator=None):
    """v16/D40: time-domain input mixup. Returns (mixed_x, lam)."""
    assert x.ndim == 3 and alpha > 0
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if generator is None else \
        float(torch.distributions.Beta(alpha, alpha).sample(generator))
    idx = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1.0 - lam) * x[idx], lam


def ema_update(shadow, model, decay):
    """v16/D40: EMA of trainable (floating) params, in place."""
    with torch.no_grad():
        for name, q in model.state_dict().items():
            if not q.dtype.is_floating_point:
                continue
            shadow[name] = decay * shadow.get(name, q.clone()) + (1.0 - decay) * q.detach()
    return shadow


def ema_swap(model, shadow, direction='apply'):
    """v16/D40: temporarily exchange raw weights with EMA weights (eval with EMA)."""
    saved = {}
    current = model.state_dict()
    for name, q in list(current.items()):
        if name in shadow and q.dtype.is_floating_point:
            saved[name] = q.clone()
            buf = shadow[name].to(q.dtype)
            current[name].copy_(buf)
    return saved


def circular_shift(x, shift):
    """Deterministic circular time shift (per example, shared across channels)."""
    assert x.ndim == 3
    if shift == 0:
        return x
    indices = (torch.arange(x.shape[-1], device=x.device).view(1, 1, -1) - shift) % x.shape[-1]
    return x.gather(-1, indices.expand_as(x))


def pseudo_label_select(probs, thr=0.9, min_per_class=8, max_per_class=200):
    """v15/D38: high-confidence pseudo-labels from UNLABELED target predictions.
    Returns (indices, labels) or (None, None) when any class is under-covered
    (fail-safe: no adaptation in that case)."""
    conf, labels = probs.max(-1)
    keep = conf >= thr
    idx = torch.nonzero(keep).squeeze(-1)
    if idx.numel() == 0:
        return None, None
    chosen = []
    classes = labels[idx]
    for c in classes.unique().tolist():
        cc = idx[classes == c]
        if cc.numel() < min_per_class:
            return None, None
        chosen.append(cc[:max_per_class])
    out = torch.cat(chosen)
    return out, labels[out]


def selftrain_adapt(model, data, device, args, ea_w=None, thr=0.9, lr=1e-4, epochs=1):
    """v15/D38 pseudo-label self-training: fine-tune a DEEP-COPY model's head on
    HIGH-CONFIDENCE pseudo-labels from UNLABELED target windows (real labels never
    used). Returns the adapted model, or None (fail-safe: under-coverage -> no adaptation).
    The copy is discarded by the caller after prediction."""
    import copy as _copy
    model.eval()
    logits_all = []
    with torch.no_grad():
        for x, _y in batches(data, model, device, ea_w):
            logits_all.append(model(x).float().cpu())
    logits = torch.cat(logits_all)
    idx, y_pl = pseudo_label_select(logits.softmax(-1), thr=thr,
                                    min_per_class=getattr(args, 'st_min_per_class', 8),
                                    max_per_class=getattr(args, 'st_max_per_class', 200))
    if idx is None:
        return None
    adapted = _copy.deepcopy(model)
    for q in adapted.head.parameters():
        q.requires_grad_(True)
    opt = torch.optim.AdamW([q for q in adapted.head.parameters()], lr=lr, weight_decay=0.0)
    adapted.head.train()
    # rebuild selected windows from a second pass (loader is re-iterable)
    xs = []
    with torch.no_grad():
        for x, _y in batches(data, model, device, ea_w):
            xs.append(x.cpu())
    allx = torch.cat(xs)[idx.tolist()].to(device)
    ally = y_pl.to(device)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            feats = adapted.encoder(allx)
        logits = adapted.head(feats.reshape(feats.shape[0], -1).float()
                              if feats.ndim > 2 else feats.float())
        loss = torch.nn.functional.cross_entropy(logits, ally)
        assert loss.requires_grad, 'selftrain loss detached (bug)'
        loss.backward()
        opt.step()
    adapted.eval()
    return adapted


def coral_affine(src_f, tgt_f, eps=1e-5):
    """v15/D38 CORAL: least-squares affine mapping of TARGET features onto SOURCE
    second-order statistics (label-free; only unlabeled target inputs used)."""
    mu_s, mu_t = src_f.mean(0), tgt_f.mean(0)
    cs = torch.cov(src_f.T) + eps * torch.eye(src_f.shape[1], device=src_f.device)
    ct = torch.cov(tgt_f.T) + eps * torch.eye(tgt_f.shape[1], device=tgt_f.device)
    vals_s, vecs_s = torch.linalg.eigh(cs)
    vals_t, vecs_t = torch.linalg.eigh(ct)
    cs_half = vecs_s @ torch.diag_embed(vals_s.clamp_min(1e-12).pow(0.5)) @ vecs_s.transpose(-1, -2)
    ct_inv_half = vecs_t @ torch.diag_embed(vals_t.clamp_min(1e-12).rsqrt()) @ vecs_t.transpose(-1, -2)
    a = cs_half @ ct_inv_half                      # maps tgt cov -> src cov
    return a, mu_t, mu_s


class _FeatureAffine(torch.nn.Module):
    """Temporary head-input wrapper applying f' = (f - mu_t) A^T + mu_s."""
    def __init__(self, head, a, mu_t, mu_s):
        super().__init__()
        self.head, self.a, self.mu_t, self.mu_s = head, a, mu_t, mu_s
    def forward(self, f):
        return self.head((f - self.mu_t) @ self.a.transpose(-1, -2) + self.mu_s)


def coral_fit_transform(model, source_data, target_data, device, ea_w=None):
    """Collect head-input features for SOURCE (train) and TARGET (unlabeled) batches,
    compute the CORAL affine, install it on model.head, and RETURN the restore fn."""
    captured = {}
    handle = model.head.register_forward_hook(
        lambda _m, _in, _out: captured.__setitem__('f', _in[0].detach()))   # HEAD INPUT features
    feats = []
    def _collect(data, w):
        out = []
        model.eval()
        with torch.no_grad():
            for x, _y in batches(data, model, device, w):
                model(x)
                f = captured.get('f')
                if f is not None:
                    out.append(f.reshape(f.shape[0], -1).float().cpu())
        return torch.cat(out) if out else None
    fs = _collect(source_data, ea_w)
    ft = _collect(target_data, ea_w)
    handle.remove()
    assert fs is not None and ft is not None and fs.shape[1] == ft.shape[1],         'CORAL feature collection failed'
    a, mu_t, mu_s = coral_affine(fs, ft)
    original = model.head
    model.head = _FeatureAffine(original, a.to(device), mu_t.to(device), mu_s.to(device))
    return lambda: setattr(model, 'head', original)


def set_bn_train(model, enabled=True):
    """v15/D38 T-BN: put ONLY BatchNorm modules in train mode (running-stat refresh);
    everything else stays frozen in eval. Returns the changed module list."""
    changed = []
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
            module.train(enabled)
            changed.append(module)
    return changed


def tbn_refresh(model, data, device, ea_w=None):
    """Single no-grad pass over the TARGET (unlabeled) batches to refresh BN stats."""
    set_bn_train(model, True)
    model.eval()          # eval first...
    changed = set_bn_train(model, True)  # ...then flip ONLY BN back to train
    try:
        with torch.no_grad():
            for batch in batches(data, model, device, ea_w):
                x = batch[0]
                model(x)
    finally:
        set_bn_train(model, False)
    return len(changed)


def ea_session_key(record):
    """v14/D36: session identifier from a sample path (ses-XX)."""
    import re as _re
    m = _re.search(r'ses[-_]\d+', str(record.get('path', '')))
    return m.group(0).replace('_', '-') if m else 'no-session'


def ea_ridge_scale(lam, mode, c):
    """D-1: tr/C^2 (original pin) vs D31: tr/C (numerical-stability fix)."""
    return lam / c if mode != 'tr_over_C' else lam


def ea_whitening_matrix(train_x, lam=1e-5, mode='tr_over_C2'):
    """Euclidean Alignment per SUBJECT: R = mean(XX^T) pooled over all provided
    TRAIN windows (channels x channels), regularized, W = R^{-1/2} ([C, C]).
    No test statistics are ever involved."""
    assert train_x.ndim == 3, 'expected [B, C, T]'
    b, c, t = train_x.shape
    r = torch.bmm(train_x, train_x.transpose(1, 2)).mean(0) / t
    ridge = ea_ridge_scale(lam, mode, c) * r.diagonal().mean().detach()
    r = r + ridge * torch.eye(c, device=r.device, dtype=r.dtype)
    vals, vecs = torch.linalg.eigh(r)
    vals = vals.clamp_min(1e-12)
    return vecs @ torch.diag_embed(vals.rsqrt()) @ vecs.transpose(-1, -2)


def ea_audit_from_loader(batch_iter, w, device, lam=1e-5, stage='search', max_windows=4096, mode='tr_over_C2'):
    """D30-errata-2 audit fields: W sha, rank/cond of regularized R~, inverse-sqrt
    eigenvalue profile (min/max/mean), and a whitened-covariance sample profile."""
    with torch.no_grad():
        c = w.shape[0]
        acc, count, xb = None, 0.0, None
        for (xb, _yb) in batch_iter:
            acc = (torch.bmm(xb, xb.transpose(1, 2)).sum(0) if acc is None
                   else acc + torch.bmm(xb, xb.transpose(1, 2)).sum(0))
            count += xb.shape[0] * xb.shape[2]
            if count >= max_windows * xb.shape[2]:
                break
        assert acc is not None and c, 'empty loader for EA audit'
        r = acc / count
        ridge = ea_ridge_scale(lam, mode, c) * r.diagonal().mean()
        r_reg = r + ridge * torch.eye(c, device=r.device, dtype=r.dtype)
        eig = torch.linalg.eigvalsh(r_reg)                 # 1D eigenvalues
        inv_sqrt = eig.clamp_min(1e-12).rsqrt()            # 1D profile [lambda^-1/2]
        z = ea_apply(xb[:64], w)
        cov = torch.einsum('bct,bdt->cd', z, z) / (z.shape[0] * z.shape[2])
        return {
            'lambda_rule': 'D-1: lam*tr/C^2' if mode != 'tr_over_C' else 'D31: lam*tr/C (stability fix)',
            'w_sha256': hashlib.sha256(w.detach().cpu().numpy().tobytes()).hexdigest(),
            'rank': int((eig > 1e-8).sum()), 'channels': int(c),
            'cond_r': round(float(eig.max() / eig.clamp_min(1e-12).min()), 3),
            'inv_sqrt_eig_min': round(float(inv_sqrt.min()), 4),
            'inv_sqrt_eig_max': round(float(inv_sqrt.max()), 4),
            'inv_sqrt_eig_mean': round(float(inv_sqrt.mean()), 4),
            'whitened_diag_mean_sample': round(float(cov.diag().mean()), 4),
            'whitened_diag_min_sample': round(float(cov.diag().min()), 4),
            'whitened_diag_max_sample': round(float(cov.diag().max()), 4),
            'fit_rows': 'full_train' if stage == 'final' else 'train'}


def ea_whitening_from_batches(batch_iter, device, lam=1e-5, mode='tr_over_C2'):
    """Pool R over an entire TRAIN batch iterator (subject statistics), then W."""
    acc, count, c = None, 0.0, None
    for (x, _y) in batch_iter:
        x = x.to(device)
        b, c, t = x.shape
        r = torch.bmm(x, x.transpose(1, 2)).sum(0)
        acc = r if acc is None else acc + r
        count += b * t
    assert acc is not None and c, 'empty train iterator for EA'
    r = acc / count
    ridge = lam * r.diagonal().mean().detach() / c
    r = r + ridge * torch.eye(c, device=r.device, dtype=r.dtype)
    vals, vecs = torch.linalg.eigh(r)
    vals = vals.clamp_min(1e-12)
    return vecs @ torch.diag_embed(vals.rsqrt()) @ vecs.transpose(-1, -2)


def ea_apply(x, w):
    """Apply EA transform (w: [C, C]) to windows (x: [B, C, T]); shape-preserving."""
    return torch.einsum('ij,bjt->bit', w, x)


def average_state_dicts(states):
    """Element-wise mean of equally-shaped fp32 state dicts (SWA / multi-seed)."""
    assert states, 'no states to average'
    keys = states[0].keys()
    out = {}
    for key in keys:
        stacked = torch.stack([s_[key].detach().float() for s_ in states])
        out[key] = stacked.mean(0)
    return out


def eval_logits(model, x, args=None, alt=None):
    """v5 inference: optional TTA (circular-shift logit average, k=3) and optional
    second-model logit average (multi-seed). Both default off."""
    def _one(m):
        if args is None or getattr(args, 'tta_frac', 0.0) <= 0:
            return m(x)
        frac = args.tta_frac
        limit = int(x.shape[-1] * frac)
        if limit == 0:
            return m(x)
        shifts = [0] if getattr(args, 'tta_k', 1) <= 1 else [0, limit, -limit]
        acc = None
        for shift in shifts:
            logits = m(circular_shift(x, shift))
            acc = logits if acc is None else acc + logits
        return acc / len(shifts)
    total = _one(model)
    if alt is not None:
        alts = alt if isinstance(alt, (list, tuple)) else [alt]
        for a in alts:
            total = total + _one(a)
        total = total / (1 + len(alts))
    return total


def class_weight_vector_for(split, stage, corpus=None):
    """Inverse-frequency CE weights from the ACTUAL training rows (full_train at
    final). Returns a CPU float tensor, or None when disabled. Lock equality is
    enforced downstream for final runs (researcher M-1)."""
    rows = training_rows_for_stage(split, stage)
    weights = inverse_class_frequency_weights(rows, split['labels'])
    vec = torch.tensor(weights, dtype=torch.float32)
    if stage == 'final' and corpus is not None:
        lock_path = lock_file()
        if lock_path.is_file():
            expected = json.loads(lock_path.read_text())['tasks'].get(corpus, {}).get('full_train_class_weights')
            if expected is not None and [round(w, 10) for w in expected] != [round(float(w), 10) for w in vec]:
                raise LockError('full_train_class_weights != lock (final stage)')
    return vec


def training_rows_for_stage(split, stage):
    """final trains on full_train (v19 full_train protocol); search/strict/fixed
    train on the train rows only (validation never updates parameters)."""
    return split['full_train'] if stage == 'final' else split['train']


def inverse_class_frequency_weights(rows, labels):
    """v2 balanced_loss.class_weights: total / (n_classes * count) per class."""
    counts = {}
    for row in rows:
        counts[row['label']] = counts.get(row['label'], 0) + 1
    assert set(counts) == set(labels), 'Every class must occur in training'
    total = len(rows)
    return [total / (len(labels) * counts[label]) for label in labels]


PROTOCOL_PV1 = {'tuning': 'full', 'head': 'linear', 'norm': 'batch',
                'time_shift': 0.0, 'class_weight': False,
                'batch': 24, 'accumulate': 6, 'epochs': 25, 'schedule_epochs': 25,
                'lr': 1e-5, 'encoder_lr': 1e-5, 'adapter_lr': 1e-5, 'renderer_lr': 1e-5,
                'select': 'last', 'seed': 4523}
# P-v2 presets come from the locked per-corpus v2 configs (filled at G2 time).
PROTOCOL_PV2_PRESET = Path('manifests/protocol_pv2_preset.json')


class LockError(RuntimeError):
    pass


#: Protocol presets may only touch these fields (DEV-1 fail-closed whitelist).
PROTOCOL_KEYS = frozenset({'tuning', 'head', 'norm', 'batch', 'accumulate', 'epochs', 'time_shift', 'class_weight',
                           'schedule_epochs', 'lr', 'encoder_lr', 'adapter_lr',
                           'renderer_lr', 'select', 'seed'})
UNIMPLEMENTED_KNOBS = frozenset({'mixup', 'label_smoothing'})  # v3.1: first two implemented


def dump(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(p)


LOCK_PATH = None


def lock_file():
    return LOCK_PATH if LOCK_PATH is not None else (BASE / 'manifests' / 'final_v3_lock.json')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def seed_all(seed=4523):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def protocol_preset(args):
    """Resolve the locked protocol preset for --protocol; fail-closed on pv2
    until manifests/protocol_pv2_preset.json exists (G2 fills it)."""
    if args.protocol == 'none':
        return {}
    if args.protocol == 'pv1':
        return dict(PROTOCOL_PV1)
    preset_path = BASE / PROTOCOL_PV2_PRESET
    if not preset_path.exists():
        raise LockError(f'--protocol pv2 requires {preset_path} (pre-registered at G2)')
    presets = json.loads(preset_path.read_text())['protocols']
    if args.corpus not in presets:
        raise LockError(f'no pv2 preset for corpus {args.corpus!r}')
    preset = dict(presets[args.corpus])
    banned = sorted(set(preset) & UNIMPLEMENTED_KNOBS)
    if banned:
        raise LockError(f'pv2 preset uses unimplemented knobs {banned}; '
                        f'implement them in experiment_v3.train() before G2 use')
    unknown = sorted(set(preset) - PROTOCOL_KEYS)
    if unknown:
        raise LockError(f'pv2 preset keys outside the protocol whitelist: {unknown}')
    return preset


def apply_protocol(args, preset, explicit=()):
    """Preset fills every protocol field the operator did NOT pass explicitly
    on the command line (explicit flags always win)."""
    unknown = sorted(set(preset) - PROTOCOL_KEYS)
    if unknown:
        raise LockError(f'protocol preset keys outside the whitelist: {unknown}')
    applied = {}
    for key, value in preset.items():
        if key in explicit:
            continue
        setattr(args, key, value)
        applied[key] = value
    return applied


TEST_STAGES = ('final', 'strict', 'fixed')   # stages allowed to read split['test']


def validate_stage_protocol(args):
    """Fail-closed stage/selection contract (D54 strict, D56 fixed)."""
    if args.stage == 'strict':
        assert args.select == 'best', 'strict stage requires --select best'
    elif args.stage == 'fixed':
        assert getattr(args, 'corpus', None) in ('bcic2a', 'seedvii'), \
            'fixed stage is only for the corpora without a val/ directory ' \
            '(bcic2a, seedvii); seed/shumi use --stage strict'
        assert args.select == 'last', \
            'fixed stage requires --select last (no validation rows for early stop)'
        assert not getattr(args, 'patience', 0), \
            'fixed stage has no validation; --patience must be 0'


class TestGuardedSplit(dict):
    """Fail-closed test guard: reading ``split['test']`` outside a test stage
    (final/strict/fixed) raises instead of leaking test rows into a search run."""

    def __init__(self, data, allow_test):
        super().__init__(data)
        self._allow_test = bool(allow_test)

    def __getitem__(self, key):
        if key == 'test' and not self._allow_test:
            raise AssertionError(f'stage attempted to read split["test"] (stage guard)')
        return super().__getitem__(key)


READOUTS = ('mean', 'token0', 'attn')


def pool_tokens(sequence, readout='mean', attn_linear=None):
    """D70 token-sequence readout.

    ``mean``   : mean over tokens (the v2/v3 default; the Model keeps the exact
                 legacy ``BSRIEncoder.forward`` path for bitwise equivalence);
    ``token0`` : first patch token (VideoMAE has no synthetic CLS -> approximate,
                 disclosed);
    ``attn``   : learned softmax attention pooling over ALL patch tokens
                 (``softmax(Linear(token))`` weighted sum), i.e. the "full-token"
                 readout; the linear layer belongs to the head side and is
                 retrained by every FT run (backbone untouched).
    """
    if readout == 'mean':
        return sequence.mean(dim=1)
    if readout == 'token0':
        return sequence[:, 0]
    if readout == 'attn':
        if attn_linear is None:
            raise ValueError('attn readout requires the head-side attention layer')
        weights = torch.softmax(attn_linear(sequence).squeeze(-1), dim=1)
        return (sequence * weights.unsqueeze(-1)).sum(dim=1)
    raise ValueError(f'unknown readout {readout!r}; choose from {READOUTS}')


class Model(nn.Module):
    """v3 checkpoint -> frozen/finetunable encoder + classification head."""

    def __init__(self, ckpt, classes, args, length):
        super().__init__()
        v3 = ckpt.get('v3') or ckpt.get('provenance', {}).get('v3')
        assert v3, 'checkpoint carries no v3 provenance'
        self.renderer_kind = v3['renderer_kind']
        self.adapter_kind = v3['adapter_kind']
        cfg = VideoMAEConfig.from_dict(ckpt['encoder_config'])
        cfg._attn_implementation = 'sdpa'
        hop = stft_hop_for_frames(length, cfg.num_frames)
        renderer = make_renderer(self.renderer_kind, system='1010', hop_length=hop)
        load_renderer = ckpt['renderer_state']
        if self.renderer_kind in ('r2', 'r3'):
            result = renderer.load_state_dict(load_renderer, strict=False)
            missing = [k for k in result.missing_keys if not k.startswith('pix2eeg.')]
            assert not missing and not result.unexpected_keys, (missing, result.unexpected_keys)
        else:
            renderer.load_state_dict(load_renderer, strict=True)
        value_mode = ckpt.get('config', {}).get('value_mode', 'minmax')
        adapter = make_adapter(self.adapter_kind, output_size=cfg.image_size, value_mode=value_mode)
        adapter.load_state_dict(ckpt['adapter_state'], strict=True)
        encoder = VideoMAEModel(cfg)
        encoder.load_state_dict(ckpt['encoder_state'], strict=True)
        if args.tuning == 'lora':
            inject_videomae_lora(encoder, rank=8, alpha=16, dropout=0)
        self.encoder = BSRIEncoder(renderer, adapter, encoder)
        if args.tuning == 'head' and args.head == 'linear':
            self.head = nn.Linear(cfg.hidden_size, classes)   # strict linear probe
        else:
            self.head = classifier_head(cfg.hidden_size, classes, args.head, args.norm)
        # D70 readout: head-side only (retrained per FT run, backbone untouched).
        self.readout = getattr(args, 'readout', 'mean')
        if self.readout not in READOUTS:
            raise ValueError(f'unknown readout {self.readout!r}; choose from {READOUTS}')
        if self.readout == 'attn':
            self.readout_attn = nn.Linear(cfg.hidden_size, 1)
        self.encoder.requires_grad_(False)
        if args.tuning == 'full':
            self.encoder.requires_grad_(True)
        elif args.tuning == 'lora':
            self.encoder.adapter.requires_grad_(True)
            for m in encoder.modules():
                if isinstance(m, LoRALinear):
                    m.lora_A.requires_grad_(True)
                    m.lora_B.requires_grad_(True)
        # The inverse renderer is not part of the classification path.
        self.encoder.renderer.pix2eeg.requires_grad_(False)
        if self.renderer_kind == 'r3':
            self.encoder.renderer.projection.requires_grad_(False)
        if args.tuning != 'head':
            encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})

    def token_sequence(self, x):
        """Patch-token sequence [B, N, D] from the same encoder path (no pooling)."""
        _, adapted = self.encoder.render_and_adapt(x)
        normalized = ((adapted - self.encoder.image_mean) / self.encoder.image_std) \
            .permute(0, 2, 1, 3, 4)
        return self.encoder.videomae(normalized).last_hidden_state

    def forward(self, x):
        if self.readout == 'mean':
            # Bitwise-identical to the legacy path: BSRIEncoder.forward mean-pools.
            return self.head(self.encoder(x))
        sequence = self.token_sequence(x)
        features = pool_tokens(sequence, self.readout,
                               getattr(self, 'readout_attn', None))
        return self.head(features)


def loader(rows, labels, length, batch, shuffle=False):
    ds = EEGPickleDataset([Path(r['path']) for r in rows],
                          label_to_id={v: i for i, v in enumerate(labels)},
                          segment_length=length, training=False)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=2,
                      pin_memory=True, drop_last=shuffle and len(ds) % batch == 1)


def batches(data, model=None, device='cuda', ea_w=None):
    for batch in data:
        if isinstance(batch, dict):
            x, y = (batch['eeg'].to(device, non_blocking=True),
                    batch['label'].to(device, non_blocking=True))
        else:
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        if ea_w is not None:
            x = ea_apply(x, ea_w.to(device))
        yield x, y


def verify_test_rows_sha(rows, what='test'):
    """D57-3: fresh fail-closed sha check of the eval rows right before test
    evaluation. The splits manifest was built from raw files; if the raw files
    drifted between build and run (rebuild/re-export), the run must stop instead
    of silently comparing different data against the historical test lock."""
    bad = []
    for row in rows:
        digest = sha(Path(row['path']))
        if digest != row.get('sha256'):
            bad.append((str(row['path']), row.get('sha256'), digest))
    if bad:
        raise RuntimeError(
            f'{what}: {len(bad)} row(s) changed since splits_dirs were built '
            f'(manifest sha != fresh file sha); refusing to evaluate test. '
            f'First: {bad[0]}')


@torch.no_grad()
def evaluate(model, data, device='cuda', args=None, alt=None):
    model.eval()
    if alt is not None:
        alts_list = alt if isinstance(alt, (list, tuple)) else [alt]
        for a in alts_list:
            a.eval()
    truth, pred = [], []
    ea_w_eval = (getattr(args, '_ea_W_val', None) if args is not None else None)
    if ea_w_eval is None:
        ea_w_eval = (getattr(args, '_ea_W', None) if args is not None else None)
    restore = None
    active = model
    st_audit = None
    if args is not None and getattr(args, 'selftrain', False):
        adapted = selftrain_adapt(model, data, device, args, ea_w_eval)
        if adapted is not None:
            active = adapted
            st_audit = 'pseudo-label head adaptation applied'
        else:
            st_audit = 'fail-safe: under-coverage, plain prediction'
    if args is not None and getattr(args, 'coral', False):
        # v15/D38 CORAL: affine feature alignment train->target (unlabeled target)
        source_data = getattr(args, '_coral_source', None)
        assert source_data is not None, 'CORAL missing source rows'
        restore = coral_fit_transform(model, source_data, data, device, ea_w_eval)
    if args is not None and getattr(args, 'tbn', False):
        tbn_refresh(model, data, device, ea_w_eval)
    if active is not model:
        alt = None   # self-training and multi-model ensembles are exclusive (v15 gates)
    for x, y in batches(data, active, device, ea_w_eval):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            p = eval_logits(active, x, args, alt).argmax(-1)
        truth.extend(y.cpu().tolist())
        pred.extend(p.cpu().tolist())
    if args is not None and st_audit is not None:
        args._selftrain_audit = st_audit
    if restore is not None:
        restore()
    return {'balanced_accuracy': float(balanced_accuracy_score(truth, pred)),
            'confusion_matrix': confusion_matrix(truth, pred).tolist(),
            'targets': truth, 'predictions': pred}


def train(model, train_loader, val_loader, args, output, device='cuda', alt=None,
          trainable_alt=None):
    model.to(device)
    epochs = args.epochs
    groups = []
    if hasattr(model, 'encoder'):
        groups = [{'params': [p for p in model.encoder.videomae.parameters() if p.requires_grad],
                   'lr': args.encoder_lr},
                  {'params': [p for p in model.encoder.adapter.parameters() if p.requires_grad],
                   'lr': args.adapter_lr},
                  {'params': [p for p in model.encoder.renderer.parameters() if p.requires_grad],
                   'lr': args.renderer_lr},
                  {'params': list(model.head.parameters()) +
                   ([*model.readout_attn.parameters()] if hasattr(model, 'readout_attn') else []),
                   'lr': args.lr}]
    else:
        groups = [{'params': list(model.parameters()), 'lr': args.lr}]
    if trainable_alt is not None:
        trainable_alt.to(device)
        # rebuild per-model groups with identical LRs (two-model ensemble training)
        groups = []
        for base in (model, trainable_alt):
            if hasattr(base, 'encoder'):
                groups += [{'params': [q for q in base.encoder.videomae.parameters() if q.requires_grad], 'lr': args.encoder_lr},
                           {'params': [q for q in base.encoder.adapter.parameters() if q.requires_grad], 'lr': args.adapter_lr},
                           {'params': [q for q in base.encoder.renderer.parameters() if q.requires_grad], 'lr': args.renderer_lr},
                           {'params': list(base.head.parameters()) +
                            ([*base.readout_attn.parameters()] if hasattr(base, 'readout_attn') else []),
                            'lr': args.lr}]
            else:
                groups.append({'params': list(base.parameters()), 'lr': args.lr})
    groups = [g for g in groups if g['params']]
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.schedule_epochs or epochs)
    best, best_epoch, last_score = -1., 0, None
    history = []
    started = time.monotonic()
    steps = 0
    limit_hit = False
    patience_hit = False
    batches_cap_hit = False
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    # R31: weights are computed in downstream() (where `split` lives) and passed in.
    class_weight_vec = getattr(args, '_class_weight_vec', None)
    if class_weight_vec is not None:
        class_weight_vec = class_weight_vec.to(device)
    swa_states = []
    ema_shadow = {}
    ema_decay = getattr(args, 'ema_decay', 0.0)
    for epoch in range(1, epochs + 1):
        model.train()
        if hasattr(model, 'encoder') and args.tuning == 'head':
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, samples = 0.0, 0
        index = -1
        for index, (x, y) in enumerate(batches(train_loader, model, device,
                                              getattr(args, '_ea_W', None))):
            denom = min(args.accumulate,
                        len(train_loader) - (index // args.accumulate) * args.accumulate)
            if getattr(args, 'time_shift', 0.0) > 0:
                x = time_shift(x, getattr(args, 'time_shift', 0.0))
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                smoothing = getattr(args, 'label_smoothing', 0.0)
                mix_alpha = getattr(args, 'mixup_alpha', 0.0)
                if mix_alpha > 0:
                    # v16/D40: time-domain input mixup, loss-side soft targets
                    x_m, lam = mixup_pair(x, mix_alpha)
                    logits = model(x_m)
                    loss = lam * nn.functional.cross_entropy(
                        logits, y, weight=class_weight_vec, label_smoothing=smoothing)
                    loss = loss + (1.0 - lam) * nn.functional.cross_entropy(
                        logits, y[torch.randperm(y.shape[0], device=y.device)],
                        weight=class_weight_vec, label_smoothing=smoothing)
                else:
                    loss = nn.functional.cross_entropy(
                        model(x), y, weight=class_weight_vec, label_smoothing=smoothing)
                    if trainable_alt is not None:
                        loss = loss + nn.functional.cross_entropy(
                            trainable_alt(x), y, weight=class_weight_vec,
                            label_smoothing=smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            (loss / denom).backward()
            loss_sum += float(loss.detach()) * len(y)
            samples += len(y)
            if (index + 1) % args.accumulate == 0 or index + 1 == len(train_loader):
                if steps == 0:
                    gradient = {}
                    for name, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            group = '.'.join(name.split('.')[:2])
                            gradient[group] = gradient.get(group, 0.) + float(p.grad.detach().float().norm())
                    dump(output / 'first_step_gradients.json', gradient)
                    assert sum(gradient.values()) > 0, 'no gradient reached any trainable parameter'
                    if hasattr(model, 'encoder') and args.tuning == 'full':
                        assert gradient.get('encoder.videomae', 0) > 0, 'encoder got no gradient'
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            if args.batch_sleep_seconds > 0:
                time.sleep(args.batch_sleep_seconds)
            if args.max_batches and index + 1 >= args.max_batches:
                batches_cap_hit = True
                break
            if args.seconds_limit > 0 and time.monotonic() - started > args.seconds_limit:
                limit_hit = True
                break
        if getattr(args, 'swa_epochs', 0) > 0:
            snap = {k: v.detach().float().cpu().clone()
                    for k, v in model.state_dict().items() if v.dtype.is_floating_point}
            swa_states.append(snap)
            if len(swa_states) > args.swa_epochs:
                swa_states.pop(0)
        schedule.step()
        if ema_decay > 0:
            ema_update(ema_shadow, model, ema_decay)
        _saved = None
        if ema_decay > 0 and val_loader is not None and ema_shadow:
            _saved = ema_swap(model, ema_shadow, 'apply')   # v16: evaluate with EMA weights
        score = evaluate(model, val_loader, device, args, alt)['balanced_accuracy'] if val_loader is not None else None
        if _saved:
            for name, q in _saved.items():
                model.state_dict()[name].copy_(q)
        if score is not None:
            last_score = score
            if score > best:
                best, best_epoch = score, epoch
        row = dict(epoch=epoch, train_loss=loss_sum / max(1, samples), validation_bacc=score,
                   epoch_complete=(index + 1 == len(train_loader)), seconds=time.monotonic() - started,
                   optimizer_steps=steps,
                   peak_gpu_gib=(torch.cuda.max_memory_allocated() / 2 ** 30)
                   if device.type == 'cuda' else None)
        history.append(row)
        dump(output / 'history.json', history)
        print(json.dumps(row), flush=True)
        if args.select == 'best' and score is not None and score == best:
            torch.save(model.state_dict(), output / 'best.pt')
        if args.select == 'last':
            torch.save(model.state_dict(), output / 'best.pt')   # 'last' == final epoch
        if args.patience and score is not None and epoch - best_epoch >= args.patience:
            patience_hit = True
            break
        if limit_hit:
            break
    selected = last_score if args.select == 'last' else best
    if limit_hit:
        stop_reason = 'seconds_limit'
    elif patience_hit:
        stop_reason = 'patience'
    elif batches_cap_hit:
        stop_reason = 'max_batches'
    elif len(history) < epochs:
        stop_reason = 'incomplete'
    else:
        stop_reason = 'max_epochs'
    epochs_complete = all(r['epoch_complete'] for r in history)
    return dict(best_validation_bacc=selected, selection=args.select,
                best_val_bacc_is_best_epoch=best, best_epoch=best_epoch,
                last_epoch_validation_bacc=last_score, epochs_run=len(history),
                complete=bool(history) and epochs_complete
                         and stop_reason in ('max_epochs', 'patience'),
                stop_reason=stop_reason,
                seconds=time.monotonic() - started, history=history)


def cache_features(model, data, key, args, device='cuda'):
    path = BASE / 'feature_cache' / f'{key}.pt'
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=True)
    model.encoder.to(device).eval()
    features, labels = [], []
    with torch.no_grad():
        for x, y in batches(data, None, device):
            micro = max(1, args.feature_microbatch_size or len(x))
            for start in range(0, len(x), micro):
                stop = min(len(x), start + micro)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    z = model.encoder(x[start:stop])
                features.append(z.float().cpu())
                labels.append(y[start:stop].cpu())
                if args.batch_sleep_seconds > 0:
                    time.sleep(args.batch_sleep_seconds)
    result = (torch.cat(features), torch.cat(labels))
    torch.save(result, path)
    return result


def load_lock_or_none(corpus, stage, checkpoint_hash, args):
    if stage not in TEST_STAGES:
        return None
    lockpath = lock_file()
    if not lockpath.exists():
        raise LockError('final/strict/fixed stage requires manifests/final_v3_lock.json')
    locked = json.loads(lockpath.read_text())['tasks'][corpus]
    problems = []
    if locked.get('checkpoint_sha256') != checkpoint_hash:
        problems.append('checkpoint_sha256')
    for key in ('protocol', 'renderer', 'adapter', 'head', 'tuning', 'epochs'):
        if key in locked and getattr(args, key, None) != locked[key]:
            problems.append(key)
    for key in ('tuning', 'head', 'norm', 'batch', 'lr', 'encoder_lr',
                'adapter_lr', 'renderer_lr', 'accumulate'):
        if key in locked.get('config', {}) and getattr(args, key, None) != locked['config'][key]:
            problems.append(f'config.{key}')
    for key in ('seed', 'select', 'schedule_epochs', 'patience'):
        if key in locked and getattr(args, key, None) != locked[key]:
            problems.append(key)
    if problems:
        raise LockError(f'final config does not match final_v3_lock.json: {problems}')
    if 'split_sha256' not in locked:
        raise LockError('final_v3_lock.json must carry per-subject split_sha256 (DEV-3)')
    return sha(lockpath)


def check_subject_split_locked(corpus, sid, split_sha, locked_split_sha):
    expected = locked_split_sha.get(corpus, {}).get(f'subject{sid}')
    if expected is None or expected != split_sha:
        raise LockError(f'subject{sid} split_sha256 not locked or mismatched (DEV-3)')


def run_cv_loso(args, checkpoint, primary_is_plain, scaffold, subjects, device):
    """D70 train-only LOSO: every held-out fold never touches val or test."""
    if args.stage != 'search':
        raise AssertionError('--cv-loso is train-only and only valid for --stage search '
                             f'(got {args.stage!r}); it never reads val/test')
    length = LENGTHS[args.corpus]
    splits = {}
    for sid in subjects:
        manifest = _splits_root() / args.corpus / f'subject{sid}.json'
        splits[sid] = json.loads(manifest.read_text())
    labels = splits[subjects[0]]['labels']
    outdir = Path(args.output) / args.corpus
    outdir.mkdir(parents=True, exist_ok=True)
    old = dict(epochs=args.epochs, select=getattr(args, 'select', None),
               patience=args.patience)
    folds = {}
    started = time.monotonic()
    try:
        args.select, args.patience = 'last', 0        # fixed epochs, no val early stop
        for held in subjects:
            other_rows = [row for sid in subjects if sid != held
                          for row in splits[sid]['train']]
            eval_rows = splits[held]['train']
            seed_all(args.seed)
            model = Model(scaffold if primary_is_plain else checkpoint,
                          len(labels), args, length)
            if primary_is_plain:
                model.load_state_dict(checkpoint, strict=False)
            fold_dir = outdir / 'cv_loso' / f'held{held}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            train_loader = loader(other_rows, labels, length, args.batch, True)
            eval_loader = loader(eval_rows, labels, length, args.batch)
            train(model, train_loader, None, args, fold_dir, device=device)
            metric = evaluate(model, eval_loader, device=device, args=args)
            folds[f'subject{held}'] = float(metric['balanced_accuracy'])
            print(json.dumps({'cv_loso_fold': held,
                              'balanced_accuracy': folds[f'subject{held}']}), flush=True)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    finally:
        for key, value in old.items():
            setattr(args, key, value)
    mean = sum(folds.values()) / max(1, len(folds))
    payload = {
        'schema': 'v27-cv-loso-v1', 'corpus': args.corpus, 'stage': 'search',
        'readout': args.readout, 'head': args.head, 'tuning': args.tuning,
        'protocol': args.protocol, 'epochs': old['epochs'], 'n_folds': len(folds),
        'folds': folds, 'mean_balanced_accuracy': mean,
        'ckpt': str(args.ckpt), 'ckpt_sha256': sha(Path(args.ckpt)),
        'fit_rows': 'other_subjects_train', 'eval_rows': 'heldout_subject_train',
        'test_read': False, 'val_read': False,
        'elapsed_seconds': time.monotonic() - started,
        'notes': ['train-only LOSO for no-val benches (D70)',
                  'fit = remaining training subjects train rows; eval = held-out '
                  'subject train rows; val/test never read',
                  'LOSO selection bias disclosed; EA skipped in LOSO (approximation)'],
    }
    dump(outdir / 'cv_loso.json', payload)
    print(json.dumps({k: payload[k] for k in ('corpus', 'readout', 'head', 'n_folds',
                                            'mean_balanced_accuracy')}), flush=True)
    return 0


def downstream(args):
    validate_stage_protocol(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.ckpt
    if '{sid' in str(ckpt_path):
        first_sid = 1 if args.subjects == 'all' else int(str(args.subjects).split(',')[0])
        ckpt_path = _Path(str(ckpt_path).format(sid=first_sid))
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    checkpoint_hash = sha(ckpt_path)
    # v7: plain state dicts (training best.pt incl. TRAINED corpus heads) load via
    # a compact scaffold. Root cause of v6 chance-level ensembles: compact ckpts
    # carry NO trained corpus head -> eval-only averaged two untrained heads.
    primary_is_plain = 'v3' not in checkpoint
    scaffold = None
    if primary_is_plain:
        assert getattr(args, 'backbone_ckpt', ''), \
            'plain state dict primary requires --backbone-ckpt scaffold (v7)'
        scaffold = torch.load(args.backbone_ckpt, map_location='cpu', weights_only=False)
    preset = protocol_preset(args)
    applied = apply_protocol(args, preset, getattr(args, '_explicit', ()))
    lock_hash = load_lock_or_none(args.corpus, args.stage, checkpoint_hash, args)
    locked_subject_shas = None
    if args.stage in TEST_STAGES:
        locked_subject_shas = json.loads(
            lock_file().read_text()
        )['tasks'][args.corpus]['split_sha256']
    subjects = (list(range(1, CORPORA[args.corpus] + 1)) if args.subjects == 'all'
                else [int(x) for x in args.subjects.split(',')])
    if getattr(args, 'cv_loso', False):
        return run_cv_loso(args, checkpoint, primary_is_plain, scaffold, subjects, device)
    for sid in subjects:
        output = Path(args.output) / args.corpus / f'subject{sid}'
        output.mkdir(parents=True, exist_ok=True)
        manifest = _splits_root() / args.corpus / f'subject{sid}.json'
        split = TestGuardedSplit(json.loads(manifest.read_text()),
                                 allow_test=(args.stage in TEST_STAGES))
        # R31: per-subject class weights (v2 semantics: rows are the subject's
        # train rows at search / full_train at final; full label set enforced).
        args._class_weight_vec = (class_weight_vector_for(split, args.stage, args.corpus)
                                  if getattr(args, 'class_weight', False) else None)
        length = LENGTHS[args.corpus]
        if getattr(args, 'ea', False):
            # D30: EA statistics from this subject's TRAIN rows only (search=train /
            # final=full_train); applied to every split of the subject; interacts
            # BEFORE the model's adapter normalization (EA -> original norm path).
            assert args.tuning == 'full', 'EA v10 gate protocol requires tuning=full'
            ea_rows = training_rows_for_stage(split, args.stage)
            ea_ds_loader = loader(ea_rows, split['labels'], length, args.batch)
            args._ea_W = ea_whitening_from_batches(
                batches(ea_ds_loader, None, device), device,
                lam=getattr(args, 'ea_lambda', 1e-5),
                mode=getattr(args, 'ea_lambda_mode', 'tr_over_C2'))
            # D30-errata-2 audit: W sha, cond/rank of R~, whitened-variance profile
            args._ea_audit = ea_audit_from_loader(
                batches(ea_ds_loader, None, device), args._ea_W, device,
                lam=getattr(args, 'ea_lambda', 1e-5), stage=args.stage,
                mode=getattr(args, 'ea_lambda_mode', 'tr_over_C2'))
            if getattr(args, 'ea_mode', 'inductive') == 'transductive':
                # v14/D36: eval side gets ITS OWN W fit on the eval split's UNLABELED
                # windows (search=val session, final/fixed=test session); no labels used.
                eval_rows = (split['test'] if args.stage in ('final', 'fixed')
                             else split['validation'])
                args._ea_audit['transductive'] = {
                    'train_sessions': sorted({ea_session_key(r) for r in ea_rows}),
                    'eval_sessions': sorted({ea_session_key(r) for r in eval_rows})}
                eval_loader = loader(eval_rows, split['labels'], length, args.batch)
                args._ea_W_val = ea_whitening_from_batches(
                    batches(eval_loader, None, device), device,
                    lam=getattr(args, 'ea_lambda', 1e-5),
                    mode=getattr(args, 'ea_lambda_mode', 'tr_over_C2'))
                args._ea_audit['eval'] = ea_audit_from_loader(
                    batches(eval_loader, None, device), args._ea_W_val, device,
                    lam=getattr(args, 'ea_lambda', 1e-5), stage='eval-transductive',
                    mode=getattr(args, 'ea_lambda_mode', 'tr_over_C2'))
            else:
                args._ea_W_val = None
        else:
            args._ea_W = None
            args._ea_W_val = None
        if getattr(args, 'coral', False):
            args._coral_source = loader(training_rows_for_stage(split, args.stage),
                                        split['labels'], length, args.batch)
        if args.stage != 'final':
            if not split.get('selection_eligible', True):
                dump(output / 'excluded.json',
                     dict(reason=split.get('selection_exclusion'), stage=args.stage,
                          split_sha256=sha(manifest)))
                continue
        if (output / 'summary.json').exists():
            prior = json.loads((output / 'summary.json').read_text())
            assert prior['split_sha256'] == sha(manifest) and \
                prior['checkpoint_sha256'] == checkpoint_hash, \
                'stale output; archive before rerunning'
            if prior.get('protocol_hash') and \
                    prior['protocol_hash'] != spec['protocol_hash']:
                raise LockError('stale output under a different protocol; '
                                'archive before rerunning (DEV-2)')
            continue
        if args.stage in TEST_STAGES:
            check_subject_split_locked(args.corpus, sid, sha(manifest), locked_subject_shas)
        seed_all(args.seed)
        if '{sid' in str(args.ckpt):
            # v7: per-subject re-load (template) inside the loop where sid exists
            checkpoint = torch.load(_Path(str(args.ckpt).format(sid=sid)),
                                    map_location='cpu', weights_only=False)
            checkpoint_hash = sha(_Path(str(args.ckpt).format(sid=sid)))
            if args.stage == 'final':
                pinned = json.loads(lock_file().read_text())[
                    'tasks'][args.corpus].get('per_subject_primary_sha256')
                if pinned is not None and pinned.get(f'subject{sid}') != checkpoint_hash:
                    raise LockError(f'subject{sid} primary ckpt sha != lock (final stage)')
            primary_is_plain = 'v3' not in checkpoint
            scaffold = None
            if primary_is_plain:
                scaffold = torch.load(args.backbone_ckpt, map_location='cpu', weights_only=False)
        if primary_is_plain:
            model = Model(scaffold, len(split['labels']), args, length)
            model.load_state_dict(checkpoint, strict=False)
        else:
            model = Model(checkpoint, len(split['labels']), args, length)
        rows = split['full_train'] if args.stage == 'final' else split['train']
        # D56 fixed: no validation rows exist for these corpora -> train the fixed
        # epoch count and select the last epoch (test read once afterwards).
        validation = None if args.stage in ('final', 'fixed') else split['validation']
        train_loader = loader(rows, split['labels'], length, args.batch, True)
        val_loader = loader(validation, split['labels'], length, args.batch) if validation else None
        spec = {k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items() if not k.startswith('_')}
        spec['implementation_sha256'] = sha(__file__)
        spec['protocol'] = args.protocol
        spec['protocol_applied'] = applied
        protocol_fields = {k: getattr(args, k) for k in sorted(PROTOCOL_KEYS)}
        spec['protocol_hash'] = hashlib.sha256(
            json.dumps(protocol_fields, sort_keys=True, default=str).encode()).hexdigest()
        spec['torch_version'] = torch.__version__
        if args.tuning == 'head' and args.head == 'linear':
            spec['linear_probe_semantics'] = 'frozen_encoder_and_single_affine_linear_layer'
        spec['renderer_hop'] = stft_hop_for_frames(length, 16)
        dump(output / 'config.json', dict(**spec, checkpoint_sha256=checkpoint_hash,
                                          split_sha256=sha(manifest)))
        if args.tuning == 'head':
            def features(records):
                key = hashlib.sha256(json.dumps(
                    [checkpoint_hash, length, sha(BASE / 'finetune.py'),
                     'v3-feature-cache-v1', args.head, args.norm,
                     [(r['path'], r['sha256']) for r in records]]).encode()).hexdigest()
                return TensorDataset(*cache_features(
                    model, loader(records, split['labels'], length, args.batch), key, args, device))
            train_set = features(rows)
            val_set = features(validation) if validation else None
            model = model.head
            train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                                      drop_last=len(train_set) % args.batch == 1)
            val_loader = DataLoader(val_set, batch_size=args.batch) if val_set is not None else None
        if args.stage == 'probe':
            args.epochs = 1
            args.max_batches = 3
        alt_model = None
        if getattr(args, 'ensemble_ckpts', ''):
            # v13/D34: multi-model logit ensemble over per-subject TRAINED best.pt
            alt_model = []
            for raw in str(args.ensemble_ckpts).split(','):
                alt_payload = torch.load(raw, map_location='cpu', weights_only=False)
                try:
                    am = Model(alt_payload, len(split['labels']), args, length)
                except AssertionError:
                    base = scaffold if primary_is_plain else checkpoint
                    am = Model(base, len(split['labels']), args, length)
                    am.load_state_dict(alt_payload, strict=False)
                am = am.to(device)
                for parameter in am.parameters():
                    parameter.requires_grad_(False)
                alt_model.append(am)
        if alt_model is None and getattr(args, 'ensemble_ckpt', ''):
            ens_path = str(args.ensemble_ckpt)
            if '{sid' in ens_path:
                ens_path = ens_path.format(sid=sid)
            alt_payload = torch.load(ens_path, map_location='cpu', weights_only=False)
            try:
                alt_model = Model(alt_payload, len(split['labels']), args, length)
            except AssertionError:
                base = scaffold if primary_is_plain else checkpoint
                alt_model = Model(base, len(split['labels']), args, length)
                alt_model.load_state_dict(alt_payload, strict=False)
            alt_model = alt_model.to(device)
            for parameter in alt_model.parameters():
                parameter.requires_grad_(False)
        if getattr(args, 'eval_only', False):
            assert args.stage == 'search', 'eval-only is a val-selection tool; never for final'
            model.to(device)
            score = evaluate(model, val_loader, device, args, alt_model)['balanced_accuracy']
            result = {'status': 'OK', 'stage': args.stage, 'stop_reason': 'eval_only',
                      'complete': True, 'epochs_run': 0, 'best_validation_bacc': score,
                      'subject': f'subject{sid}', 'corpus': args.corpus,
                      'checkpoint_sha256': checkpoint_hash, 'split_sha256': sha(manifest),
                      'selection_uses_test': False}
            dump(output / 'summary.json', result)
            print(json.dumps(result), flush=True)
            continue
        trainable_alt = None
        if args.stage == 'final' and getattr(args, 'train_ensemble_alt', ''):
            # A1 (D26): train the SECOND model (R1B1-init) alongside at final stage
            alt_payload2 = torch.load(args.train_ensemble_alt, map_location='cpu',
                                      weights_only=False)
            trainable_alt = Model(alt_payload2, len(split['labels']), args, length)
            if args.tuning == 'head':
                raise LockError('train-ensemble-alt supports full/lora tuning only')
            inner = getattr(trainable_alt.encoder, 'videomae', None)
            enable = getattr(inner, 'gradient_checkpointing_enable', None)
            if enable is not None:
                enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        training = train(model, train_loader, val_loader, args, output, device,
                         alt_model, trainable_alt)
        result = {**training, 'subject': f'subject{sid}', 'corpus': args.corpus,
                  'stage': args.stage, 'checkpoint_sha256': checkpoint_hash,
                  'split_sha256': sha(manifest), 'config': spec,
                  'final_lock_sha256': lock_hash}
        if args.stage in TEST_STAGES:
            if args.stage == 'final':
                assert training['epochs_run'] == args.epochs, \
                    'final training stopped before the locked epoch count'
            if args.stage == 'fixed':
                # D56: fixed epoch count, last epoch selected, no validation rows.
                assert training['epochs_run'] == args.epochs, \
                    'fixed training stopped before the locked epoch count'
            assert training['complete'], \
                f"training incomplete (stop_reason={training.get('stop_reason')})"
            if args.stage == 'strict':
                # v21/D54 strict protocol: val only selects/early-stops (no gradients),
                # test is evaluated ONCE from the selected checkpoint.
                assert args.select == 'best', 'strict stage requires --select best'
                best_path = output / 'best.pt'
                assert best_path.exists(), 'strict stage requires the selected best.pt'
                selected_state = torch.load(best_path, map_location='cpu', weights_only=True)
                if args.tuning == 'head':
                    backbone = Model(checkpoint, len(split['labels']), args, length).to(device)
                    backbone.head = model
                    model = backbone
                    model.head.load_state_dict(
                        {k: v.to(device) for k, v in selected_state.items()})
                else:
                    model.load_state_dict({k: v.to(device) for k, v in selected_state.items()},
                                          strict=False)
                model.to(device)
                if getattr(args, 'ea', False) and getattr(args, 'ea_mode', 'inductive') == 'transductive':
                    # final-time transductive W is fit on the TEST session's unlabeled
                    # windows (no labels), exactly as in the v19 final protocol (D36)
                    test_loader_ea = loader(split['test'], split['labels'], length, args.batch)
                    args._ea_W_val = ea_whitening_from_batches(
                        batches(test_loader_ea, None, device), device,
                        lam=getattr(args, 'ea_lambda', 1e-5),
                        mode=getattr(args, 'ea_lambda_mode', 'tr_over_C2'))
                result['strict_selection'] = {
                    'select': args.select, 'patience': args.patience,
                    'best_epoch': training.get('best_epoch'),
                    'epochs_run': training.get('epochs_run'),
                    'stop_reason': training.get('stop_reason'),
                    'fit_rows': 'train', 'val_rows': 'validation',
                    'val_updates_parameters': False}
            else:
                if args.stage == 'fixed':
                    # D56 fixed: same in-memory last-epoch model as `final`, no val.
                    assert not split['validation'], \
                        'fixed stage requires empty validation rows (bcic2a/seedvii, D56)'
                    result['fixed_selection'] = {
                        'select': 'last', 'epochs': args.epochs,
                        'epochs_run': training.get('epochs_run'),
                        'stop_reason': training.get('stop_reason'),
                        'fit_rows': 'train', 'val_role': 'none',
                        'val_updates_parameters': False}
                    result['fit_rows'] = 'train'
                    result['val_role'] = 'none'
                if args.tuning == 'head':
                    backbone = Model(checkpoint, len(split['labels']), args, length).to(device)
                    backbone.head = model
                    model = backbone
            with torch.no_grad():
                assert 'test' in split, 'final/strict/fixed stage requires the locked test split'
            if args.stage in ('strict', 'fixed'):
                # D57-3: fail-closed fresh sha check right before the one test read.
                verify_test_rows_sha(split['test'], f'test/{args.corpus}/subject{sid}')
            eval_alt = alt_model if alt_model is not None else trainable_alt
            metric = evaluate(model, loader(split['test'], split['labels'], length, args.batch),
                              device, args, eval_alt)
            result.update(metric)
            if getattr(args, 'ea', False):
                # D30-errata-2 runtime health: finite predictions, no degenerate single-class
                preds = metric['predictions']
                assert all(isinstance(q, int) for q in preds), 'non-int predictions'
                assert len(set(preds)) >= 2, 'degenerate single-class predictions'
                assert torch.isfinite(torch.tensor(metric['confusion_matrix'], dtype=torch.float32)).all()
                result['ea_audit'] = getattr(args, '_ea_audit', None)
            torch.save(model.state_dict(), output /
                       ('final.pt' if args.stage == 'final' else 'selected.pt'))
        dump(output / 'summary.json', result)
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ('history', 'config', 'targets', 'predictions')}), flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def build_parser():
    """CLI parser (exposed for contract tests)."""
    p = argparse.ArgumentParser(description='v3 downstream runner (search never reads test).')
    p.add_argument('--stage', choices=['probe', 'search', 'final', 'strict', 'fixed'],
                   required=True,
                   help="strict: train rows only, validation for early-stop/selection "
                        "(--select best --patience N), test evaluated once from the "
                        "selected checkpoint (v21/D54). fixed: train rows only, NO "
                        "validation, fixed --epochs, --select last, test once (v22/D56).")
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--corpus', choices=list(CORPORA), required=True)
    p.add_argument('--subjects', default='all')
    p.add_argument('--protocol', choices=('none', 'pv1', 'pv2'), default='none',
                   help='none = v2-style explicit config; pv1/pv2 apply locked presets '
                        'to every field NOT passed explicitly on the command line.')
    p.add_argument('--tuning', choices=['head', 'lora', 'full'], default='head')
    p.add_argument('--head', choices=['linear', 'mlp'], default='linear')
    p.add_argument('--cv-loso', action='store_true',
                   help='D70 train-only leave-one-subject-out: fit on the other training '
                        'subjects train rows and evaluate on the held-out subject train '
                        'rows (never reads val/test); for bcic2a/seedvii readout/head '
                        'selection. LOSO selection bias is disclosed.')
    p.add_argument('--readout', choices=list(READOUTS), default='mean',
                   help='D70 token readout: mean (default/legacy, bitwise-identical), '
                        'token0 (first patch token; no synthetic CLS -> approximate), '
                        'attn (learned softmax attention over all patch tokens; '
                        'head-side params, retrained per FT run)')
    p.add_argument('--norm', choices=['none', 'running', 'batch'], default='running')
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--accumulate', type=int, default=1)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--schedule-epochs', type=int, default=0)
    p.add_argument('--select', choices=('best', 'last'), default='best')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--encoder-lr', type=float, default=1e-5)
    p.add_argument('--adapter-lr', type=float, default=1e-4)
    p.add_argument('--renderer-lr', type=float, default=1e-4)
    p.add_argument('--tta-frac', type=float, default=0.0,
                   help='v5: circular-shift TTA fraction (e.g. 1/6); 0 disables')
    p.add_argument('--tta-k', type=int, default=3, help='v5: TTA copies (k=3 -> 0/+/-frac)')
    p.add_argument('--swa-epochs', type=int, default=0,
                   help='v5: average the last K epoch snapshots (SWA-like); 0 disables')
    p.add_argument('--train-ensemble-alt', default='',
                   help='v7: second compact init TRAINED alongside at final stage '
                        '(two-model ensemble; test logits averaged)')
    p.add_argument('--backbone-ckpt', default='',
                   help='v7: compact scaffold when --ckpt is a PLAIN state dict (trained heads)')
    p.add_argument('--ensemble-ckpts', default='',
                   help='v13: comma-separated per-subject trained ckpts (logits-averaged with --ckpt)')
    p.add_argument('--ensemble-ckpt', default='',
                   help='v5: second checkpoint for multi-seed logit averaging (val selection/final)')
    p.add_argument('--eval-only', action='store_true',
                   help='v5: skip training, evaluate val once (search only; val-selection tool)')
    p.add_argument('--ea', action='store_true',
                   help='v10: per-subject Euclidean Alignment (train-cov whitening) before the model')
    p.add_argument('--ea-lambda', type=float, default=1e-5, help='v10: EA ridge regularization')
    p.add_argument('--mixup-alpha', type=float, default=0.0,
                   help='v16/D40: time-domain input mixup alpha (0 disables)')
    p.add_argument('--label-smoothing', type=float, default=0.0,
                   help='v16/D40: CE label smoothing (0 disables)')
    p.add_argument('--ema-decay', type=float, default=0.0,
                   help='v16/D40: weight EMA decay for evaluation (0 disables)')
    p.add_argument('--selftrain', action='store_true',
                   help='v15/D38: test-time pseudo-label self-training on unlabeled target (head only)')
    p.add_argument('--st-min-per-class', type=int, default=8)
    p.add_argument('--st-max-per-class', type=int, default=200)
    p.add_argument('--coral', action='store_true',
                   help='v15/D38: CORAL affine feature alignment (train->target, unlabeled target)')
    p.add_argument('--tbn', action='store_true',
                   help='v15/D38: test-time BatchNorm refresh on target-session unlabeled batches')
    p.add_argument('--ea-mode', choices=['inductive', 'transductive'], default='inductive',
                   help='v14/D36: transductive = per-split unlabeled-window whitening (no labels)')
    p.add_argument('--ea-lambda-mode', choices=['tr_over_C2', 'tr_over_C'], default='tr_over_C2',
                   help='D31: tr_over_C = stronger ridge (1e-4*tr/C) for rank-deficient subjects')
    p.add_argument('--time-shift', type=float, default=0.0,
                   help='v3.1 knob: training-only circular temporal shift fraction (<=0.5)')
    p.add_argument('--class-weight', action='store_true',
                   help='v3.1 knob: inverse train-class-frequency CE weights (v2 balanced)')
    p.add_argument('--seconds-limit', type=float, default=1800)
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--batch-sleep-seconds', type=float, default=0.0,
                   help='thermal pacing for Spark (sleep after every batch/micro-batch)')
    p.add_argument('--feature-microbatch-size', type=int, default=0)
    p.add_argument('--patience', type=int, default=0)
    p.add_argument('--seed', type=int, default=4523)
    p.add_argument('--splits-root', default='',
                   help='candidate splits dir (default manifests/splits; the public scripts use manifests/splits_dirs)')
    p.add_argument('--lock', default='',
                   help='lock file generated by scripts/make_lock.py (default manifests/final_v3_lock.json)')
    return p


def main(argv=None):
    p = build_parser()
    argv = list(_sys.argv[1:] if argv is None else argv)
    args = p.parse_args(argv)
    global SPLITS_ROOT, LOCK_PATH
    if args.lock:
        LOCK_PATH = _Path(args.lock)
        assert LOCK_PATH.is_file(), f'lock missing: {LOCK_PATH}'
    if args.splits_root:
        SPLITS_ROOT = _Path(args.splits_root)
        assert SPLITS_ROOT.is_dir(), f'splits-root missing: {SPLITS_ROOT}'
    explicit = {p._option_string_actions[tok].dest for tok in argv
                if tok.startswith('--') and tok in p._option_string_actions}
    args._explicit = explicit
    seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    downstream(args)


if __name__ == '__main__':
    main()
