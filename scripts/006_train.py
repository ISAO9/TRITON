"""
006_train.py
------------
Training script for TRITON (CNN_v2 and MIRROR).

Corresponds to: "Training Procedure" in Methods.

Settings (paper):
  Optimiser:   AdamW (lr=1e-4, weight_decay=1e-4)
  Scheduler:   CosineAnnealingWarmRestarts (T0=30, T_mult=2)
  Precision:   bfloat16 mixed
  Grad clip:   1.0
  Epochs:      100  (early stop patience=20)
  Curriculum:  first 20 epochs SNR≥15dB only
  Checkpoint:  best validation MAE retained

Usage:
    python scripts/006_train.py --model cnn_v2
    python scripts/006_train.py --model mirror
    python scripts/006_train.py --model cnn_v2 --generate-data
"""

import argparse
import json
import os
import shutil
import time
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset


# ── Paths ────────────────────────────────────────────────────────────────────
FEAT_H5   = 'data/features.h5'
RAW_H5    = 'data/synthetic.h5'
MODEL_DIR = 'models'
T_MAX     = 15.0
N_CH      = 128
NT_OUT    = 2048


# ── Dataset ──────────────────────────────────────────────────────────────────

class TRITONDataset(Dataset):
    """
    PyTorch Dataset for TRITON feature HDF5.

    Returns per-sample dict with:
      gather_norm (1, 128, 2048)
      fk_spectrum (1, 128, 128)
      t_p_norm    (128,)
      t_ac_norm   (128,)
      snr_db      scalar
    """
    def __init__(self, feat_h5: str, split: str = 'train'):
        self.h5   = h5py.File(feat_h5, 'r')
        self.grp  = self.h5[split]
        self.n    = self.grp['gather_norm'].shape[0]
        self.t_p  = self.grp['t_p'][:]   if 't_p'  in self.grp else None
        self.t_ac = self.grp['t_ac'][:] if 't_ac' in self.grp else None
        self.snr  = self.grp['snr_db'][:] if 'snr_db' in self.grp else \
            np.ones(self.n, dtype=np.float32) * 15.0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        gn = torch.from_numpy(self.grp['gather_norm'][i])   # (1,128,2048)
        fk = torch.from_numpy(self.grp['fk_spectrum'][i])   # (1,128,128)

        sample = {
            'gather_norm': gn,
            'fk_spectrum': fk,
            'snr_db':      torch.tensor(float(self.snr[i])),
        }
        if self.t_p is not None:
            sample['t_p_norm'] = torch.from_numpy(
                np.clip(self.t_p[i] / T_MAX, 0.0, 1.0).astype(np.float32))
        if self.t_ac is not None:
            sample['t_ac_norm'] = torch.from_numpy(
                np.clip(self.t_ac[i] / T_MAX, 0.0, 1.0).astype(np.float32))
        return sample


def get_loaders(feat_h5: str, batch_size: int = 32,
                num_workers: int = 4) -> dict:
    """Build DataLoaders for train / val / test."""
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = TRITONDataset(feat_h5, split)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
        )
    return loaders


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model_type:   str   = 'cnn_v2',
    feat_h5:      str   = FEAT_H5,
    model_dir:    str   = MODEL_DIR,
    epochs:       int   = 100,
    patience:     int   = 20,
    batch_size:   int   = 32,
    lr:           float = 1e-4,
    weight_decay: float = 1e-4,
    t0_sched:     int   = 30,
    t_mult:       int   = 2,
    curriculum_ep:int   = 20,
    curriculum_snr:float= 15.0,
    num_workers:  int   = 4,
    seed:         int   = 42,
) -> str:
    """
    Full training run.

    Returns
    -------
    str : path to best checkpoint
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}  |  model: {model_type}')
    os.makedirs(model_dir, exist_ok=True)

    # ── Build model ───────────────────────────────────────────────────────
    if model_type == 'cnn_v2':
        from triton.models.cnn_v2 import build_cnn_v2
        model = build_cnn_v2().to(device)
    elif model_type == 'mirror':
        from triton.models.mirror import build_mirror
        model = build_mirror().to(device)
    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    # ── Loss, optimiser, scheduler ────────────────────────────────────────
    from triton.losses import TRITONLoss
    loss_fn = TRITONLoss(model_type=model_type)
    opt     = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched   = CosineAnnealingWarmRestarts(opt, T_0=t0_sched, T_mult=t_mult)
    scaler  = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    # ── Data ──────────────────────────────────────────────────────────────
    loaders    = get_loaders(feat_h5, batch_size, num_workers)
    print(f'Train: {len(loaders["train"].dataset)} '
          f'Val: {len(loaders["val"].dataset)} '
          f'Test: {len(loaders["test"].dataset)}')

    best_mae   = float('inf')
    pat_count  = 0
    history    = {'train': [], 'val': []}
    ckpt_path  = os.path.join(model_dir, f'{model_type}_best.pt')

    for epoch in range(epochs):
        curriculum = (epoch < curriculum_ep)
        # ── Train epoch ───────────────────────────────────────────────
        model.train()
        tl = tm = tn = 0.0
        t0 = time.time()

        for batch in loaders['train']:
            # Curriculum: keep only high-SNR samples
            if curriculum:
                mask = batch['snr_db'] >= curriculum_snr
                if mask.sum() == 0:
                    continue
                batch = {k: v[mask] if torch.is_tensor(v) else v
                         for k, v in batch.items()}

            bd = {k: v.to(device) if torch.is_tensor(v) else v
                  for k, v in batch.items()}

            opt.zero_grad(set_to_none=True)

            # Forward
            if device == 'cuda' and scaler is not None:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    preds = _forward(model, model_type, bd)
                    loss, _bd = loss_fn(preds, bd)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                preds = _forward(model, model_type, bd)
                loss, _bd = loss_fn(preds, bd)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            B   = bd['t_p_norm'].shape[0]
            mae = (preds['t_p'].detach() - bd['t_p_norm']).abs().mean().item()
            tl += loss.item() * B
            tm += mae * B
            tn += B

        sched.step()

        # ── Validation epoch ──────────────────────────────────────────
        model.eval()
        vm = vn = 0.0
        with torch.no_grad():
            for batch in loaders['val']:
                bd = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in batch.items()}
                preds = _forward(model, model_type, bd)
                B  = bd['t_p_norm'].shape[0]
                vm += (preds['t_p'] - bd['t_p_norm']).abs().mean().item() * B
                vn += B

        tr_ms = tm / tn * T_MAX * 1000
        va_ms = vm / vn * T_MAX * 1000
        history['train'].append({'mae': tr_ms})
        history['val'].append({'mae': va_ms})

        # ── Checkpoint ────────────────────────────────────────────────
        if va_ms < best_mae:
            best_mae  = va_ms
            pat_count = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_mae':     va_ms / T_MAX / 1000,
                'model_type':  model_type,
            }, ckpt_path)
            marker = ' ← BEST'
        else:
            pat_count += 1
            marker = ''

        elapsed = time.time() - t0
        if (epoch + 1) % 5 == 0 or pat_count == 0:
            print(f'E{epoch:03d}  '
                  f'tr={tr_ms:.0f}ms  val={va_ms:.0f}ms  '
                  f'best={best_mae:.0f}ms  '
                  f'({elapsed:.0f}s){marker}')

        if pat_count >= patience:
            print(f'Early stop at epoch {epoch}')
            break

    # ── Save history ──────────────────────────────────────────────────────
    hist_path = os.path.join(model_dir, f'{model_type}_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nTraining complete. Best val MAE = {best_mae:.1f} ms')
    print(f'Checkpoint: {ckpt_path}')
    return ckpt_path


def _forward(model: nn.Module, model_type: str, batch: dict) -> dict:
    """Unified forward pass for CNN_v2 and MIRROR."""
    if model_type == 'cnn_v2':
        return model(batch['gather_norm'])
    elif model_type == 'mirror':
        return model(batch['gather_norm'], batch['fk_spectrum'])
    raise ValueError(model_type)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train TRITON (Paper: Training Procedure)'
    )
    parser.add_argument('--model',       choices=['cnn_v2', 'mirror'],
                        default='cnn_v2')
    parser.add_argument('--feat-h5',     default=FEAT_H5)
    parser.add_argument('--model-dir',   default=MODEL_DIR)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--patience',    type=int,   default=20)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--num-workers', type=int,   default=4)
    # Data generation helpers
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate synthetic dataset before training')
    parser.add_argument('--raw-h5',  default=RAW_H5)
    args = parser.parse_args()

    # Optionally generate synthetic data
    if args.generate_data:
        print('=== Generating synthetic dataset ===')
        from triton.data.generator import generate_dataset
        from triton.data.features  import build_feature_h5
        generate_dataset(args.raw_h5, n_train=4000, n_val=800, n_test=800)
        build_feature_h5(args.raw_h5, args.feat_h5)

    # Train
    print(f'\n=== Training {args.model.upper()} ===')
    train(
        model_type   = args.model,
        feat_h5      = args.feat_h5,
        model_dir    = args.model_dir,
        epochs       = args.epochs,
        patience     = args.patience,
        batch_size   = args.batch_size,
        lr           = args.lr,
        seed         = args.seed,
        num_workers  = args.num_workers,
    )


if __name__ == '__main__':
    main()
