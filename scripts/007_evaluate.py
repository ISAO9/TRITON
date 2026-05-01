"""
007_evaluate.py
---------------
Evaluation script reproducing Table 1 of the manuscript.

Metrics computed:
  MAE (ms)          Mean absolute error in P-wave arrival time
  R²                Coefficient of determination
  Moveout err (ms)  Mean absolute error in per-gather total moveout
  Dir. err (%)      Percentage of gathers with wrong moveout direction
  SNR<8dB MAE (ms)  MAE restricted to low-SNR gathers

Also evaluates conventional baselines:
  FK-Pick  (f-k filter + amplitude threshold)
  STA/LTA  (Allen 1978)

Usage:
    python scripts/007_evaluate.py --model cnn_v2 --checkpoint models/cnn_v2_best.pt
    python scripts/007_evaluate.py --model mirror  --checkpoint models/mirror_best.pt
    python scripts/007_evaluate.py --baselines-only
"""

import argparse
import json
import os

import h5py
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from typing import Dict, Tuple


# ── Constants ────────────────────────────────────────────────────────────────
T_MAX  = 15.0
N_CH   = 128
FS_RAW = 500.0
FEAT_H5 = 'data/features.h5'
RAW_H5  = 'data/synthetic.h5'


# ── Metric helpers ───────────────────────────────────────────────────────────

def compute_metrics(
    t_p_pred: np.ndarray,   # (N, n_ch)  normalised [0,1]
    t_p_gt:   np.ndarray,   # (N, n_ch)  normalised [0,1]
    snr:      np.ndarray,   # (N,)
    t_max:    float = T_MAX,
) -> Dict[str, float]:
    """
    Compute all Table 1 metrics.

    Parameters
    ----------
    t_p_pred : (N, n_ch)  predicted P-wave arrival (normalised)
    t_p_gt   : (N, n_ch)  ground-truth P-wave arrival (normalised)
    snr      : (N,)       input SNR per sample [dB]
    """
    # Convert to seconds
    tp_s  = t_p_pred * t_max
    tg_s  = t_p_gt   * t_max

    # MAE [ms]
    mae = float(np.abs(tp_s - tg_s).mean() * 1000)

    # R²
    ss_res = ((tp_s - tg_s) ** 2).sum()
    ss_tot = ((tg_s - tg_s.mean()) ** 2).sum() + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)

    # Moveout per gather: t(last_ch) − t(first_ch)  [ms]
    mo_pred = (tp_s[:, -1] - tp_s[:, 0]) * 1000   # (N,)
    mo_gt   = (tg_s[:, -1] - tg_s[:, 0]) * 1000   # (N,)
    mo_err  = float(np.abs(mo_pred - mo_gt).mean())

    # Directional error: wrong sign of moveout [%]
    dir_err = float(np.mean(np.sign(mo_pred) != np.sign(mo_gt)) * 100)

    # Low-SNR MAE
    mask_low = snr < 8.0
    if mask_low.any():
        mae_low = float(np.abs(tp_s[mask_low] - tg_s[mask_low]).mean() * 1000)
    else:
        mae_low = float('nan')

    return {
        'MAE_ms':      mae,
        'R2':          r2,
        'moveout_err': mo_err,
        'dir_err_pct': dir_err,
        'MAE_snr8_ms': mae_low,
    }


# ── Conventional baselines ───────────────────────────────────────────────────

def fk_pick(gather_raw: np.ndarray, fs: float = FS_RAW,
            v_low: float = 4000.0, v_high: float = 8000.0,
            threshold_sigma: float = 3.0) -> np.ndarray:
    """
    FK-Pick baseline.

    1. 2-D FFT
    2. Zero energy outside P-wave velocity passband
    3. Inverse FFT
    4. First sample > threshold × channel MAD
    """
    n_ch, n_t = gather_raw.shape
    dx    = 12.5   # m

    # Velocity grid
    freq  = np.fft.fftfreq(n_t, d=1.0 / fs)       # (n_t,)
    waven = np.fft.fftfreq(n_ch, d=dx)             # (n_ch,)
    FREQ, WAVEN = np.meshgrid(freq, waven)
    with np.errstate(divide='ignore', invalid='ignore'):
        vapp = np.where(np.abs(WAVEN) > 1e-12,
                        FREQ / WAVEN, 0.0)

    # f-k filter mask
    mask = (vapp >= v_low) & (vapp <= v_high)

    fk_raw  = np.fft.fft2(gather_raw)
    fk_filt = fk_raw * mask
    gather_f = np.real(np.fft.ifft2(fk_filt)).astype(np.float32)

    # Pick: first threshold crossing
    t_picks = np.zeros(n_ch, dtype=np.float32)
    for ch in range(n_ch):
        sig = gather_f[ch]
        thr = threshold_sigma * np.median(np.abs(sig))
        idx = np.where(np.abs(sig) > thr)[0]
        t_picks[ch] = (idx[0] / fs) if len(idx) > 0 else np.argmax(np.abs(sig)) / fs

    return t_picks / T_MAX   # normalise


def stalta_pick(gather_raw: np.ndarray, fs: float = FS_RAW,
                sta_s: float = 0.05, lta_s: float = 1.0,
                threshold: float = 3.0) -> np.ndarray:
    """
    STA/LTA onset picker (Allen, 1978).

    STA window: 50 ms, LTA window: 1000 ms, threshold: 3.0.
    Falls back to argmax if no trigger found.
    """
    n_ch, n_t = gather_raw.shape
    sta_n = int(sta_s * fs)
    lta_n = int(lta_s * fs)

    t_picks = np.zeros(n_ch, dtype=np.float32)
    for ch in range(n_ch):
        sig2 = gather_raw[ch] ** 2
        sta  = np.convolve(sig2, np.ones(sta_n) / sta_n, mode='same')
        lta  = np.convolve(sig2, np.ones(lta_n) / lta_n, mode='same')
        ratio = sta / (lta + 1e-10)
        idx   = np.where(ratio > threshold)[0]
        t_picks[ch] = (idx[0] / fs) if len(idx) > 0 else np.argmax(ratio) / fs

    return t_picks / T_MAX


def evaluate_baselines(raw_h5: str, n_eval: int = 800) -> Dict[str, dict]:
    """Evaluate FK-Pick and STA/LTA on the test set."""
    with h5py.File(raw_h5, 'r') as f:
        grp  = f['test']
        n    = min(n_eval, grp['gather'].shape[0])
        snr  = grp['snr_db'][:n]

        tp_fk  = np.zeros((n, N_CH), dtype=np.float32)
        tp_sta = np.zeros((n, N_CH), dtype=np.float32)
        tp_gt  = np.clip(grp['t_p'][:n] / T_MAX, 0, 1)

        for i in tqdm(range(n), desc='baselines'):
            raw = grp['gather'][i]
            tp_fk[i]  = fk_pick(raw)
            tp_sta[i] = stalta_pick(raw)

    return {
        'FK-Pick': compute_metrics(tp_fk,  tp_gt, snr),
        'STA/LTA': compute_metrics(tp_sta, tp_gt, snr),
    }


# ── Deep-learning model evaluation ───────────────────────────────────────────

def evaluate_model(
    model_type: str,
    ckpt_path:  str,
    feat_h5:    str = FEAT_H5,
) -> Dict[str, float]:
    """
    Evaluate a trained TRITON model on the test set.
    Returns metrics dict matching Table 1.
    """
    from triton.train import TRITONDataset, _forward
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    if model_type == 'cnn_v2':
        from triton.models.cnn_v2 import build_cnn_v2
        model = build_cnn_v2().to(device)
    else:
        from triton.models.mirror import build_mirror
        model = build_mirror().to(device)

    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck['model_state'])
    model.eval()
    print(f'Loaded {model_type} checkpoint  '
          f'(val MAE = {ck["val_mae"]*T_MAX*1000:.1f} ms)')

    from torch.utils.data import DataLoader
    ds = TRITONDataset(feat_h5, 'test')
    loader = DataLoader(ds, batch_size=64, shuffle=False,
                        num_workers=4, pin_memory=True)

    all_pred = []
    all_gt   = []
    all_snr  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=model_type):
            bd    = {k: v.to(device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            preds = _forward(model, model_type, bd)
            all_pred.append(preds['t_p'].cpu().numpy())
            all_gt.append(batch['t_p_norm'].numpy())
            all_snr.append(batch['snr_db'].numpy())

    t_p_pred = np.concatenate(all_pred)   # (N, n_ch)
    t_p_gt   = np.concatenate(all_gt)
    snr      = np.concatenate(all_snr)

    return compute_metrics(t_p_pred, t_p_gt, snr)


# ── Print table ──────────────────────────────────────────────────────────────

def print_table(results: Dict[str, dict]) -> None:
    """Print Table 1 in plain text."""
    hdr = ['Method', 'MAE(ms)', 'R²', 'Moveout err(ms)',
           'Dir err(%)', 'SNR<8dB MAE(ms)']
    print('\n' + '='*75)
    print(f'{"Method":<12} {"MAE":>8} {"R²":>8} '
          f'{"MO err":>12} {"Dir err%":>10} {"LowSNR":>12}')
    print('-'*75)
    for name, m in results.items():
        print(f'{name:<12} '
              f'{m["MAE_ms"]:>8.0f} '
              f'{m["R2"]:>8.4f} '
              f'{m["moveout_err"]:>12.0f} '
              f'{m["dir_err_pct"]:>10.1f} '
              f'{m.get("MAE_snr8_ms", float("nan")):>12.0f}')
    print('='*75 + '\n')


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate TRITON models (reproduces Table 1)'
    )
    parser.add_argument('--model',       choices=['cnn_v2', 'mirror'])
    parser.add_argument('--checkpoint',  default=None)
    parser.add_argument('--feat-h5',     default=FEAT_H5)
    parser.add_argument('--raw-h5',      default=RAW_H5)
    parser.add_argument('--baselines-only', action='store_true')
    parser.add_argument('--output-json', default=None)
    args = parser.parse_args()

    results = {}

    # Baselines
    if args.baselines_only or args.model is None:
        print('=== Evaluating baselines ===')
        bl = evaluate_baselines(args.raw_h5)
        results.update(bl)
        print_table(results)

    # Deep-learning model
    if args.model and args.checkpoint:
        print(f'=== Evaluating {args.model} ===')
        m = evaluate_model(args.model, args.checkpoint, args.feat_h5)
        results[args.model] = m
        print_table({args.model: m})

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Results saved to {args.output_json}')


if __name__ == '__main__':
    main()
