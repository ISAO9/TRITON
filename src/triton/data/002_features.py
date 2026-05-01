"""
002_features.py
---------------
Feature extraction pipeline for TRITON.

Corresponds to: "Feature Extraction" subsection of Methods (manuscript).

Pipeline:
  1. Zero-phase 4th-order Butterworth bandpass filter (1–40 Hz)
  2. Linear resampling to n_t = 2,048 samples
  3. Channel-wise z-score normalization (ε = 1e-10)
  4. 2-D FFT → log-amplitude → global z-score  [f-k spectrum]

Input:  raw gather (n_ch, NT_RAW)  float32, 500 Hz
Output:
  gather_norm  (1, n_ch, NT_OUT)  float32
  fk_spectrum  (1, n_ch, n_ch)   float32

Usage:
    from triton.data.features import extract_features, build_feature_h5
    feats = extract_features(gather_raw)   # single gather
    build_feature_h5("data/synthetic.h5", "data/features.h5")
"""

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import h5py
import os
from tqdm import tqdm
from typing import Dict


# ── Constants ────────────────────────────────────────────────────────────────
FS_RAW  = 500.0    # Hz  raw sampling rate
NT_RAW  = 7500     # raw samples  (15 s × 500 Hz)
NT_OUT  = 2048     # output samples after resampling
N_CH    = 128
T_MAX   = 15.0     # s  gather duration
EPS     = 1e-10    # normalization stability constant

# BPF design: 1–40 Hz zero-phase
_BPF_LOW  = 1.0
_BPF_HIGH = 40.0


def _butter_bandpass(fs: float, low: float = _BPF_LOW,
                     high: float = _BPF_HIGH, order: int = 4):
    """Design a Butterworth bandpass filter."""
    nyq = fs / 2.0
    return butter(order, [low / nyq, high / nyq], btype='band')


def extract_features(
    gather_raw: np.ndarray,
    fs: float   = FS_RAW,
    nt_out: int = NT_OUT,
) -> Dict[str, np.ndarray]:
    """
    Extract features from a single raw DAS gather.

    Parameters
    ----------
    gather_raw : np.ndarray, shape (n_ch, n_t_raw)
        Raw DAS gather at original sampling rate.
    fs : float
        Sampling rate of gather_raw [Hz].
    nt_out : int
        Target number of time samples after resampling.

    Returns
    -------
    dict with keys:
        'gather_norm'  : np.ndarray (1, n_ch, nt_out)  channel-normalised gather
        'fk_spectrum'  : np.ndarray (1, n_ch, n_ch)    log f-k spectrum
    """
    n_ch, n_t_raw = gather_raw.shape

    # Step 1: Bandpass filter (zero-phase)
    b, a = _butter_bandpass(fs)
    gather_filt = filtfilt(b, a, gather_raw, axis=1).astype(np.float32)

    # Step 2: Temporal resampling to nt_out
    t_orig = np.linspace(0.0, 1.0, n_t_raw)
    t_new  = np.linspace(0.0, 1.0, nt_out)
    gather_rs = np.zeros((n_ch, nt_out), dtype=np.float32)
    for ch in range(n_ch):
        gather_rs[ch] = interp1d(
            t_orig, gather_filt[ch], kind='linear'
        )(t_new)

    # Step 3: Channel-wise z-score normalization
    mu  = gather_rs.mean(axis=1, keepdims=True)
    std = gather_rs.std(axis=1,  keepdims=True) + EPS
    gather_norm = ((gather_rs - mu) / std).astype(np.float32)

    # Step 4: 2-D FFT → log-amplitude → global z-score
    fk  = np.fft.fft2(gather_norm, s=(n_ch, n_ch))
    fk_amp  = np.log(np.abs(fk) + 1e-6).astype(np.float32)
    fk_norm = ((fk_amp - fk_amp.mean()) / (fk_amp.std() + EPS)).astype(np.float32)

    return {
        'gather_norm': gather_norm[np.newaxis],   # (1, n_ch, nt_out)
        'fk_spectrum': fk_norm[np.newaxis],        # (1, n_ch, n_ch)
    }


def build_feature_h5(
    raw_h5:  str,
    feat_h5: str,
    nt_out:  int   = NT_OUT,
    fs_raw:  float = FS_RAW,
    splits:  tuple = ('train', 'val', 'test'),
) -> None:
    """
    Build feature HDF5 from raw synthetic dataset.

    Reads gather arrays from raw_h5, applies the full feature extraction
    pipeline, and writes normalised gather + f-k spectrum to feat_h5.

    Parameters
    ----------
    raw_h5 : str
        Path to raw dataset (output of 001_generator.py).
    feat_h5 : str
        Output path for feature dataset.
    nt_out : int
        Number of output time samples.
    fs_raw : float
        Sampling rate of raw gathers.
    splits : tuple of str
        Dataset splits to process.
    """
    import time
    os.makedirs(os.path.dirname(os.path.abspath(feat_h5)), exist_ok=True)
    t0 = time.time()

    with h5py.File(raw_h5, 'r') as fr, h5py.File(feat_h5, 'w') as fw:
        # Copy global attributes
        fw.attrs['out_nt'] = nt_out
        fw.attrs['t_max']  = T_MAX
        fw.attrs['fs_raw'] = fs_raw
        if 'dx' in fr.attrs:
            fw.attrs['dx'] = float(fr.attrs['dx'])

        for split in splits:
            if split not in fr:
                print(f'  {split}: not found, skipping')
                continue
            gr = fr[split]
            gw = fw.create_group(split)
            n  = gr['gather'].shape[0]
            n_ch = gr['gather'].shape[1]

            # Allocate output datasets
            ds_gn = gw.create_dataset(
                'gather_norm', (n, 1, n_ch, nt_out), dtype=np.float32)
            ds_fk = gw.create_dataset(
                'fk_spectrum', (n, 1, n_ch, n_ch),   dtype=np.float32)

            # Copy label arrays verbatim
            for key in ('t_p', 't_ac', 'snr_db', 'vp', 'offset_m'):
                if key in gr:
                    gw.create_dataset(key, data=gr[key][:])

            # Extract features sample by sample
            for i in tqdm(range(n), desc=f'{split:5s}', leave=True):
                feats = extract_features(gr['gather'][i], fs=fs_raw,
                                         nt_out=nt_out)
                ds_gn[i] = feats['gather_norm']
                ds_fk[i] = feats['fk_spectrum']

    elapsed = time.time() - t0
    print(f'Features saved to {feat_h5}  [{elapsed:.0f} s]')


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract features from raw synthetic dataset (Paper: Feature Extraction)'
    )
    parser.add_argument('--raw',   default='data/synthetic.h5',
                        help='Input raw HDF5 path')
    parser.add_argument('--feat',  default='data/features.h5',
                        help='Output feature HDF5 path')
    parser.add_argument('--nt-out', type=int, default=NT_OUT)
    args = parser.parse_args()

    build_feature_h5(args.raw, args.feat, nt_out=args.nt_out)
