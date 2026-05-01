"""
001_generator.py
----------------
Synthetic marine DAS shot gather generator.

Corresponds to: "Synthetic Data Generation" section of the manuscript.

Each gather superimposes four phases:
  1. Direct water-column acoustic arrival (v_ac = 1500 m/s)
  2. Crustal P-wave head wave (v_p ~ Uniform[5000, 6500] m/s)
  3. PS-converted phase (v_ps ~ Uniform[2500, 4000] m/s)
  4. Scholte wave (1000–1300 m/s, dispersive)

Noise model: swell (1/f^2), shipping (flat), laser, broadband 1/f^2.

Usage:
    from triton.data.generator import generate_gather, generate_dataset
    gather, labels = generate_gather()
    generate_dataset("data/synthetic.h5", n_train=4000)
"""

import numpy as np
from scipy.signal import butter, filtfilt
import h5py
import os
import time
from tqdm import tqdm


# ── Physical constants ──────────────────────────────────────────────────────
V_ACOUSTIC  = 1500.0   # m/s  water-column acoustic velocity
FS          = 500.0    # Hz   sampling rate
DX          = 12.5     # m    channel spacing
N_CH        = 128      # number of channels
T_TOTAL     = 15.0     # s    gather duration
NT          = 7500     # samples per channel (15 s × 500 Hz)


def _ricker(f_peak: float, n_samples: int, dt: float) -> np.ndarray:
    """Ricker (Mexican hat) wavelet centred at n_samples//2."""
    t = (np.arange(n_samples) - n_samples // 2) * dt
    pft2 = (np.pi * f_peak * t) ** 2
    return (1.0 - 2.0 * pft2) * np.exp(-pft2)


def _noise_1f2(n_ch: int, n_t: int, fs: float,
                rng: np.random.Generator) -> np.ndarray:
    """Generate spatially independent 1/f^2 noise."""
    noise = rng.standard_normal((n_ch, n_t)).astype(np.float32)
    freq  = np.fft.rfftfreq(n_t, d=1.0 / fs)
    freq[0] = 1.0
    spec  = np.fft.rfft(noise, axis=1)
    spec /= freq[np.newaxis, :]
    return np.fft.irfft(spec, n=n_t, axis=1).astype(np.float32)


def generate_gather(
    n_ch: int = N_CH,
    n_t:  int = NT,
    fs:   float = FS,
    dx:   float = DX,
    rng:  np.random.Generator = None,
) -> tuple[np.ndarray, dict]:
    """
    Generate one synthetic marine DAS shot gather.

    Parameters
    ----------
    n_ch : int
        Number of DAS channels.
    n_t : int
        Number of time samples.
    fs : float
        Sampling rate [Hz].
    dx : float
        Channel spacing [m].
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    gather : np.ndarray, shape (n_ch, n_t), float32
        Synthetic DAS gather (strain-rate proxy).
    labels : dict
        Ground-truth labels and physical parameters:
          t_p      (n_ch,)  P-wave arrival times [s]
          t_ac     (n_ch,)  acoustic arrival times [s]
          vp       float    P-wave apparent velocity [m/s]
          snr_db   float    signal-to-noise ratio [dB]
          offset_m float    nearest source-receiver offset [m]
    """
    if rng is None:
        rng = np.random.default_rng()

    dt       = 1.0 / fs
    t_total  = n_t * dt
    x_arr    = np.arange(n_ch) * dx        # offsets from first channel [m]

    # ── Source parameters ────────────────────────────────────────────────
    src_offset = rng.uniform(5_000, 20_000)   # m  nearest source offset
    src_depth  = rng.uniform(5, 10)            # m  source depth
    water_depth= rng.uniform(500, 3_000)       # m  water depth

    # Acoustic amplitude scale factor (8–30× P-wave)
    A_ratio    = rng.uniform(8, 30)
    snr_db     = rng.uniform(4, 22)

    # ── Phase 1: Direct acoustic arrival ────────────────────────────────
    v_ac  = V_ACOUSTIC
    f_ac  = rng.uniform(5, 30)              # dominant frequency [Hz]
    amp_ac= rng.uniform(0.8, 1.0)
    wl_ac = int(fs * 0.2)                   # 200 ms wavelet window

    # Slant moveout
    t_ac  = (src_offset + x_arr) / v_ac    # [s]
    # Clamp to valid window
    if t_ac.min() < 0.3 or t_ac.max() > t_total - 0.5:
        # Regenerate: invalid geometry
        return None, None

    gather = np.zeros((n_ch, n_t), dtype=np.float32)

    # Direct wave
    rk_ac = _ricker(f_ac, wl_ac, dt)
    for ch in range(n_ch):
        i0  = int(t_ac[ch] * fs)
        geo = 1.0 / np.sqrt(src_offset + x_arr[ch] + 1.0)
        is_ = max(0, i0 - wl_ac // 2)
        ie_ = min(n_t, i0 + wl_ac // 2)
        rs_ = is_ - (i0 - wl_ac // 2)
        re_ = rs_ + (ie_ - is_)
        gather[ch, is_:ie_] += amp_ac * geo * rk_ac[rs_:re_]

    # Water-column multiples (3 reverberations)
    for m in range(1, 4):
        delay = 2 * m * water_depth / v_ac
        amp_m = amp_ac * (0.5 ** m)
        f_m   = f_ac * rng.uniform(0.8, 1.2)
        rk_m  = _ricker(f_m, wl_ac, dt)
        for ch in range(n_ch):
            i0  = int((t_ac[ch] + delay) * fs)
            if i0 < 0 or i0 >= n_t - wl_ac // 2:
                continue
            geo = 1.0 / np.sqrt(src_offset + x_arr[ch] + 1.0)
            is_ = max(0, i0 - wl_ac // 2)
            ie_ = min(n_t, i0 + wl_ac // 2)
            rs_ = is_ - (i0 - wl_ac // 2)
            re_ = rs_ + (ie_ - is_)
            gather[ch, is_:ie_] += amp_m * geo * rk_m[rs_:re_]

    # ── Phase 2: Crustal P-wave head wave ────────────────────────────────
    vp    = rng.uniform(5_000, 6_500)        # apparent velocity [m/s]
    f_p   = rng.uniform(2, 12)
    amp_p = amp_ac / A_ratio
    wl_p  = int(fs * 0.3)

    # Intercept time: P arrives after acoustic, before T - 2 s
    t_p0_min = t_ac.min() + 0.5
    t_p0_max = t_total - 2.0
    if t_p0_min >= t_p0_max:
        return None, None
    t_p0 = rng.uniform(t_p0_min, t_p0_max)
    t_p  = t_p0 + x_arr / vp              # slant moveout [s]

    if t_p.max() > t_total - 0.5:
        return None, None

    rk_p  = _ricker(f_p, wl_p, dt)
    for ch in range(n_ch):
        i0  = int(t_p[ch] * fs)
        geo = 1.0 / np.sqrt(src_offset + x_arr[ch] + 1.0)
        is_ = max(0, i0 - wl_p // 2)
        ie_ = min(n_t, i0 + wl_p // 2)
        rs_ = is_ - (i0 - wl_p // 2)
        re_ = rs_ + (ie_ - is_)
        gather[ch, is_:ie_] += amp_p * geo * rk_p[rs_:re_]

        # P-wave coda (2–5 reverberations)
        for _ in range(rng.integers(2, 6)):
            dt_c  = rng.uniform(0.05, 0.5)
            i0_c  = i0 + int(dt_c * fs)
            if i0_c >= n_t - wl_p // 2:
                continue
            amp_c = rng.uniform(0.05, 0.3) * amp_p
            f_c   = rng.uniform(1, 8)
            rk_c  = _ricker(f_c, wl_p, dt)
            is_c  = max(0, i0_c - wl_p // 2)
            ie_c  = min(n_t, i0_c + wl_p // 2)
            rs_c  = is_c - (i0_c - wl_p // 2)
            re_c  = rs_c + (ie_c - is_c)
            gather[ch, is_c:ie_c] += amp_c * rk_c[rs_c:re_c]

    # ── Phase 3: PS-converted phase ──────────────────────────────────────
    v_ps  = rng.uniform(2_500, 4_000)
    f_ps  = rng.uniform(1, 6)
    amp_ps= rng.uniform(0.1, 0.4) * amp_p
    wl_ps = int(fs * 0.25)
    t_ps0 = rng.uniform(t_ac.min() + 0.2, t_p0 - 0.1)
    if t_ps0 > 0:
        t_ps = t_ps0 + x_arr / v_ps
        rk_ps = _ricker(f_ps, wl_ps, dt)
        for ch in range(n_ch):
            i0  = int(t_ps[ch] * fs)
            if i0 < 0 or i0 >= n_t - wl_ps // 2:
                continue
            is_ = max(0, i0 - wl_ps // 2)
            ie_ = min(n_t, i0 + wl_ps // 2)
            rs_ = is_ - (i0 - wl_ps // 2)
            re_ = rs_ + (ie_ - is_)
            gather[ch, is_:ie_] += amp_ps * rk_ps[rs_:re_]

    # ── Phase 4: Scholte wave (interface wave) ───────────────────────────
    v_sch = rng.uniform(1_000, 1_300)
    f_sch = rng.uniform(0.5, 3.0)
    amp_sch = rng.uniform(0.05, 0.2) * amp_p
    t_sch0  = rng.uniform(t_ac.min() + 1.0, t_total - 2.0)
    if t_sch0 < t_total - 1.0:
        for ch in range(n_ch):
            i0 = int((t_sch0 + x_arr[ch] / v_sch) * fs)
            if i0 < 0 or i0 >= n_t - int(fs * 0.5):
                continue
            t_env = np.arange(int(fs * 1.5)) / fs
            env   = np.exp(-t_env * 2.0) * np.sin(2 * np.pi * f_sch * t_env)
            ie_   = min(n_t, i0 + len(env))
            gather[ch, i0:ie_] += amp_sch * env[:ie_ - i0].astype(np.float32)

    # ── Noise ────────────────────────────────────────────────────────────
    # 1/f² swell noise (0.1–2 Hz)
    noise_swell = _noise_1f2(n_ch, n_t, fs, rng)
    b, a = butter(4, [0.1 / (fs / 2), 2.0 / (fs / 2)], btype='band')
    noise_swell = filtfilt(b, a, noise_swell, axis=1).astype(np.float32)

    # Flat shipping noise (10–100 Hz)
    noise_ship = rng.standard_normal((n_ch, n_t)).astype(np.float32)
    b, a = butter(4, [10.0 / (fs / 2), 100.0 / (fs / 2)], btype='band')
    noise_ship = filtfilt(b, a, noise_ship, axis=1).astype(np.float32)

    # Laser/interrogator noise (100–250 Hz)
    noise_laser = rng.standard_normal((n_ch, n_t)).astype(np.float32)
    b, a = butter(4, [100.0 / (fs / 2), min(249.0, fs / 2 - 1) / (fs / 2)],
                  btype='band')
    noise_laser = filtfilt(b, a, noise_laser, axis=1).astype(np.float32)

    # Broadband 1/f² background
    noise_bg = _noise_1f2(n_ch, n_t, fs, rng)

    # Combine noise (relative amplitudes validated against OOI RCA PSD)
    noise = (noise_swell * 1.0 + noise_ship * 0.3 +
             noise_laser * 0.1 + noise_bg  * 0.5).astype(np.float32)

    # Scale noise to target SNR
    sig_rms   = np.sqrt((gather ** 2).mean()) + 1e-10
    noise_rms = np.sqrt((noise  ** 2).mean()) + 1e-10
    noise_scale = sig_rms / noise_rms / (10 ** (snr_db / 20.0))
    gather += noise * noise_scale

    labels = dict(
        t_p      = t_p.astype(np.float32),
        t_ac     = t_ac.astype(np.float32),
        vp       = float(vp),
        snr_db   = float(snr_db),
        offset_m = float(src_offset),
        A_ratio  = float(A_ratio),
    )
    return gather.astype(np.float32), labels


def generate_dataset(
    out_path: str,
    n_train:  int = 4_000,
    n_val:    int = 800,
    n_test:   int = 800,
    seed:     int = 2024,
    n_ch:     int = N_CH,
    n_t:      int = NT,
    fs:       float = FS,
    dx:       float = DX,
) -> None:
    """
    Generate the full synthetic dataset and save to HDF5.

    Parameters
    ----------
    out_path : str
        Output file path (.h5).
    n_train, n_val, n_test : int
        Number of samples per split.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    t0 = time.time()

    with h5py.File(out_path, 'w') as f:
        f.attrs['fs']     = fs
        f.attrs['dx']     = dx
        f.attrs['n_ch']   = n_ch
        f.attrs['n_t']    = n_t
        f.attrs['t_total']= n_t / fs
        f.attrs['seed']   = seed

        for split, n in [('train', n_train), ('val', n_val), ('test', n_test)]:
            grp  = f.create_group(split)
            dg   = grp.create_dataset('gather',  (n, n_ch, n_t),
                                      dtype=np.float32,
                                      chunks=(1, n_ch, n_t))
            dp   = grp.create_dataset('t_p',     (n, n_ch), dtype=np.float32)
            dac  = grp.create_dataset('t_ac',    (n, n_ch), dtype=np.float32)
            dsnr = grp.create_dataset('snr_db',  (n,),      dtype=np.float32)
            dvp  = grp.create_dataset('vp',      (n,),      dtype=np.float32)
            doff = grp.create_dataset('offset_m',(n,),      dtype=np.float32)

            ok = 0
            attempts = 0
            pbar = tqdm(total=n, desc=f'{split:5s}')
            while ok < n:
                attempts += 1
                gather, labels = generate_gather(n_ch, n_t, fs, dx, rng)
                if gather is None:
                    continue
                dg[ok]   = gather
                dp[ok]   = labels['t_p']
                dac[ok]  = labels['t_ac']
                dsnr[ok] = labels['snr_db']
                dvp[ok]  = labels['vp']
                doff[ok] = labels['offset_m']
                ok += 1
                pbar.update(1)
            pbar.close()
            eff = ok / attempts * 100
            print(f'  {split}: {n} samples, {eff:.1f}% efficiency')

    elapsed = time.time() - t0
    print(f'Dataset saved to {out_path}  [{elapsed:.0f} s]')


# ── CLI ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate synthetic marine DAS dataset (Paper: Synthetic Data section)'
    )
    parser.add_argument('--output',   default='data/synthetic.h5')
    parser.add_argument('--n-train',  type=int, default=4_000)
    parser.add_argument('--n-val',    type=int, default=800)
    parser.add_argument('--n-test',   type=int, default=800)
    parser.add_argument('--seed',     type=int, default=2024)
    args = parser.parse_args()

    generate_dataset(
        out_path=args.output,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
    )
