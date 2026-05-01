"""
008_inference.py
----------------
Single-gather inference with TRITON.

Given a raw DAS gather (numpy array or HDF5), outputs per-channel
P-wave and acoustic arrival times in seconds.

Usage:
    from triton.inference import TRITONInference
    inf = TRITONInference('models/cnn_v2_best.pt', model_type='cnn_v2')
    t_p, t_ac = inf.predict(gather_raw)   # seconds

    # CLI
    python scripts/008_inference.py \
        --checkpoint models/cnn_v2_best.pt \
        --model cnn_v2 \
        --input path/to/gather.npy \
        --output results/picks.json
"""

import argparse
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch


# ── Constants ─────────────────────────────────────────────────────────────────
T_MAX  = 15.0
N_CH   = 128
NT_OUT = 2048
FS_RAW = 500.0


# ── Inference class ──────────────────────────────────────────────────────────

class TRITONInference:
    """
    Wrapper for TRITON inference on a single DAS gather.

    Parameters
    ----------
    checkpoint : str
        Path to .pt checkpoint file (output of 006_train.py).
    model_type : str
        'cnn_v2' or 'mirror'.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    fs : float
        Sampling rate of the input gather [Hz].
    t_max : float
        Gather duration [s].
    """
    def __init__(
        self,
        checkpoint:  str,
        model_type:  str   = 'cnn_v2',
        device:      Optional[str] = None,
        fs:          float = FS_RAW,
        t_max:       float = T_MAX,
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device     = device
        self.model_type = model_type
        self.fs         = fs
        self.t_max      = t_max

        # Build model
        if model_type == 'cnn_v2':
            from triton.models.cnn_v2 import build_cnn_v2
            self.model = build_cnn_v2().to(device)
        elif model_type == 'mirror':
            from triton.models.mirror import build_mirror
            self.model = build_mirror().to(device)
        else:
            raise ValueError(f'Unknown model_type: {model_type}')

        # Load weights
        ck = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(ck['model_state'])
        self.model.eval()
        val_mae_ms = ck.get('val_mae', float('nan')) * t_max * 1000
        print(f'Loaded {model_type}  val_MAE={val_mae_ms:.1f} ms  '
              f'device={device}')

    @torch.no_grad()
    def predict(
        self,
        gather_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict P-wave and acoustic arrival times.

        Parameters
        ----------
        gather_raw : np.ndarray, shape (n_ch, n_t)
            Raw DAS gather at self.fs Hz.

        Returns
        -------
        t_p  : np.ndarray (n_ch,)  P-wave arrival times [s]
        t_ac : np.ndarray (n_ch,)  acoustic arrival times [s]
        """
        from triton.data.features import extract_features
        feats = extract_features(gather_raw, fs=self.fs, nt_out=NT_OUT)

        gn = torch.from_numpy(feats['gather_norm'][np.newaxis]).to(self.device)

        if self.model_type == 'cnn_v2':
            out = self.model(gn)
        else:
            fk = torch.from_numpy(
                feats['fk_spectrum'][np.newaxis]).to(self.device)
            out = self.model(gn, fk)

        t_p  = out['t_p'][0].cpu().numpy()  * self.t_max
        t_ac = out['t_ac'][0].cpu().numpy() * self.t_max
        return t_p, t_ac

    def predict_batch(
        self,
        gathers: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict for a batch of gathers.

        Parameters
        ----------
        gathers : np.ndarray, shape (B, n_ch, n_t)

        Returns
        -------
        t_p  : (B, n_ch)  P-wave arrivals [s]
        t_ac : (B, n_ch)  acoustic arrivals [s]
        """
        from triton.data.features import extract_features
        B = gathers.shape[0]
        gn_list = []
        fk_list = []
        for i in range(B):
            feats = extract_features(gathers[i], fs=self.fs, nt_out=NT_OUT)
            gn_list.append(feats['gather_norm'])
            fk_list.append(feats['fk_spectrum'])

        gn = torch.from_numpy(
            np.stack(gn_list)).to(self.device)   # (B,1,n_ch,nt)

        if self.model_type == 'cnn_v2':
            out = self.model(gn)
        else:
            fk = torch.from_numpy(
                np.stack(fk_list)).to(self.device)
            out = self.model(gn, fk)

        t_p  = out['t_p'].cpu().numpy()  * self.t_max
        t_ac = out['t_ac'].cpu().numpy() * self.t_max
        return t_p, t_ac

    def predict_and_visualise(
        self,
        gather_raw: np.ndarray,
        out_pdf:    Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict and optionally save a summary PDF.
        Returns (t_p, t_ac) in seconds.
        """
        t_p, t_ac = self.predict(gather_raw)

        if out_pdf is not None:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from triton.data.features import extract_features

            feats = extract_features(gather_raw, fs=self.fs, nt_out=NT_OUT)
            gn    = feats['gather_norm'][0]   # (n_ch, nt_out)
            t_v   = np.linspace(0, self.t_max, NT_OUT)
            x_arr = np.arange(gn.shape[0]) * 12.5 / 1000   # km

            fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                                     facecolor='white')

            # Gather image
            vm = np.percentile(np.abs(gn), 97)
            axes[0].imshow(
                gn.T, aspect='auto', cmap='seismic',
                extent=[x_arr[0], x_arr[-1], t_v[-1], t_v[0]],
                vmin=-vm, vmax=vm, interpolation='bilinear',
            )
            axes[0].plot(x_arr, t_p,  'g--', lw=2, label='P-wave (pred)')
            axes[0].plot(x_arr, t_ac, 'b-',  lw=1, label='Acoustic (pred)')
            axes[0].set_xlabel('Offset (km)')
            axes[0].set_ylabel('Time (s)')
            axes[0].set_title(f'TRITON-{self.model_type.upper()} Predictions')
            axes[0].legend(loc='lower right', fontsize=9)

            # Moveout
            axes[1].plot(x_arr, t_p,  'g--', lw=2, label='P-wave')
            axes[1].plot(x_arr, t_ac, 'b-',  lw=1, label='Acoustic')
            mo_p = (t_p[-1] - t_p[0]) * 1000
            axes[1].set_xlabel('Offset (km)')
            axes[1].set_ylabel('Arrival time (s)')
            axes[1].set_title(f'Moveout  P: {mo_p:.0f} ms')
            axes[1].legend(fontsize=9)
            axes[1].invert_yaxis()

            plt.tight_layout()
            os.makedirs(os.path.dirname(os.path.abspath(out_pdf)),
                        exist_ok=True)
            fig.savefig(out_pdf, dpi=200, bbox_inches='tight',
                        facecolor='white')
            plt.close(fig)
            print(f'Figure saved: {out_pdf}')

        return t_p, t_ac


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='TRITON single-gather inference'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pt checkpoint')
    parser.add_argument('--model', choices=['cnn_v2', 'mirror'],
                        default='cnn_v2')
    parser.add_argument('--input',  required=True,
                        help='Input gather (.npy, shape (n_ch, n_t))')
    parser.add_argument('--output', default='results/picks.json',
                        help='Output JSON with per-channel picks [s]')
    parser.add_argument('--figure', default=None,
                        help='Optional output PDF for visualisation')
    parser.add_argument('--fs', type=float, default=FS_RAW)
    args = parser.parse_args()

    # Load gather
    gather = np.load(args.input)
    print(f'Input gather: {gather.shape}  fs={args.fs} Hz')

    # Inference
    inf = TRITONInference(
        args.checkpoint, model_type=args.model, fs=args.fs)

    if args.figure:
        t_p, t_ac = inf.predict_and_visualise(gather, out_pdf=args.figure)
    else:
        t_p, t_ac = inf.predict(gather)

    mo_p  = (t_p[-1]  - t_p[0])  * 1000
    mo_ac = (t_ac[-1] - t_ac[0]) * 1000
    print(f'P-wave  moveout: {mo_p:.1f} ms')
    print(f'Acoustic moveout: {mo_ac:.1f} ms')

    # Save picks
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    picks = {
        'model_type':       args.model,
        'checkpoint':       args.checkpoint,
        'n_channels':       int(t_p.shape[0]),
        't_p_seconds':      t_p.tolist(),
        't_ac_seconds':     t_ac.tolist(),
        'moveout_p_ms':     float(mo_p),
        'moveout_ac_ms':    float(mo_ac),
    }
    with open(args.output, 'w') as f:
        json.dump(picks, f, indent=2)
    print(f'Picks saved: {args.output}')


if __name__ == '__main__':
    main()
