"""
003_cnn_v2.py
-------------
CNN_v2: UNet-based P-wave identification network.

Corresponds to: "CNN_v2 Architecture (4.7 M Parameters)" in Methods.

Architecture:
  Encoder: 7×7 stem (c=32) + 3 strided blocks (c,2c,4c,8c)
           each block = stride-2 CBG + RB(dil=1) + RB(dil=2)
  Decoder: 3 × (TransposedConv + skip-cat + RB)
  Heads:
    - ArrivalHead (soft-argmax): t̂_p, t̂_ac   equation (1)
    - DetectionHead (binary): P-wave present?

Total parameters: ~4.7 M

Usage:
    from triton.models.cnn_v2 import CNNv2
    model = CNNv2(c=32, n_ch=128, nt=2048)
    out = model(gather_norm)   # {'t_p': (B,128), 't_ac': (B,128), 'det': (B,1)}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ── Building blocks ──────────────────────────────────────────────────────────

class ConvBnGelu(nn.Module):
    """Conv2d → BatchNorm2d → GELU."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1, pad: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with dilated convolutions."""
    def __init__(self, ch: int, dilation: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3,
                            padding=dilation, dilation=dilation, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3,
                            padding=dilation, dilation=dilation, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.b1(self.c1(x)))
        return F.gelu(x + self.b2(self.c2(h)))


# ── Encoder ──────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    4-stage encoder: stem + 3 strided blocks.
    Channel progression: 1 → c → 2c → 4c → 8c.
    """
    def __init__(self, c: int = 32):
        super().__init__()
        self.stem = ConvBnGelu(1, c, kernel=7, pad=3)
        self.s1 = nn.Sequential(
            ConvBnGelu(c,     c * 2, stride=2),
            ResidualBlock(c * 2, dilation=1),
            ResidualBlock(c * 2, dilation=2),
        )
        self.s2 = nn.Sequential(
            ConvBnGelu(c * 2, c * 4, stride=2),
            ResidualBlock(c * 4, dilation=1),
            ResidualBlock(c * 4, dilation=2),
            ResidualBlock(c * 4, dilation=4),
        )
        self.s3 = nn.Sequential(
            ConvBnGelu(c * 4, c * 8, stride=2),
            ResidualBlock(c * 8, dilation=1),
            ResidualBlock(c * 8, dilation=2),
        )

    def forward(self, x: torch.Tensor):
        s0 = self.stem(x)
        s1 = self.s1(s0)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        return s0, s1, s2, s3


# ── Decoder ──────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """3-stage decoder with skip connections from encoder."""
    def __init__(self, c: int = 32):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.d3  = nn.Sequential(ConvBnGelu(c * 8, c * 4), ResidualBlock(c * 4))
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.d2  = nn.Sequential(ConvBnGelu(c * 4, c * 2), ResidualBlock(c * 2))
        self.up1 = nn.ConvTranspose2d(c * 2, c,     kernel_size=2, stride=2)
        self.d1  = nn.Sequential(ConvBnGelu(c * 2, c),     ResidualBlock(c))

    def _cat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, s0, s1, s2, s3) -> torch.Tensor:
        x = self.d3(self._cat(self.up3(s3), s2))
        x = self.d2(self._cat(self.up2(x),  s1))
        x = self.d1(self._cat(self.up1(x),  s0))
        return x


# ── Arrival-time head (soft-argmax, equation 1) ──────────────────────────────

class ArrivalHead(nn.Module):
    """
    Soft-argmax arrival-time head.

    Equation (1):  t̂(ch) = Σ_k p(ch, k) · (k / n_t)

    Resolution: ±(T / n_t) seconds  (±7.3 ms at 500 Hz, T=15 s, n_t=2048)
    """
    def __init__(self, in_ch: int, n_ch: int, n_t: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, 1, kernel_size=1)
        self.n_ch = n_ch
        self.n_t  = n_t

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # Upsample to (n_ch, n_t)
        logits = F.interpolate(
            self.proj(feat),
            size=(self.n_ch, self.n_t),
            mode='bilinear', align_corners=False,
        ).squeeze(1)                                         # (B, n_ch, n_t)

        # Row-wise softmax → temporal probability distribution
        prob = torch.softmax(logits, dim=-1)                 # (B, n_ch, n_t)

        # Soft-argmax: expected normalized time in [0, 1]
        t_idx = torch.linspace(0.0, 1.0, self.n_t,
                               device=feat.device)          # (n_t,)
        t_hat = (prob * t_idx[None, None, :]).sum(dim=-1)   # (B, n_ch)
        return t_hat


# ── Detection head ───────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """Binary P-wave detection head."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, in_ch // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(in_ch // 2, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.linear(self.pool(feat))


# ── Full CNN_v2 model ─────────────────────────────────────────────────────────

class CNNv2(nn.Module):
    """
    CNN_v2: UNet P-wave identification network.

    Parameters
    ----------
    c : int
        Base channel count (default 32 → 4.7 M params).
    n_ch : int
        Number of DAS channels.
    nt : int
        Number of output time samples.
    """
    def __init__(self, c: int = 32, n_ch: int = 128, nt: int = 2048):
        super().__init__()
        self.encoder   = Encoder(c)
        self.decoder   = Decoder(c)
        self.head_p    = ArrivalHead(c, n_ch, nt)   # P-wave
        self.head_ac   = ArrivalHead(c, n_ch, nt)   # acoustic
        self.head_det  = DetectionHead(c)

    def forward(self, gather_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        gather_norm : torch.Tensor, shape (B, 1, n_ch, nt)
            Channel-normalised gather (output of feature extraction).

        Returns
        -------
        dict with keys:
            't_p'  : (B, n_ch)  normalised P-wave arrival times  [0, 1]
            't_ac' : (B, n_ch)  normalised acoustic arrival times [0, 1]
            'det'  : (B, 1)     P-wave detection logit
        """
        s0, s1, s2, s3 = self.encoder(gather_norm)
        feat = self.decoder(s0, s1, s2, s3)
        return {
            't_p':  self.head_p(feat),
            't_ac': self.head_ac(feat),
            'det':  self.head_det(feat),
        }

    @staticmethod
    def count_params(model: 'CNNv2') -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Convenience constructor ───────────────────────────────────────────────────

def build_cnn_v2(c: int = 32, n_ch: int = 128, nt: int = 2048) -> CNNv2:
    """Build CNN_v2 and print parameter count."""
    model = CNNv2(c=c, n_ch=n_ch, nt=nt)
    n_params = CNNv2.count_params(model)
    print(f'CNN_v2 | c={c} | n_ch={n_ch} | nt={nt} | params={n_params:,}')
    return model


if __name__ == '__main__':
    model = build_cnn_v2()
    x = torch.randn(2, 1, 128, 2048)
    out = model(x)
    print('t_p  shape:', out['t_p'].shape)
    print('t_ac shape:', out['t_ac'].shape)
    print('det  shape:', out['det'].shape)
