"""
004_mirror.py
-------------
MIRROR: Dual-encoder P-wave identification network.

Corresponds to: "MIRROR Architecture (~15 M Parameters)" in Methods.

Architecture:
  Kähler encoder   : processes time-domain gather  (amplitude / moveout)
  Holomorphic enc. : processes f-k spectrum         (velocity / wavenumber)
  Cross-consistency bridge (eq. 2):  φ: z_A → ẑ_B,  φ⁻¹: z_B → ẑ_A
  Hodge decomposer  (eq. 3):  z → [z_P, z_ac, z_bg]  (orthogonal subspaces)
  Moveout GNN       (eq. 4):  inter-channel message passing
  Decoder + soft-argmax heads (eq. 1)

Total parameters: ~15 M

Usage:
    from triton.models.mirror import MIRROR
    model = MIRROR()
    out = model(gather_norm, fk_spectrum)
    # out: {'t_p': (B,128), 't_ac': (B,128), 'det': (B,1)}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# Re-use CNN_v2 building blocks
from triton.models.cnn_v2 import (
    ConvBnGelu, ResidualBlock, Decoder, ArrivalHead, DetectionHead,
    build_cnn_v2,
)


# ── Holomorphic encoder (3-stage, for f-k spectrum) ──────────────────────────

class HolomorphicEncoder(nn.Module):
    """
    3-stage convolutional encoder for the f-k spectrum.
    Input: (B, 1, n_ch, n_ch)  →  latent z_B ∈ ℝ^{d_lat}
    """
    def __init__(self, c: int = 32, d_lat: int = 256):
        super().__init__()
        self.s1 = nn.Sequential(
            ConvBnGelu(1, c, kernel=7, pad=3),
            ConvBnGelu(c, c * 2, stride=2),
            ResidualBlock(c * 2),
        )
        self.s2 = nn.Sequential(
            ConvBnGelu(c * 2, c * 4, stride=2),
            ResidualBlock(c * 4, dilation=1),
            ResidualBlock(c * 4, dilation=2),
        )
        self.s3 = nn.Sequential(
            ConvBnGelu(c * 4, c * 8, stride=2),
            ResidualBlock(c * 8),
        )
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.proj   = nn.Linear(c * 8, d_lat)

    def forward(self, fk: torch.Tensor) -> torch.Tensor:
        x = self.s3(self.s2(self.s1(fk)))
        x = self.pool(x).flatten(1)
        return self.proj(x)                  # (B, d_lat)


# ── Kähler encoder (wrapper around CNN_v2 Encoder + pool) ────────────────────

class KahlerEncoder(nn.Module):
    """
    Kähler encoder: processes time-domain gather.
    Identical structure to CNN_v2 Encoder; adds global-average-pool + proj.
    Also returns skip features {s0..s3} for the decoder.
    """
    def __init__(self, c: int = 32, d_lat: int = 256):
        super().__init__()
        from triton.models.cnn_v2 import Encoder
        self.enc  = Encoder(c)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c * 8, d_lat)

    def forward(self, gather: torch.Tensor):
        s0, s1, s2, s3 = self.enc(gather)
        z_A = self.proj(self.pool(s3).flatten(1))  # (B, d_lat)
        return z_A, (s0, s1, s2, s3)


# ── Cross-consistency bridge (equation 2) ────────────────────────────────────

class CrossConsistencyBridge(nn.Module):
    """
    Two-layer MLP implementing φ and φ⁻¹.

    Equation (2):
        ℒ_cross = ‖z_A − φ⁻¹(z_B)‖² + ‖z_B − φ(z_A)‖²

    The bridge enforces mutual predictability between the two latent spaces,
    preventing collapse onto the dominant acoustic arrival.
    """
    def __init__(self, d_lat: int = 256):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_lat, d_lat * 2),
            nn.GELU(),
            nn.Linear(d_lat * 2, d_lat),
        )
        self.phi_inv = nn.Sequential(
            nn.Linear(d_lat, d_lat * 2),
            nn.GELU(),
            nn.Linear(d_lat * 2, d_lat),
        )

    def forward(self, z_A: torch.Tensor,
                z_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z_hat_B : φ(z_A)    predicted B from A
        z_hat_A : φ⁻¹(z_B)  predicted A from B
        """
        return self.phi(z_A), self.phi_inv(z_B)

    @staticmethod
    def consistency_loss(z_A: torch.Tensor, z_B: torch.Tensor,
                         z_hat_B: torch.Tensor,
                         z_hat_A: torch.Tensor) -> torch.Tensor:
        """Compute ℒ_cross (equation 2)."""
        return (
            F.mse_loss(z_hat_A, z_A.detach()) +
            F.mse_loss(z_hat_B, z_B.detach())
        )


# ── Hodge decomposer (equation 3) ────────────────────────────────────────────

class HodgeDecomposer(nn.Module):
    """
    Projects fused latent vector onto three orthogonal subspaces:
      z_P   : P-wave content
      z_ac  : acoustic content
      z_bg  : background / noise

    Orthogonality enforced by Gram–Schmidt regularization:
    Equation (3):  ℒ_Hodge = Σ_{i≠j} |z_i · z_j| / (‖z_i‖‖z_j‖ + ε)
    """
    def __init__(self, d_lat: int = 256, d_sub: int = 64):
        super().__init__()
        self.proj_P  = nn.Linear(d_lat, d_sub)
        self.proj_ac = nn.Linear(d_lat, d_sub)
        self.proj_bg = nn.Linear(d_lat, d_sub)
        self.d_sub   = d_sub
        self.eps     = 1e-8

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        z_P  = self.proj_P(z)
        z_ac = self.proj_ac(z)
        z_bg = self.proj_bg(z)
        return z_P, z_ac, z_bg

    def orthogonality_loss(self, z_P: torch.Tensor,
                           z_ac: torch.Tensor,
                           z_bg: torch.Tensor) -> torch.Tensor:
        """Compute ℒ_Hodge (equation 3)."""
        eps = self.eps
        pairs = [(z_P, z_ac), (z_P, z_bg), (z_ac, z_bg)]
        loss = torch.tensor(0.0, device=z_P.device)
        for a, b in pairs:
            dot   = (a * b).sum(dim=-1).abs()
            norm_a = a.norm(dim=-1)
            norm_b = b.norm(dim=-1)
            loss  = loss + (dot / (norm_a * norm_b + eps)).mean()
        return loss


# ── Moveout GNN (equation 4) ─────────────────────────────────────────────────

class MoveoutGNN(nn.Module):
    """
    Inter-channel graph neural network for moveout correlation.

    Equation (4):
        m_i = Σ_{Δ} MLP([h_i ; h_{i+Δ} ; h_{i+Δ} − h_i]),  Δ ∈ {1,2,4,8}

    Aggregates messages from neighbours at fixed channel offsets.
    Input: feature map (B, c, n_ch, n_t)
    """
    def __init__(self, c: int = 32, offsets: tuple = (1, 2, 4, 8)):
        super().__init__()
        self.offsets = offsets
        d_msg = c * 3   # [h_i ; h_j ; h_j - h_i]
        self.msg_mlp = nn.Sequential(
            nn.Conv2d(d_msg, c * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c * 2, c, kernel_size=1),
        )
        self.update = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        feat : (B, c, n_ch, n_t)

        Returns
        -------
        feat_updated : (B, c, n_ch, n_t)
        """
        B, c, n_ch, n_t = feat.shape
        agg = torch.zeros_like(feat)

        for delta in self.offsets:
            # Shift along channel dimension with zero padding
            h_j = torch.roll(feat, shifts=-delta, dims=2)
            # Mask out wrapped-around channels
            if delta > 0:
                h_j[:, :, -delta:, :] = 0.0
            msg = torch.cat([feat, h_j, h_j - feat], dim=1)  # (B, 3c, n_ch, n_t)
            agg = agg + self.msg_mlp(msg)

        # Update: aggregate + original
        feat_updated = self.update(torch.cat([feat, agg], dim=1))
        return feat_updated


# ── Full MIRROR model ─────────────────────────────────────────────────────────

class MIRROR(nn.Module):
    """
    MIRROR: Dual-encoder P-wave identification network.

    Parameters
    ----------
    c : int
        Base channel count for spatial encoders/decoder.
    d_lat : int
        Latent vector dimensionality for cross-consistency bridge.
    d_sub : int
        Hodge subspace dimensionality.
    n_ch : int
        Number of DAS channels.
    nt : int
        Number of output time samples.
    """
    def __init__(
        self,
        c:     int = 32,
        d_lat: int = 256,
        d_sub: int = 64,
        n_ch:  int = 128,
        nt:    int = 2048,
    ):
        super().__init__()
        # Kähler encoder: time-domain gather
        self.kahler_enc    = KahlerEncoder(c=c, d_lat=d_lat)
        # Holomorphic encoder: f-k spectrum
        self.holo_enc      = HolomorphicEncoder(c=c, d_lat=d_lat)
        # Cross-consistency bridge (eq. 2)
        self.bridge        = CrossConsistencyBridge(d_lat=d_lat)
        # Fusion MLP
        self.fusion_mlp    = nn.Sequential(
            nn.Linear(d_lat * 2, d_lat),
            nn.GELU(),
            nn.Linear(d_lat, d_lat),
        )
        # Hodge decomposer (eq. 3)
        self.hodge         = HodgeDecomposer(d_lat=d_lat, d_sub=d_sub)
        # Spatial decoder (same as CNN_v2)
        self.decoder       = Decoder(c=c)
        # Moveout GNN (eq. 4)
        self.gnn           = MoveoutGNN(c=c)
        # Arrival heads (eq. 1)
        self.head_p        = ArrivalHead(c, n_ch, nt)
        self.head_ac       = ArrivalHead(c, n_ch, nt)
        # Detection head
        self.head_det      = DetectionHead(c)

    def forward(
        self,
        gather_norm: torch.Tensor,
        fk_spectrum: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        gather_norm : (B, 1, n_ch, nt)   channel-normalised gather
        fk_spectrum : (B, 1, n_ch, n_ch) log f-k spectrum

        Returns
        -------
        dict:
            't_p'       : (B, n_ch)   P-wave arrival times  [normalised 0–1]
            't_ac'      : (B, n_ch)   acoustic arrivals      [normalised 0–1]
            'det'       : (B, 1)      P-wave detection logit
            'z_A'       : (B, d_lat)  Kähler latent
            'z_B'       : (B, d_lat)  holomorphic latent
            'z_hat_B'   : (B, d_lat)  φ(z_A)    for eq. 2
            'z_hat_A'   : (B, d_lat)  φ⁻¹(z_B)  for eq. 2
            'hodge'     : tuple (z_P, z_ac, z_bg) for eq. 3
        """
        # ── Encode ────────────────────────────────────────────────────
        z_A, (s0, s1, s2, s3) = self.kahler_enc(gather_norm)
        z_B                   = self.holo_enc(fk_spectrum)

        # ── Cross-consistency (eq. 2) ─────────────────────────────────
        z_hat_B, z_hat_A = self.bridge(z_A, z_B)

        # ── Fuse ──────────────────────────────────────────────────────
        z_fused = self.fusion_mlp(torch.cat([z_A, z_B], dim=-1))

        # ── Hodge decompose (eq. 3) ───────────────────────────────────
        z_P, z_ac, z_bg = self.hodge(z_fused)

        # ── Spatial decode ────────────────────────────────────────────
        feat = self.decoder(s0, s1, s2, s3)

        # ── Moveout GNN (eq. 4) ───────────────────────────────────────
        feat = self.gnn(feat)

        return {
            't_p':     self.head_p(feat),
            't_ac':    self.head_ac(feat),
            'det':     self.head_det(feat),
            # Auxiliary outputs for loss computation
            'z_A':     z_A,
            'z_B':     z_B,
            'z_hat_B': z_hat_B,
            'z_hat_A': z_hat_A,
            'hodge':   (z_P, z_ac, z_bg),
        }

    @staticmethod
    def count_params(model: 'MIRROR') -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_mirror(
    c: int = 32, d_lat: int = 256, d_sub: int = 64,
    n_ch: int = 128, nt: int = 2048,
) -> MIRROR:
    """Build MIRROR and print parameter count."""
    model = MIRROR(c=c, d_lat=d_lat, d_sub=d_sub, n_ch=n_ch, nt=nt)
    n = MIRROR.count_params(model)
    print(f'MIRROR | c={c} d_lat={d_lat} d_sub={d_sub} | params={n:,}')
    return model


if __name__ == '__main__':
    model = build_mirror()
    gn = torch.randn(2, 1, 128, 2048)
    fk = torch.randn(2, 1, 128, 128)
    out = model(gn, fk)
    print('t_p  shape:', out['t_p'].shape)
    print('t_ac shape:', out['t_ac'].shape)
    print('z_A  shape:', out['z_A'].shape)
