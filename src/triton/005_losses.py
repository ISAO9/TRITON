"""
005_losses.py
-------------
TRITON loss functions.

Corresponds to: "Loss Functions" in Methods.

Equations implemented:
  (5)  ℒ_reg  = Σ_ch Hδ(t̂_p − t_p) + Σ_ch Hδ(t̂_ac − t_ac)   [Huber, δ=0.01]
  (6)  ℒ_ord  = (1/N) Σ_ch ReLU(t̂_p − t̂_ac · T)              [ordering]
  (7)  ℒ_mo   = H_2(Δt̂_p − Δt_p) / N                           [moveout, δ=2ms]
  (8)  ℒ_total= ℒ_reg + 0.5·ℒ_ord + 0.3·ℒ_mo
                + α·ℒ_cross + β·ℒ_Hodge    [MIRROR only; α=0.1, β=0.05]

Usage:
    from triton.losses import TRITONLoss
    loss_fn = TRITONLoss(model_type='cnn_v2')
    total, breakdown = loss_fn(predictions, batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ── Constants ────────────────────────────────────────────────────────────────
T_MAX       = 15.0    # s  gather duration (normalisation factor)
DELTA_REG   = 0.01    # Huber δ for regression loss  (eq. 5)
DELTA_MO    = 2.0     # Huber δ for moveout loss [ms] (eq. 7)
W_ORD       = 0.5     # weight of ordering constraint (eq. 6)
W_MO        = 0.3     # weight of moveout loss         (eq. 7)
W_CROSS     = 0.1     # α: cross-consistency weight    (eq. 2, MIRROR)
W_HODGE     = 0.05    # β: Hodge orthogonality weight  (eq. 3, MIRROR)


def regression_loss(
    t_p_hat:  torch.Tensor,
    t_ac_hat: torch.Tensor,
    t_p_gt:   torch.Tensor,
    t_ac_gt:  torch.Tensor,
    delta:    float = DELTA_REG,
) -> torch.Tensor:
    """
    Equation (5): ℒ_reg = Σ_ch Hδ(t̂_p − t_p) + Σ_ch Hδ(t̂_ac − t_ac)

    All times are normalised to [0, 1].
    """
    return (
        F.huber_loss(t_p_hat,  t_p_gt,  delta=delta, reduction='mean') +
        F.huber_loss(t_ac_hat, t_ac_gt, delta=delta, reduction='mean')
    )


def ordering_loss(
    t_p_hat:  torch.Tensor,
    t_ac_hat: torch.Tensor,
    t_max:    float = T_MAX,
) -> torch.Tensor:
    """
    Equation (6): ℒ_ord = (1/N) Σ_ch ReLU(t̂_p − t̂_ac · T)

    Penalises predictions where P-wave arrives after acoustic.
    Note: inputs are normalised times; multiply by T_MAX for ms comparison.
    """
    return F.relu(t_p_hat - t_ac_hat).mean()


def moveout_loss(
    t_p_hat:  torch.Tensor,
    t_p_gt:   torch.Tensor,
    t_max:    float = T_MAX,
    delta_ms: float = DELTA_MO,
) -> torch.Tensor:
    """
    Equation (7): ℒ_mo = H_2(Δt̂_p − Δt_p) / N

    Inter-channel differential moveout in milliseconds.
    δ = 2 ms (DELTA_MO).
    """
    # Convert normalised → ms
    tp_ms  = t_p_hat * t_max * 1000.0
    tpg_ms = t_p_gt  * t_max * 1000.0
    # Differential moveout: Δt(i) = t(i+1) − t(i)
    dp  = tp_ms[:,  1:] - tp_ms[:,  :-1]
    dg  = tpg_ms[:, 1:] - tpg_ms[:, :-1]
    return F.huber_loss(dp, dg, delta=delta_ms, reduction='mean') / 1e4


def cross_consistency_loss(
    z_A: torch.Tensor, z_B: torch.Tensor,
    z_hat_B: torch.Tensor, z_hat_A: torch.Tensor,
) -> torch.Tensor:
    """
    Equation (2): ℒ_cross = ‖z_A − φ⁻¹(z_B)‖² + ‖z_B − φ(z_A)‖²
    """
    return (
        F.mse_loss(z_hat_A, z_A.detach()) +
        F.mse_loss(z_hat_B, z_B.detach())
    )


def hodge_orthogonality_loss(
    z_P: torch.Tensor,
    z_ac: torch.Tensor,
    z_bg: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Equation (3): ℒ_Hodge = Σ_{i≠j} |z_i · z_j| / (‖z_i‖‖z_j‖ + ε)
    """
    pairs = [(z_P, z_ac), (z_P, z_bg), (z_ac, z_bg)]
    loss = torch.tensor(0.0, device=z_P.device)
    for a, b in pairs:
        dot = (a * b).sum(dim=-1).abs()
        norm_a = a.norm(dim=-1)
        norm_b = b.norm(dim=-1)
        loss = loss + (dot / (norm_a * norm_b + eps)).mean()
    return loss


# ── Combined loss class ───────────────────────────────────────────────────────

class TRITONLoss(nn.Module):
    """
    Combined TRITON training loss.

    Equation (8):
        ℒ_total = ℒ_reg + w_ord·ℒ_ord + w_mo·ℒ_mo
                 + α·ℒ_cross + β·ℒ_Hodge   (MIRROR only)

    Parameters
    ----------
    model_type : str
        'cnn_v2' or 'mirror'.
    w_ord : float
        Weight of ordering constraint (default 0.5).
    w_mo : float
        Weight of moveout loss (default 0.3).
    alpha : float
        Weight of cross-consistency loss for MIRROR (default 0.1).
    beta : float
        Weight of Hodge orthogonality loss for MIRROR (default 0.05).
    """
    def __init__(
        self,
        model_type: str  = 'cnn_v2',
        w_ord:      float = W_ORD,
        w_mo:       float = W_MO,
        alpha:      float = W_CROSS,
        beta:       float = W_HODGE,
    ):
        super().__init__()
        assert model_type in ('cnn_v2', 'mirror'), \
            f"model_type must be 'cnn_v2' or 'mirror', got {model_type}"
        self.model_type = model_type
        self.w_ord  = w_ord
        self.w_mo   = w_mo
        self.alpha  = alpha
        self.beta   = beta

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        batch:       Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and per-component breakdown.

        Parameters
        ----------
        predictions : dict
            Output of CNNv2.forward() or MIRROR.forward().
        batch : dict
            Batch from DataLoader; must contain 't_p_norm', 't_ac_norm'.

        Returns
        -------
        total : torch.Tensor   scalar loss
        breakdown : dict       component values for logging
        """
        t_p_hat  = predictions['t_p']
        t_ac_hat = predictions['t_ac']
        t_p_gt   = batch['t_p_norm']
        t_ac_gt  = batch['t_ac_norm']

        # Equation (5): regression
        l_reg = regression_loss(t_p_hat, t_ac_hat, t_p_gt, t_ac_gt)

        # Equation (6): ordering
        l_ord = ordering_loss(t_p_hat, t_ac_hat)

        # Equation (7): moveout
        l_mo  = moveout_loss(t_p_hat, t_p_gt)

        # Equation (8): assemble
        total = l_reg + self.w_ord * l_ord + self.w_mo * l_mo

        breakdown = {
            'l_reg': l_reg.item(),
            'l_ord': l_ord.item(),
            'l_mo':  l_mo.item(),
        }

        # MIRROR-specific terms (eqs. 2, 3)
        if self.model_type == 'mirror':
            l_cross = cross_consistency_loss(
                predictions['z_A'],    predictions['z_B'],
                predictions['z_hat_B'], predictions['z_hat_A'],
            )
            z_P, z_ac, z_bg = predictions['hodge']
            l_hodge = hodge_orthogonality_loss(z_P, z_ac, z_bg)

            total = total + self.alpha * l_cross + self.beta * l_hodge
            breakdown['l_cross'] = l_cross.item()
            breakdown['l_hodge'] = l_hodge.item()

        return total, breakdown


# ── Convenience constructors ─────────────────────────────────────────────────

def build_loss(model_type: str = 'cnn_v2') -> TRITONLoss:
    """Build loss function for the given model type."""
    loss_fn = TRITONLoss(model_type=model_type)
    print(f'Loss function: {model_type}  '
          f'w_ord={loss_fn.w_ord}  w_mo={loss_fn.w_mo}')
    if model_type == 'mirror':
        print(f'  MIRROR terms: alpha={loss_fn.alpha}  beta={loss_fn.beta}')
    return loss_fn


if __name__ == '__main__':
    # Smoke test
    B, N_CH = 4, 128
    pred_cnn = {
        't_p':  torch.rand(B, N_CH),
        't_ac': torch.rand(B, N_CH) - 0.3,
    }
    batch = {
        't_p_norm':  torch.rand(B, N_CH),
        't_ac_norm': torch.rand(B, N_CH),
    }
    loss_fn = build_loss('cnn_v2')
    total, bd = loss_fn(pred_cnn, batch)
    print('CNN_v2 total loss:', total.item(), bd)

    # MIRROR
    pred_mirror = {
        't_p':     torch.rand(B, N_CH),
        't_ac':    torch.rand(B, N_CH) - 0.3,
        'z_A':     torch.randn(B, 256),
        'z_B':     torch.randn(B, 256),
        'z_hat_B': torch.randn(B, 256),
        'z_hat_A': torch.randn(B, 256),
        'hodge':   (torch.randn(B, 64),
                    torch.randn(B, 64),
                    torch.randn(B, 64)),
    }
    loss_fn_m = build_loss('mirror')
    total_m, bd_m = loss_fn_m(pred_mirror, batch)
    print('MIRROR  total loss:', total_m.item(), bd_m)
