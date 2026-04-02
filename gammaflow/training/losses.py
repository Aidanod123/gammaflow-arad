"""Shared loss and scoring functions for spectral reconstruction models.

Normalization summary
---------------------
These functions assume the following data flow:

1. **Input spectra** are already L1-normalized (sum to 1.0 per sample) before
   entering the model.
2. The model's encoder applies **Linf normalization** (divide by per-sample max)
   inside ``_normalize_input``, so the peak bin becomes ~1.0.
3. The decoder's final layer uses **Sigmoid**, producing values in [0, 1].

Scoring / loss then operates as follows:

* **JSD**: Both target and reconstruction are re-normalized to valid probability
  distributions via ``normalize_prob`` (L1 norm + clamp).  Because the target is
  already L1-normalized this is nearly a no-op for the target; for the sigmoid
  reconstruction it converts to a proper distribution before computing
  Jensen-Shannon divergence.

* **Chi-squared**: The sigmoid reconstruction is *denormalized* by multiplying
  by ``max(target)`` to approximately undo the Linf step, then a Pearson
  chi-squared statistic is computed against the raw target.

All functions return **per-sample** values (shape ``(batch,)``).  Training code
should call ``.mean()`` on the result to get a scalar loss for back-propagation.
"""

from __future__ import annotations

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for loss functions. Install with: pip install torch"
    ) from exc


def normalize_prob(x: torch.Tensor) -> torch.Tensor:
    """L1-normalize tensor along the last dimension, clamping negatives."""
    eps = 1e-10
    x = torch.clamp(x, min=eps)
    return x / torch.clamp(x.sum(dim=-1, keepdim=True), min=eps)


def jsd_per_sample(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Per-sample Jensen-Shannon divergence between target and reconstruction.

    Both inputs are L1-normalized before computing the divergence.

    Returns shape ``(batch,)``.
    """
    p = normalize_prob(target)
    q = normalize_prob(recon)
    m = 0.5 * (p + q)
    kld_pm = torch.sum(p * torch.log(torch.clamp(p / m, min=1e-10)), dim=-1)
    kld_qm = torch.sum(q * torch.log(torch.clamp(q / m, min=1e-10)), dim=-1)
    return torch.sqrt(0.5 * (kld_pm + kld_qm))


def chi2_per_sample(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Per-sample chi-squared score between target and reconstruction.

    The sigmoid reconstruction is denormalized by ``max(target)`` before
    computing the statistic.

    Returns shape ``(batch,)``.
    """
    eps = 1e-8
    max_vals = torch.max(target, dim=1, keepdim=True).values
    recon_denorm = recon * (max_vals + eps)
    recon_denorm = torch.clamp(recon_denorm, min=eps)
    return torch.sum((target - recon_denorm) ** 2 / recon_denorm, dim=-1)


def jsd_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Mean JSD loss for training (scalar)."""
    return jsd_per_sample(target, recon).mean()


def chi2_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Mean chi-squared loss for training (scalar)."""
    return chi2_per_sample(target, recon).mean()


def get_loss_fn(loss_name: str):
    """Return a training loss function by name.

    Supported: ``'jsd'``, ``'chi2'``, ``'mse'``.
    """
    name = str(loss_name).lower()
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "jsd":
        return jsd_loss
    if name == "chi2":
        return chi2_loss
    raise ValueError(f"Unsupported loss '{loss_name}'. Use one of: mse, jsd, chi2")


def score_batch(
    targets: torch.Tensor,
    recon: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """Per-sample anomaly scores for inference.

    Returns shape ``(batch,)``.
    """
    if loss_type == "jsd":
        return jsd_per_sample(targets, recon)
    if loss_type == "chi2":
        return chi2_per_sample(targets, recon)
    raise ValueError(f"Unsupported loss_type '{loss_type}' for scoring")
