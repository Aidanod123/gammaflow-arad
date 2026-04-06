"""Shared loss and scoring functions for spectral reconstruction models.

JSD is computed on probability-normalized spectra, while chi-squared can be
computed in a physical domain by de-normalizing both target and reconstruction
using per-spectrum scale factors (for example, gross ``count_rates``).
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


def _resolve_target_scales(
    target: torch.Tensor,
    target_scales: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Return shape ``(batch, 1)`` scale factors for chi2 de-normalization."""
    if target_scales is None:
        return torch.clamp(target.sum(dim=-1, keepdim=True), min=eps)

    scales = target_scales
    if scales.dim() == 1:
        scales = scales.unsqueeze(-1)
    if scales.dim() != 2 or scales.shape[1] != 1:
        raise ValueError(
            "target_scales must have shape (batch,) or (batch, 1), "
            f"got {tuple(target_scales.shape)}"
        )
    if scales.shape[0] != target.shape[0]:
        raise ValueError(
            "target_scales batch dimension must match target batch dimension, "
            f"got {scales.shape[0]} and {target.shape[0]}"
        )
    return torch.clamp(scales.to(device=target.device, dtype=target.dtype), min=eps)


def chi2_per_sample(
    target: torch.Tensor,
    recon: torch.Tensor,
    target_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample chi-squared score between target and reconstruction.

    Both target and reconstruction are first L1-normalized to probability
    vectors, then de-normalized with per-sample scales before computing the
    Pearson chi-squared statistic.

    Returns shape ``(batch,)``.
    """
    eps = 1e-3
    scale = _resolve_target_scales(target, target_scales, eps)
    target_denorm = normalize_prob(target) * scale
    recon_denorm = normalize_prob(recon) * scale
    recon_denorm = torch.clamp(recon_denorm, min=eps)
    return torch.sum((target_denorm - recon_denorm) ** 2 / recon_denorm, dim=-1)


def jsd_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Mean JSD loss for training (scalar)."""
    return jsd_per_sample(target, recon).mean()


def chi2_loss(
    target: torch.Tensor,
    recon: torch.Tensor,
    target_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean chi-squared loss for training (scalar)."""
    return chi2_per_sample(target, recon, target_scales=target_scales).mean()


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
    target_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample anomaly scores for inference.

    Returns shape ``(batch,)``.
    """
    if loss_type == "jsd":
        return jsd_per_sample(targets, recon)
    if loss_type == "chi2":
        return chi2_per_sample(targets, recon, target_scales=target_scales)
    raise ValueError(f"Unsupported loss_type '{loss_type}' for scoring")
