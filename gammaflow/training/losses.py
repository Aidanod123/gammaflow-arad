"""Shared loss and scoring functions for spectral reconstruction models.

JSD is computed on probability-normalized spectra, while chi-squared can be
computed in a physical domain by de-normalizing both target and reconstruction
using per-spectrum scale factors (for example, gross ``count_rates``).

Poisson NLL is the statistically correct loss for count data.  The N-normalized
form reduces to KL(p || q) on L1-normalized inputs, which is rate-invariant:
the N cancels exactly, leaving only the shape divergence between target and
reconstruction.
"""

from __future__ import annotations

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for loss functions. Install with: pip install torch"
    ) from exc

# ---------------------------------------------------------------------------
# Debug counter — prints every DEBUG_EVERY calls, set to 0 to disable
# ---------------------------------------------------------------------------
_DEBUG_EVERY = 0
_debug_call_count = 0


def _debug_print(label: str, t: torch.Tensor, extra: str = "") -> None:
    print(
        f"  [{label}] shape={tuple(t.shape)} "
        f"min={t.min().item():.5f} max={t.max().item():.5f} "
        f"mean={t.mean().item():.5f} sum_last={t.sum(-1).mean().item():.5f}"
        + (f"  {extra}" if extra else "")
    )


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
    global _debug_call_count
    _debug_call_count += 1
    do_debug = (_DEBUG_EVERY > 0) and (_debug_call_count % _DEBUG_EVERY == 0)

    p = normalize_prob(target)
    q = normalize_prob(recon)
    m = 0.5 * (p + q)
    kld_pm = torch.sum(p * torch.log(torch.clamp(p / m, min=1e-10)), dim=-1)
    kld_qm = torch.sum(q * torch.log(torch.clamp(q / m, min=1e-10)), dim=-1)
    result = torch.sqrt(0.5 * (kld_pm + kld_qm))

    if do_debug:
        target_sums = target.sum(dim=-1)
        recon_sums = recon.sum(dim=-1)
        print(f"\n--- JSD debug (call #{_debug_call_count}) ---")
        print(f"  target: shape={tuple(target.shape)}  sum_per_row min={target_sums.min():.4f} max={target_sums.max():.4f} mean={target_sums.mean():.4f}  <- should be ~1.0 if L1-normalized")
        print(f"  recon:  shape={tuple(recon.shape)}   sum_per_row min={recon_sums.min():.4f} max={recon_sums.max():.4f} mean={recon_sums.mean():.4f}  <- sigmoid: ~n_bins/2, softmax: ~1.0")
        print(f"  recon raw values: min={recon.min():.5f} max={recon.max():.5f} mean={recon.mean():.5f}")
        print(f"  p (normalized target): sum_check min={p.sum(-1).min():.6f} max={p.sum(-1).max():.6f}")
        print(f"  q (normalized recon):  sum_check min={q.sum(-1).min():.6f} max={q.sum(-1).max():.6f}")
        print(f"  JSD scores: min={result.min():.5f} max={result.max():.5f} mean={result.mean():.5f} std={result.std():.5f}")
        if result.isnan().any() or result.isinf().any():
            print(f"  *** NaN={result.isnan().sum().item()} Inf={result.isinf().sum().item()} ***")

    return result


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
    vectors (were sigmoid activated), then de-normalized with per-sample scales before computing the
    Pearson chi-squared statistic.

    Returns shape ``(batch,)``.
    """
    global _debug_call_count
    _debug_call_count += 1
    do_debug = (_DEBUG_EVERY > 0) and (_debug_call_count % _DEBUG_EVERY == 0)

    eps = 1e-3
    scale = _resolve_target_scales(target, target_scales, eps)
    target_denorm = normalize_prob(target) * scale
    recon_denorm = normalize_prob(recon) * scale
    recon_denorm = torch.clamp(recon_denorm, min=eps)
    result = torch.sum((target_denorm - recon_denorm) ** 2 / recon_denorm, dim=-1)

    if do_debug:
        target_sums = target.sum(dim=-1)
        recon_sums = recon.sum(dim=-1)
        scale_sq = scale.squeeze(-1)
        print(f"\n--- Chi2 debug (call #{_debug_call_count}) ---")
        print(f"  target: shape={tuple(target.shape)}  sum_per_row min={target_sums.min():.4f} max={target_sums.max():.4f} mean={target_sums.mean():.4f}  <- should be ~1.0 if L1-normalized")
        print(f"  recon:  shape={tuple(recon.shape)}   sum_per_row min={recon_sums.min():.4f} max={recon_sums.max():.4f} mean={recon_sums.mean():.4f}")
        print(f"  recon raw values: min={recon.min():.5f} max={recon.max():.5f} mean={recon.mean():.5f}")
        print(f"  target_scales (total counts): min={scale_sq.min():.1f} max={scale_sq.max():.1f} mean={scale_sq.mean():.1f}  <- should be gross count total per spectrum")
        print(f"  target_denorm sum_per_row: min={target_denorm.sum(-1).min():.2f} max={target_denorm.sum(-1).max():.2f}  <- should match target_scales")
        print(f"  recon_denorm  sum_per_row: min={recon_denorm.sum(-1).min():.2f} max={recon_denorm.sum(-1).max():.2f}  <- should match target_scales")
        print(f"  chi2 scores: min={result.min():.3f} max={result.max():.3f} mean={result.mean():.3f} std={result.std():.3f}")
        if result.isnan().any() or result.isinf().any():
            print(f"  *** NaN={result.isnan().sum().item()} Inf={result.isinf().sum().item()} ***")

    return result


def poisson_nll_per_sample(
    target: torch.Tensor,
    recon: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Per-sample N-normalized Poisson NLL between target and reconstruction.

    Derivation
    ----------
    Poisson NLL = Σ_i [μ_i - k_i * log(μ_i)] where k_i are observed counts
    and μ_i are expected counts.  Setting μ_i = q_i * N (recon probability ×
    total counts) and k_i = p_i * N (target probability × total counts), then
    dividing by N and subtracting the constant minimum (perfect reconstruction),
    the N cancels exactly and the excess NLL reduces to:

        KL(p || q) = Σ_i p_i * log(p_i / q_i)

    This is fully rate-invariant — it only measures shape divergence between
    target and reconstruction.  It equals zero at perfect reconstruction and
    is more sensitive to rare high-energy lines than JSD (unbounded above;
    diverges if q_i → 0 where p_i > 0).

    Both inputs are L1-normalized internally before computing the divergence.

    Returns shape ``(batch,)``.
    """
    p = normalize_prob(target)  # L1-normalized target
    q = normalize_prob(recon)   # L1-normalized reconstruction
    q = torch.clamp(q, min=eps)
    # KL(p || q) = Σ p_i * log(p_i / q_i)
    # Terms where p_i = 0 contribute 0 (0 * log(...) = 0), handled by clamp.
    result = torch.sum(p * torch.log(torch.clamp(p / q, min=eps)), dim=-1)
    return result


def poisson_nll_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Mean Poisson NLL loss for training (scalar)."""
    return poisson_nll_per_sample(target, recon).mean()


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

    Supported: ``'jsd'``, ``'chi2'``, ``'poisson'``, ``'mse'``.
    """
    name = str(loss_name).lower()
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "jsd":
        return jsd_loss
    if name == "chi2":
        return chi2_loss
    if name == "poisson":
        return poisson_nll_loss
    raise ValueError(f"Unsupported loss '{loss_name}'. Use one of: mse, jsd, chi2, poisson")


def score_batch(
    targets: torch.Tensor,
    recon: torch.Tensor,
    loss_type: str,
    target_scales: torch.Tensor | None = None,
    score_type: str | None = None,
) -> torch.Tensor:
    """Per-sample anomaly scores for inference.

    Parameters
    ----------
    score_type : str or None
        Override the scoring metric independently of the training loss.
        When ``None`` (default), uses ``loss_type``.  Supported values:
        ``"jsd"`` — Jensen-Shannon divergence (rate-invariant, [0,1]);
        ``"corrected_jsd"`` — JSD minus the analytical Poisson noise floor
        ``sqrt(n_bins / (8 * total_counts))``, removing count-rate dependence;
        ``"chi2"`` — Pearson chi-squared on count-domain data (rate-dependent);
        ``"normalized_chi2"`` — chi2 divided by total counts, isolating shape
        error independent of count rate;
        ``"reduced_chi2"`` — ``(chi2 - n_bins) / n_bins``, removes the
        expected noise offset so a perfect reconstruction scores ~0;
        ``"combined"`` — ``JSD + reduced_chi2``, captures both flat shape
        deviation (JSD) and count-weighted bin deviation (chi2), both ~0
        for background;
        ``"poisson"`` — N-normalized Poisson NLL = KL(target || recon),
        rate-invariant, background ~0, unbounded above.

    Returns shape ``(batch,)``.
    """
    metric = score_type if score_type is not None else loss_type
    if metric == "jsd":
        return jsd_per_sample(targets, recon)
    if metric == "corrected_jsd":
        eps = 1e-3
        raw = jsd_per_sample(targets, recon)
        n_bins = targets.shape[-1]
        scale = _resolve_target_scales(targets, target_scales, eps).squeeze(-1)
        noise_floor = torch.sqrt(torch.tensor(n_bins / 8.0, device=raw.device) / scale)
        return raw - noise_floor
    if metric == "chi2":
        return chi2_per_sample(targets, recon, target_scales=target_scales)
    if metric == "normalized_chi2":
        eps = 1e-3
        raw = chi2_per_sample(targets, recon, target_scales=target_scales)
        scale = _resolve_target_scales(targets, target_scales, eps).squeeze(-1)
        return raw / scale
    if metric == "reduced_chi2":
        n_bins = targets.shape[-1]
        raw = chi2_per_sample(targets, recon, target_scales=target_scales)
        return (raw - n_bins) / n_bins
    if metric == "combined":
        # JSD: rate-invariant, flat across bins, background ~0.06-0.09
        # reduced_chi2: (chi2 - n_bins) / n_bins, background ~0, weights
        #   high-count bins more — complementary to JSD's flat weighting
        n_bins = targets.shape[-1]
        jsd = jsd_per_sample(targets, recon)
        chi2_raw = chi2_per_sample(targets, recon, target_scales=target_scales)
        reduced = (chi2_raw - n_bins) / n_bins
        return jsd + reduced
    if metric == "poisson":
        return poisson_nll_per_sample(targets, recon)
    raise ValueError(f"Unsupported score_type '{metric}' for scoring")
