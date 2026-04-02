"""Visualize LSTM temporal model input vs reconstruction on preprocessed runs.

This script loads a trained ``LSTMTemporalDetector`` checkpoint and a single
preprocessed ``run*.pt`` file, then saves side-by-side plots comparing the
window target spectrum against the model reconstruction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gammaflow import SpectralTimeSeries, Spectrum
from gammaflow.algorithms.lstm_temporal import LSTMTemporalDetector
from gammaflow.visualization import plot_spectrum_comparison


def _as_numpy_1d(value: object, length: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(length, default, dtype=np.float64)

    arr = np.asarray(value)

    if arr.dtype == object:
        out = np.empty(length, dtype=np.float64)
        for i in range(length):
            v = arr[i] if i < len(arr) else default
            out[i] = float(default if v is None else v)
        return out

    arr = arr.astype(np.float64, copy=False)
    if arr.ndim == 0:
        return np.full(length, float(arr), dtype=np.float64)

    if arr.shape != (length,):
        raise ValueError(f"Expected shape ({length},), got {arr.shape}")

    return arr


def _load_run(path: Path) -> SpectralTimeSeries:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "spectra" not in obj:
        raise ValueError(f"Expected dict with 'spectra' in {path}")

    spectra = obj["spectra"]
    if torch.is_tensor(spectra):
        counts = spectra.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        counts = np.asarray(spectra, dtype=np.float64)

    if counts.ndim != 2:
        raise ValueError(f"spectra in {path} must be 2D, got {counts.shape}")

    n_spectra = counts.shape[0]
    integration_time = obj.get("integration_time")
    rt_default = float(integration_time) if integration_time is not None else 1.0

    timestamps = _as_numpy_1d(obj.get("timestamps"), n_spectra, default=0.0)
    real_times = _as_numpy_1d(obj.get("real_times"), n_spectra, default=rt_default)
    live_times_value = obj.get("live_times")
    live_times = (
        None
        if live_times_value is None
        else _as_numpy_1d(live_times_value, n_spectra, default=rt_default)
    )

    energy_edges = obj.get("energy_edges")
    if energy_edges is not None:
        energy_edges = np.asarray(energy_edges, dtype=np.float64)

    return SpectralTimeSeries.from_array(
        counts,
        energy_edges=energy_edges,
        timestamps=timestamps,
        live_times=live_times,
        real_times=real_times,
    )


def _to_target_scale(
    detector: LSTMTemporalDetector,
    target: np.ndarray,
    recon_raw: np.ndarray,
) -> np.ndarray:
    eps = 1e-8
    if detector.loss_type == "chi2":
        max_val = float(np.max(target))
        return recon_raw * (max_val + eps)

    # JSD is shape-based; scale reconstruction to target total for plotting.
    recon_sum = float(np.sum(recon_raw))
    target_sum = float(np.sum(target))
    if recon_sum <= eps:
        return np.zeros_like(recon_raw)
    return (recon_raw / recon_sum) * target_sum


def _score_and_reconstruct_at_index(
    detector: LSTMTemporalDetector,
    counts: np.ndarray,
    idx: int,
    latent_mask_pct: float,
    mask_seed: Optional[int],
) -> Optional[tuple[float, np.ndarray, np.ndarray]]:
    window = detector._build_window_for_index(counts.astype(np.float32, copy=False), idx)
    if window is None:
        return None

    window_indices = detector._window_indices_for_end(int(idx))
    rng = np.random.default_rng(mask_seed) if latent_mask_pct > 0 else None
    latent_mask_np = detector._build_latent_mask_for_window(
        window_indices=window_indices,
        masked_index_set=None,
        latent_mask_pct=float(latent_mask_pct),
        rng=rng,
        mask_target_timestep=bool(detector.use_attention),
    )

    with torch.no_grad():
        window_tensor = torch.from_numpy(window).unsqueeze(0).to(detector.device)
        target_tensor = torch.from_numpy(counts[idx].astype(np.float32)).unsqueeze(0).to(detector.device)
        latent_mask_tensor = None
        if latent_mask_np is not None:
            latent_mask_tensor = torch.from_numpy(latent_mask_np).unsqueeze(0).to(detector.device)

        recon_raw = detector.model_(window_tensor, latent_timestep_mask=latent_mask_tensor)
        score_t = detector._score_batch(target_tensor, recon_raw)

    recon_raw_np = recon_raw.detach().cpu().numpy().squeeze(0)
    target_np = counts[idx].astype(np.float32, copy=False)
    recon_scaled = _to_target_scale(detector, target_np, recon_raw_np)

    return float(score_t.item()), target_np, recon_scaled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LSTM temporal reconstructions")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--latent-mask-pct", type=float, default=0.0)
    parser.add_argument("--latent-mask-seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ts = _load_run(args.run_file)
    counts = np.asarray(ts.counts, dtype=np.float32)

    detector = LSTMTemporalDetector(device=args.device)

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "seq_len" not in ckpt:
        keys = sorted(list(ckpt.keys())) if isinstance(ckpt, dict) else []
        raise ValueError(
            "This script expects an LSTM temporal checkpoint with key 'seq_len'. "
            f"Got keys: {keys}."
        )

    detector.load(str(args.model_path))

    if args.indices:
        selected = [int(i) for i in args.indices if 0 <= int(i) < ts.n_spectra]
    else:
        scores = detector.score_time_series(
            ts,
            latent_mask_pct=float(args.latent_mask_pct),
            mask_seed=args.latent_mask_seed,
        )
        valid_idx = np.where(np.isfinite(scores))[0]
        if valid_idx.size == 0:
            raise RuntimeError("No finite scores available for this run/model")
        valid_scores = scores[valid_idx]
        order = np.argsort(valid_scores)[::-1]
        k = max(1, int(args.top_k))
        selected = [int(valid_idx[j]) for j in order[:k]]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_path": str(args.model_path.resolve()),
        "run_file": str(args.run_file.resolve()),
        "seq_len": int(detector.seq_len),
        "seq_stride": int(detector.seq_stride),
        "loss_type": str(detector.loss_type),
        "use_attention": bool(detector.use_attention),
        "selected_indices": selected,
        "examples": [],
    }

    for idx in selected:
        out = _score_and_reconstruct_at_index(
            detector=detector,
            counts=counts,
            idx=int(idx),
            latent_mask_pct=float(args.latent_mask_pct),
            mask_seed=args.latent_mask_seed,
        )
        if out is None:
            continue

        score, target_arr, recon_arr = out

        spec = ts[int(idx)]
        target_spec = Spectrum(
            counts=target_arr,
            energy_edges=spec.energy_edges,
            timestamp=spec.timestamp,
            real_time=spec.real_time,
            live_time=spec.live_time,
        )
        recon_spec = Spectrum(
            counts=recon_arr,
            energy_edges=spec.energy_edges,
            timestamp=spec.timestamp,
            real_time=spec.real_time,
            live_time=spec.live_time,
        )

        fig, ax = plot_spectrum_comparison(
            [target_spec, recon_spec],
            labels=["Input (target)", "Reconstruction"],
            mode="count_rate",
            log_y=bool(args.log_y),
        )
        ax.set_title(f"Index {idx} | Score={score:.6f}")

        out_png = args.output_dir / f"lstm_reconstruction_idx{idx}.png"
        fig.savefig(out_png, dpi=160, bbox_inches="tight")

        import matplotlib.pyplot as plt

        plt.close(fig)

        summary["examples"].append(
            {
                "index": int(idx),
                "score": float(score),
                "plot": str(out_png.resolve()),
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved {len(summary['examples'])} reconstruction plots to {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
