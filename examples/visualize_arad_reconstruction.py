"""Visualize ARAD input vs reconstruction on preprocessed RADAI runs.

This script loads a trained ARAD model and a single preprocessed run file,
then saves comparison plots for selected spectra.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gammaflow import SpectralTimeSeries, Spectrum
from gammaflow.algorithms.arad import ARADDetector
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


def _spectrum_time(spec: Spectrum) -> float:
    if spec.live_time is not None and not np.isnan(spec.live_time):
        return float(spec.live_time)
    return float(spec.real_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ARAD reconstructions")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--log-y", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ts = _load_run(args.run_file)

    # Validate checkpoint format early so users get a clear actionable message.
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "n_bins" not in ckpt:
        keys = sorted(list(ckpt.keys())) if isinstance(ckpt, dict) else []
        raise ValueError(
            "This script expects an ARAD checkpoint with key 'n_bins'. "
            f"Got keys: {keys}. "
            "It looks like you passed a non-ARAD model (likely LSTM temporal). "
            "Use an ARAD model such as models/arad_chi2_no_lstm.pt."
        )

    detector = ARADDetector(device=args.device, verbose=False)
    detector.load(str(args.model_path))

    scores = detector.score_time_series(ts)
    finite_mask = np.isfinite(scores)
    finite_indices = np.where(finite_mask)[0]

    if args.indices:
        selected = [i for i in args.indices if 0 <= i < ts.n_spectra]
    else:
        if finite_indices.size == 0:
            raise RuntimeError("No finite scores found in run")
        finite_scores = scores[finite_mask]
        order = np.argsort(finite_scores)[::-1]
        k = max(1, int(args.top_k))
        selected = [int(finite_indices[j]) for j in order[:k]]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_path": str(args.model_path.resolve()),
        "run_file": str(args.run_file.resolve()),
        "n_spectra": int(ts.n_spectra),
        "selected_indices": selected,
        "examples": [],
    }

    for idx in selected:
        spec = ts[idx]
        score = float(scores[idx])

        recon_count_rate = detector.reconstruct(spec)
        t = _spectrum_time(spec)
        recon_counts = recon_count_rate * t

        recon_spec = Spectrum(
            counts=recon_counts,
            energy_edges=spec.energy_edges,
            timestamp=spec.timestamp,
            real_time=spec.real_time,
            live_time=spec.live_time,
        )

        fig, ax = plot_spectrum_comparison(
            [spec, recon_spec],
            labels=["Input", "Reconstruction"],
            mode="count_rate",
            log_y=bool(args.log_y),
        )
        ax.set_title(f"Index {idx} | Score={score:.6f}")

        out_png = args.output_dir / f"reconstruction_idx{idx}.png"
        fig.savefig(out_png, dpi=160, bbox_inches="tight")

        import matplotlib.pyplot as plt

        plt.close(fig)

        summary["examples"].append(
            {
                "index": int(idx),
                "score": score,
                "plot": str(out_png.resolve()),
            }
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved {len(selected)} reconstruction plots to {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
