"""Train standard ARAD (no LSTM) from preprocessed RADAI run files.

This script trains ``gammaflow.algorithms.arad.ARADDetector`` directly on
``run*.pt`` files in a preprocessed directory.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gammaflow.algorithms.arad import ARADDetector
from gammaflow.core.time_series import SpectralTimeSeries


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


def _load_preprocessed_dir(preprocessed_dir: Path, max_runs: Optional[int]) -> SpectralTimeSeries:
    run_files = sorted(preprocessed_dir.glob("run*.pt"))
    if not run_files:
        raise FileNotFoundError(f"No run*.pt files found in {preprocessed_dir}")

    if max_runs is not None:
        run_files = run_files[: max_runs]

    counts_parts = []
    timestamps_parts = []
    real_times_parts = []
    live_times_parts = []
    has_any_live = False
    shared_energy_edges = None
    t_offset = 0.0

    for idx, path in enumerate(run_files):
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
            if shared_energy_edges is None:
                shared_energy_edges = energy_edges
            elif not np.array_equal(shared_energy_edges, energy_edges):
                raise ValueError(
                    f"Energy edges mismatch between runs; first mismatch at {path}"
                )

        counts_parts.append(counts)
        timestamps_parts.append(timestamps + t_offset)
        real_times_parts.append(real_times)

        if live_times is not None:
            has_any_live = True
            live_times_parts.append(live_times)

        if idx < (len(run_files) - 1):
            # Keep strictly increasing timestamps across concatenated runs.
            t_offset = float(timestamps_parts[-1][-1]) + 1e-6

    all_counts = np.vstack(counts_parts)
    all_timestamps = np.concatenate(timestamps_parts)
    all_real_times = np.concatenate(real_times_parts)

    all_live_times = None
    if has_any_live and live_times_parts:
        all_live_times = np.concatenate(live_times_parts)

    return SpectralTimeSeries.from_array(
        all_counts,
        energy_edges=shared_energy_edges,
        timestamps=all_timestamps,
        live_times=all_live_times,
        real_times=all_real_times,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standard ARAD detector (no LSTM) from preprocessed run files"
    )
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)

    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--l1-lambda", type=float, default=1e-3)
    parser.add_argument("--l2-lambda", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--loss-type", type=str, default="chi2", choices=["jsd", "chi2"])
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ts = _load_preprocessed_dir(args.preprocessed_dir, args.max_runs)

    detector = ARADDetector(
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l1_lambda=args.l1_lambda,
        l2_lambda=args.l2_lambda,
        early_stopping_patience=args.early_stopping_patience,
        validation_split=args.validation_split,
        device=args.device,
        loss_type=args.loss_type,
        verbose=not args.quiet,
    )

    detector.fit(ts)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(args.output_model))

    print(f"Saved ARAD model: {args.output_model}")
    print(f"Trained on spectra: {ts.n_spectra}, bins: {ts.n_bins}")


if __name__ == "__main__":
    main()
