#!/usr/bin/env python3
"""Plot count-rate-normalized anomaly scores.

Corrects for the systematic correlation between JSD/chi2 score and count rate
caused by Poisson noise in L1-normalized spectra.  A baseline curve
E[score | count_rate] is learned from background-only runs, then subtracted
at inference so the anomaly score is centred at zero across all count rates.

Workflow
--------
1. Score background runs (--baseline-dir) to build E[score | count_rate].
2. Score evaluation runs (--eval-dir).
3. anomaly_score = raw_score - baseline(count_rate)
4. One plot per run: corrected score (top) + gross count rate (bottom).

The baseline is saved to JSON automatically.  Pass --load-baseline to skip
background scoring and reuse a previously built baseline.

Example
-------
# Build baseline + plot
python examples/plot_rate_normalized_scores.py \\
    --model-type lstm \\
    --model-path models/4-7-lstm_jsd_masktarget_on_2.0-1.0.pt \\
    --baseline-dir preprocessed-data/no-sources-2.0-1.0 \\
    --eval-dir preprocessed-data/with-sources-2.0-1.0 \\
    --h5-path RADAI/training_v4.3.h5 \\
    --output-dir eval-results/rate-normalized-jsd \\
    --num-runs 6 --device cuda

# Reuse a saved baseline
python examples/plot_rate_normalized_scores.py \\
    --model-type lstm \\
    --model-path models/4-7-lstm_jsd_masktarget_on_2.0-1.0.pt \\
    --load-baseline eval-results/rate-normalized-jsd/baseline.json \\
    --eval-dir preprocessed-data/with-sources-2.0-1.0 \\
    --h5-path RADAI/training_v4.3.h5 \\
    --output-dir eval-results/rate-normalized-jsd \\
    --num-runs 6 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import h5py
except ImportError as exc:
    raise ImportError("h5py is required. Install with: pip install h5py") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gammaflow import SpectralTimeSeries
from gammaflow.algorithms import LSTMTemporalDetector
from gammaflow.algorithms.arad import ARADDetector


# ---------------------------------------------------------------------------
# Shared helpers (mirrored from plot_high_snr_error_and_rate.py)
# ---------------------------------------------------------------------------

@dataclass
class RunSNR:
    run_id: int
    run_name: str
    source_events: int
    total_events: int
    source_fraction: float
    peak_to_median_rate: float
    std_to_mean_rate: float


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


def _load_run(path: Path) -> Tuple[Dict[str, Any], SpectralTimeSeries]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "spectra" not in obj:
        raise ValueError(f"Expected dict with 'spectra' in {path}")
    spectra = obj["spectra"]
    if torch.is_tensor(spectra):
        counts = spectra.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        counts = np.asarray(spectra, dtype=np.float64)
    n_spectra = counts.shape[0]
    integration_time = obj.get("integration_time")
    rt_default = float(integration_time) if integration_time is not None else 1.0
    timestamps = _as_numpy_1d(obj.get("timestamps"), n_spectra, default=0.0)
    real_times = _as_numpy_1d(obj.get("real_times"), n_spectra, default=rt_default)
    live_times_val = obj.get("live_times")
    live_times = (
        None if live_times_val is None
        else _as_numpy_1d(live_times_val, n_spectra, default=rt_default)
    )
    energy_edges = obj.get("energy_edges")
    if energy_edges is not None:
        energy_edges = np.asarray(energy_edges, dtype=np.float64)
    ts = SpectralTimeSeries.from_array(
        counts, energy_edges=energy_edges,
        timestamps=timestamps, live_times=live_times, real_times=real_times,
    )
    return obj, ts


def _run_id_from_name(path: Path) -> Optional[int]:
    stem = path.stem
    if not stem.startswith("run"):
        return None
    try:
        return int(stem.replace("run", ""))
    except ValueError:
        return None


def _discover_runs(directory: Path) -> List[Path]:
    run_files = sorted(
        directory.glob("run*.pt"),
        key=lambda p: int(p.stem.replace("run", "")),
    )
    if not run_files:
        raise FileNotFoundError(f"No run*.pt files found in {directory}")
    return run_files


def _time_unit_scale(units: str) -> float:
    u = units.strip().lower()
    if u == "us":
        return 1e-6
    if u == "ms":
        return 1e-3
    if u == "s":
        return 1.0
    raise ValueError(f"Unsupported time units '{units}'")


def _compute_snr_rows(h5_path: Path, run_ids: Sequence[int]) -> List[RunSNR]:
    rows: List[RunSNR] = []
    with h5py.File(h5_path, "r") as f:
        runs = f["runs"]
        for run_id in run_ids:
            run_key = f"run{run_id}"
            if run_key not in runs:
                continue
            run_group = runs[run_key]
            listmode = run_group["listmode"]
            dt = np.asarray(listmode["dt"])
            total_events = int(dt.shape[0])
            source_events = 0
            if "id" in listmode:
                event_ids = np.asarray(listmode["id"])
                source_group = run_group.get("sources") or run_group.get("source")
                if source_group is not None and "id" in source_group:
                    source_ids = np.asarray(source_group["id"])
                    if source_ids.size > 0:
                        source_events = int(np.isin(event_ids, source_ids).sum())
            scale = _time_unit_scale("us")
            if dt.size > 0:
                abs_times = np.cumsum(np.asarray(dt, dtype=np.float64) * scale)
                n_windows = min(512, max(64, int(np.sqrt(abs_times.size))))
                centers = np.linspace(abs_times[0], abs_times[-1], num=n_windows)
                width = max((abs_times[-1] - abs_times[0]) / max(n_windows, 1), 1e-6)
                half = width / 2.0
                cnts = np.empty(n_windows, dtype=np.float64)
                for i, c in enumerate(centers):
                    left = np.searchsorted(abs_times, c - half, side="left")
                    right = np.searchsorted(abs_times, c + half, side="left")
                    cnts[i] = right - left
                rates = cnts / width
                median_rate = float(np.median(rates)) if rates.size else 0.0
                p95_rate = float(np.percentile(rates, 95.0)) if rates.size else 0.0
                mean_rate = float(np.mean(rates)) if rates.size else 0.0
                std_rate = float(np.std(rates)) if rates.size else 0.0
                peak_to_median = p95_rate / median_rate if median_rate > 0 else 0.0
                std_to_mean = std_rate / mean_rate if mean_rate > 0 else 0.0
            else:
                peak_to_median = std_to_mean = 0.0
            frac = source_events / total_events if total_events > 0 else 0.0
            rows.append(RunSNR(
                run_id=int(run_id), run_name=f"run{run_id}.pt",
                source_events=source_events, total_events=total_events,
                source_fraction=float(frac),
                peak_to_median_rate=float(peak_to_median),
                std_to_mean_rate=float(std_to_mean),
            ))
    return rows


def _gross_count_rate_from_h5(
    h5_path: Path, run_id: int, timestamps: np.ndarray,
    integration_time: float, time_units: str,
) -> np.ndarray:
    run_key = f"run{run_id}"
    with h5py.File(h5_path, "r") as f:
        run_group = f["runs"][run_key]
        dt = np.asarray(run_group["listmode"]["dt"], dtype=np.float64)
    if dt.size == 0:
        return np.zeros_like(timestamps, dtype=np.float64)
    abs_times = np.cumsum(dt * _time_unit_scale(time_units))
    half = integration_time / 2.0
    rates = np.zeros_like(timestamps, dtype=np.float64)
    for i, center in enumerate(timestamps):
        left = np.searchsorted(abs_times, center - half, side="left")
        right = np.searchsorted(abs_times, center + half, side="left")
        rates[i] = (right - left) / integration_time if integration_time > 0 else 0.0
    return rates


def _source_times_from_h5(h5_path: Path, run_id: int) -> np.ndarray:
    run_key = f"run{run_id}"
    with h5py.File(h5_path, "r") as f:
        runs = f["runs"]
        if run_key not in runs:
            return np.array([], dtype=np.float64)
        run_group = runs[run_key]
        source_group = run_group.get("sources") or run_group.get("source")
        if source_group is None or "time" not in source_group:
            return np.array([], dtype=np.float64)
        times_ms = np.asarray(source_group["time"], dtype=np.float64)
    return times_ms / 1000.0


def _score_run(
    model_type: str,
    model_path: Path,
    ts: SpectralTimeSeries,
    device: str,
    payload: Dict[str, Any],
    score_type: Optional[str],
    latent_mask_pct: float,
    latent_mask_seed: Optional[int],
    normalize_inputs_l1: bool,
) -> np.ndarray:
    if model_type == "lstm":
        detector = LSTMTemporalDetector(device=device, verbose=False)
        detector.load(str(model_path))
        target_count_rates = payload.get("count_rates")
        if target_count_rates is not None:
            if torch.is_tensor(target_count_rates):
                target_count_rates = target_count_rates.detach().cpu().numpy()
            else:
                target_count_rates = np.asarray(target_count_rates, dtype=np.float32)
            live_times = payload.get("live_times")
            if live_times is not None:
                if torch.is_tensor(live_times):
                    live_times = live_times.detach().cpu().numpy()
                else:
                    live_times = np.asarray(live_times, dtype=np.float32)
                target_count_rates = target_count_rates * live_times
        return detector.score_time_series(
            ts,
            target_count_rates=target_count_rates,
            normalize_inputs_l1=normalize_inputs_l1,
            latent_mask_pct=latent_mask_pct,
            mask_seed=latent_mask_seed,
            score_type=score_type,
        )
    else:
        detector = ARADDetector(device=device, verbose=False)
        detector.load(str(model_path))
        return detector.score_time_series(ts)


def _suppress_low_count_tail_artifact(
    scores: np.ndarray,
    count_rates: Optional[np.ndarray],
    min_fraction_of_median: float = 0.05,
    min_rate_floor: float = 25.0,
) -> np.ndarray:
    """Mask trailing low-count windows that can create end-of-run score spikes."""
    if count_rates is None:
        return np.asarray(scores, dtype=np.float64)

    out = np.asarray(scores, dtype=np.float64).copy()
    rates = np.asarray(count_rates, dtype=np.float64)
    if out.shape != rates.shape:
        return out

    finite_positive = rates[np.isfinite(rates) & (rates > 0.0)]
    if finite_positive.size == 0:
        return out

    median_rate = float(np.median(finite_positive))
    cutoff = max(float(min_rate_floor), float(min_fraction_of_median) * median_rate)

    idx = len(rates) - 1
    while idx >= 0 and np.isfinite(rates[idx]) and (rates[idx] < cutoff):
        out[idx] = np.nan
        idx -= 1

    return out


# ---------------------------------------------------------------------------
# Baseline: build, save, load, apply
# ---------------------------------------------------------------------------

def build_baseline(
    scores: np.ndarray,
    count_rates: np.ndarray,
    n_rate_bins: int = 30,
    percentile: float = 50.0,
) -> Dict[str, Any]:
    """Fit E[score | count_rate] from background-only (score, count_rate) pairs.

    Uses equal-frequency binning so each bin has roughly the same number of
    observations regardless of count-rate distribution skew.

    Returns a dict that can be serialised to JSON and applied with
    ``apply_baseline()``.
    """
    finite = np.isfinite(scores) & np.isfinite(count_rates) & (count_rates > 0)
    s = scores[finite]
    r = count_rates[finite]

    if s.size < n_rate_bins * 5:
        raise ValueError(
            f"Not enough finite background samples ({s.size}) to build a "
            f"reliable baseline with {n_rate_bins} bins. "
            "Reduce --n-rate-bins or add more background runs."
        )

    # Equal-frequency bin edges so each bin is well-populated
    edge_pcts = np.linspace(0, 100, n_rate_bins + 1)
    rate_edges = np.percentile(r, edge_pcts)
    # Avoid duplicate edges at the extremes
    rate_edges = np.unique(rate_edges)

    bin_centers: List[float] = []
    baseline_values: List[float] = []

    for lo, hi in zip(rate_edges[:-1], rate_edges[1:]):
        mask = (r >= lo) & (r <= hi)
        if mask.sum() < 5:
            continue
        bin_centers.append(float((lo + hi) / 2.0))
        baseline_values.append(float(np.percentile(s[mask], percentile)))

    if len(bin_centers) < 2:
        raise ValueError("Too few populated bins to build baseline. Add more background runs.")

    print(f"  Baseline built from {s.size} spectra across {len(bin_centers)} rate bins")
    print(f"  Count rate range: {r.min():.0f} – {r.max():.0f} cts/s")
    print(f"  Baseline score range: {min(baseline_values):.5f} – {max(baseline_values):.5f}")

    return {
        "bin_centers": bin_centers,
        "baseline_values": baseline_values,
        "percentile": float(percentile),
        "n_spectra": int(s.size),
        "count_rate_min": float(r.min()),
        "count_rate_max": float(r.max()),
    }


def save_baseline(baseline: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  Baseline saved to {path}")


def load_baseline(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        baseline = json.load(f)
    print(
        f"  Loaded baseline: {baseline['n_spectra']} spectra, "
        f"{len(baseline['bin_centers'])} bins, "
        f"p{baseline['percentile']:.0f}"
    )
    return baseline


def apply_baseline(
    scores: np.ndarray,
    count_rates: np.ndarray,
    baseline: Dict[str, Any],
) -> np.ndarray:
    """Subtract the baseline curve from raw scores.

    Uses linear interpolation between bin centres with constant extrapolation
    at the edges (clamp to nearest known value).
    """
    centers = np.asarray(baseline["bin_centers"], dtype=np.float64)
    values = np.asarray(baseline["baseline_values"], dtype=np.float64)

    baseline_at_rate = np.interp(count_rates, centers, values,
                                 left=values[0], right=values[-1])
    return scores - baseline_at_rate


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_run(
    run_name: str,
    timestamps: np.ndarray,
    raw_scores: np.ndarray,
    corrected_scores: np.ndarray,
    gross_rate: np.ndarray,
    source_times: Optional[np.ndarray],
    model_type: str,
    snr: RunSNR,
    output_path: Path,
    show_raw: bool,
) -> None:
    n_panels = 3 if show_raw else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 4 * n_panels), sharex=True)

    panel = 0

    if show_raw:
        ax_raw = axes[panel]
        finite = np.isfinite(raw_scores)
        ax_raw.plot(timestamps[finite], raw_scores[finite],
                    color="tab:blue", linewidth=1.0, alpha=0.8, label="Raw score")
        ax_raw.set_ylabel("Raw Score")
        ax_raw.grid(True, alpha=0.3)
        ax_raw.legend(fontsize=8)
        panel += 1

    ax_corr = axes[panel]
    finite = np.isfinite(corrected_scores)
    ax_corr.plot(timestamps[finite], corrected_scores[finite],
                 color="tab:green", linewidth=1.2, label="Baseline-corrected score")
    ax_corr.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                    label="Expected background (0)")
    ax_corr.set_ylabel("Corrected Score\n(anomaly above 0)")
    ax_corr.grid(True, alpha=0.3)
    ax_corr.legend(fontsize=8)
    ax_corr.set_title(
        f"{model_type.upper()} | {run_name} "
        f"({snr.source_events}/{snr.total_events} source events)"
    )
    panel += 1

    ax_rate = axes[panel]
    ax_rate.plot(timestamps, gross_rate, color="tab:orange", linewidth=1.2)
    ax_rate.set_ylabel("Gross Count Rate (cts/s)")
    ax_rate.set_xlabel("Time (s)")
    ax_rate.grid(True, alpha=0.3)

    if source_times is not None and len(source_times) > 0:
        for ax in axes:
            for t in source_times:
                ax.axvline(float(t), color="black", linestyle=":", linewidth=1.8, alpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot count-rate-baseline-corrected anomaly scores"
    )
    # Model
    parser.add_argument("--model-type", choices=["lstm", "arad"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    # Data
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="Preprocessed dir for evaluation runs (with sources)")
    parser.add_argument("--h5-path", type=Path, required=True,
                        help="H5 file for gross count rate and source times")
    parser.add_argument("--output-dir", type=Path, required=True)

    # Baseline — mutually exclusive: build from dir OR load from file
    baseline_group = parser.add_mutually_exclusive_group(required=True)
    baseline_group.add_argument(
        "--baseline-dir", type=Path, default=None,
        help="Preprocessed dir of background-only runs to build the baseline from"
    )
    baseline_group.add_argument(
        "--load-baseline", type=Path, default=None,
        help="Path to a previously saved baseline JSON (skips background scoring)"
    )

    # Baseline tuning
    parser.add_argument("--n-rate-bins", type=int, default=30,
                        help="Number of count-rate bins for baseline curve (default: 30)")
    parser.add_argument("--baseline-percentile", type=float, default=50.0,
                        help="Percentile of background scores used as baseline (default: 50)")
    parser.add_argument("--max-baseline-runs", type=int, default=None,
                        help="Limit number of background runs used (speeds up baseline build)")

    # Run selection
    parser.add_argument("--num-runs", type=int, default=6,
                        help="Number of highest-SNR eval runs to plot")
    parser.add_argument("--run-ids", nargs="*", type=int, default=None,
                        help="Specific run IDs to plot (overrides --num-runs SNR selection)")
    parser.add_argument("--time-units", choices=["us", "ms", "s"], default="us")

    # Scoring
    parser.add_argument(
        "--score-type", type=str, default=None,
        choices=["jsd", "chi2", "normalized_chi2"],
        help="Override scoring metric (default: use model's training loss_type)"
    )
    parser.add_argument("--latent-mask-pct", type=float, default=0.0)
    parser.add_argument("--latent-mask-seed", type=int, default=None)
    parser.add_argument("--normalize-inputs-l1", action="store_true",
                        help="L1-normalize inputs at scoring time (leave off for already-normalized runs)")

    # Display
    parser.add_argument("--show-raw", action="store_true",
                        help="Add a third panel showing the raw (uncorrected) score")
    parser.add_argument("--no-source-lines", action="store_true",
                        help="Disable source-injection dotted lines")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    eval_dir = args.eval_dir.resolve()
    h5_path = args.h5_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for p, name in [(eval_dir, "--eval-dir"), (h5_path, "--h5-path"),
                    (args.model_path, "--model-path")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")

    # ------------------------------------------------------------------
    # Step 1: build or load baseline
    # ------------------------------------------------------------------
    if args.load_baseline is not None:
        print("Loading pre-built baseline...")
        baseline = load_baseline(args.load_baseline.resolve())
    else:
        print(f"Building baseline from {args.baseline_dir} ...")
        baseline_dir = args.baseline_dir.resolve()
        if not baseline_dir.exists():
            raise FileNotFoundError(f"--baseline-dir not found: {baseline_dir}")

        bg_run_files = _discover_runs(baseline_dir)
        if args.max_baseline_runs is not None:
            bg_run_files = bg_run_files[: args.max_baseline_runs]

        all_bg_scores: List[np.ndarray] = []
        all_bg_rates: List[np.ndarray] = []

        for i, run_path in enumerate(bg_run_files, 1):
            try:
                payload, ts = _load_run(run_path)
            except Exception as exc:
                print(f"  [SKIP] {run_path.name}: {exc}")
                continue

            scores = _score_run(
                model_type=args.model_type,
                model_path=args.model_path.resolve(),
                ts=ts,
                device=args.device,
                payload=payload,
                score_type=args.score_type,
                latent_mask_pct=float(args.latent_mask_pct),
                latent_mask_seed=args.latent_mask_seed,
                normalize_inputs_l1=bool(args.normalize_inputs_l1),
            )

            count_rates = payload.get("count_rates")
            if count_rates is None:
                print(f"  [SKIP] {run_path.name}: no count_rates in payload")
                continue
            if torch.is_tensor(count_rates):
                count_rates = count_rates.detach().cpu().numpy()
            else:
                count_rates = np.asarray(count_rates, dtype=np.float32)

            scores = _suppress_low_count_tail_artifact(
                scores,
                count_rates,
            )

            all_bg_scores.append(np.asarray(scores, dtype=np.float64))
            all_bg_rates.append(count_rates.astype(np.float64))

            if i % 20 == 0 or i == len(bg_run_files):
                print(f"  Scored {i}/{len(bg_run_files)} background runs...", end="\r")

        print()

        if not all_bg_scores:
            raise RuntimeError("No background runs scored successfully.")

        all_scores = np.concatenate(all_bg_scores)
        all_rates = np.concatenate(all_bg_rates)

        baseline = build_baseline(
            all_scores, all_rates,
            n_rate_bins=int(args.n_rate_bins),
            percentile=float(args.baseline_percentile),
        )

        baseline_path = output_dir / "baseline.json"
        save_baseline(baseline, baseline_path)

    # ------------------------------------------------------------------
    # Step 2: select eval runs by SNR
    # ------------------------------------------------------------------
    eval_run_files = _discover_runs(eval_dir)
    file_by_id: Dict[int, Path] = {}
    for p in eval_run_files:
        rid = _run_id_from_name(p)
        if rid is not None:
            file_by_id[rid] = p

    candidate_ids = sorted(file_by_id.keys())

    if args.run_ids:
        requested = {int(x) for x in args.run_ids}
        candidate_ids = [rid for rid in candidate_ids if rid in requested]

    snr_rows = _compute_snr_rows(h5_path=h5_path, run_ids=candidate_ids)
    if not snr_rows:
        raise RuntimeError("No overlapping runs found between eval dir and H5.")

    has_sources = any(r.source_events > 0 for r in snr_rows)
    if has_sources:
        snr_rows.sort(key=lambda r: (r.source_fraction, r.source_events), reverse=True)
    else:
        snr_rows.sort(key=lambda r: (r.peak_to_median_rate, r.std_to_mean_rate), reverse=True)

    selected = snr_rows[: max(1, int(args.num_runs))]

    # ------------------------------------------------------------------
    # Step 3: score eval runs, apply baseline, plot
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []

    for row in selected:
        run_path = file_by_id[row.run_id]
        print(f"Processing {run_path.name}...")
        payload, ts = _load_run(run_path)

        raw_scores = _score_run(
            model_type=args.model_type,
            model_path=args.model_path.resolve(),
            ts=ts,
            device=args.device,
            payload=payload,
            score_type=args.score_type,
            latent_mask_pct=float(args.latent_mask_pct),
            latent_mask_seed=args.latent_mask_seed,
            normalize_inputs_l1=bool(args.normalize_inputs_l1),
        )

        count_rates = payload.get("count_rates")
        if count_rates is not None:
            if torch.is_tensor(count_rates):
                count_rates = count_rates.detach().cpu().numpy()
            else:
                count_rates = np.asarray(count_rates, dtype=np.float32)
            count_rates = count_rates.astype(np.float64)
        else:
            count_rates = np.ones(len(raw_scores), dtype=np.float64)

        raw_scores = _suppress_low_count_tail_artifact(raw_scores, count_rates)

        corrected_scores = apply_baseline(
            np.asarray(raw_scores, dtype=np.float64),
            count_rates,
            baseline,
        )
        # nan-propagation: keep warmup nans as nan
        corrected_scores[~np.isfinite(raw_scores)] = np.nan

        integration_time = payload.get("integration_time")
        if integration_time is None:
            integration_time = float(np.median(np.asarray(ts.real_times, dtype=np.float64)))

        gross_rate = _gross_count_rate_from_h5(
            h5_path=h5_path,
            run_id=row.run_id,
            timestamps=np.asarray(ts.timestamps, dtype=np.float64),
            integration_time=float(integration_time),
            time_units=str(args.time_units),
        )

        source_times: Optional[np.ndarray] = None
        if not bool(args.no_source_lines):
            source_times = _source_times_from_h5(h5_path=h5_path, run_id=row.run_id)

        out_png = output_dir / f"{args.model_type}_run{row.run_id}_rate_normalized.png"
        _plot_run(
            run_name=run_path.name,
            timestamps=np.asarray(ts.timestamps, dtype=np.float64),
            raw_scores=np.asarray(raw_scores, dtype=np.float64),
            corrected_scores=corrected_scores,
            gross_rate=gross_rate,
            source_times=source_times,
            model_type=args.model_type,
            snr=row,
            output_path=out_png,
            show_raw=bool(args.show_raw),
        )
        print(f"  Wrote {out_png}")

        finite = np.isfinite(corrected_scores)
        results.append({
            "run_id": row.run_id,
            "run_name": run_path.name,
            "source_fraction": row.source_fraction,
            "source_events": row.source_events,
            "corrected_score_mean": float(corrected_scores[finite].mean()) if finite.any() else None,
            "corrected_score_max": float(corrected_scores[finite].max()) if finite.any() else None,
            "plot_path": str(out_png.resolve()),
        })

    summary = {
        "model_type": args.model_type,
        "model_path": str(args.model_path.resolve()),
        "eval_dir": str(eval_dir),
        "baseline_percentile": baseline["percentile"],
        "n_rate_bins": len(baseline["bin_centers"]),
        "score_type": args.score_type,
        "runs": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
