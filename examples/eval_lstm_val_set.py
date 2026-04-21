"""Evaluate a trained LSTMTemporalDetector on its training validation set.

Reads val_runs from the .metrics.json produced by train_lstm_temporal_preprocessed.py,
scores every run, and saves one plot per run showing:
  - Reconstruction error (score) vs time
  - Horizontal dashed line at the detection threshold
  - Vertical dotted lines bracketing source-present windows (when H5 supplied)

Threshold is taken from the saved model by default.  Pass --alarms-per-hour to
calibrate it from a separate clean-background directory instead.

Example
-------
python examples/eval_lstm_val_set.py \\
    --model-path  models/lstm_jsd_seq20.pt \\
    --metrics-path models/lstm_jsd_seq20.metrics.json \\
    --preprocessed-dir preprocessed-data/with-sources-2.0-1.0-b256 \\
    --h5-path /data/testing_v4.3.h5 \\
    --output-dir eval-results/lstm_jsd_seq20_val \\
    --device cuda

# FAR-based threshold calibration
python examples/eval_lstm_val_set.py \\
    --model-path  models/lstm_jsd_seq20.pt \\
    --metrics-path models/lstm_jsd_seq20.metrics.json \\
    --preprocessed-dir preprocessed-data/with-sources-2.0-1.0-b256 \\
    --calibration-dir preprocessed-data/no-sources-2.0-1.0-b256 \\
    --alarms-per-hour 1.0 \\
    --h5-path /data/testing_v4.3.h5 \\
    --output-dir eval-results/lstm_jsd_seq20_val \\
    --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gammaflow import SpectralTimeSeries
from gammaflow.algorithms import LSTMTemporalDetector

try:
    import h5py
except ImportError:
    h5py = None


# ---------------------------------------------------------------------------
# I/O helpers (reused from plot_high_snr_error_and_rate.py)
# ---------------------------------------------------------------------------

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
    if counts.ndim != 2:
        raise ValueError(f"spectra must be 2D, got {counts.shape}")
    n = counts.shape[0]
    integration_time = obj.get("integration_time")
    rt_default = float(integration_time) if integration_time is not None else 1.0
    timestamps = _as_numpy_1d(obj.get("timestamps"), n, default=0.0)
    real_times = _as_numpy_1d(obj.get("real_times"), n, default=rt_default)
    live_times_raw = obj.get("live_times")
    live_times = (
        None
        if live_times_raw is None
        else _as_numpy_1d(live_times_raw, n, default=rt_default)
    )
    energy_edges = obj.get("energy_edges")
    if energy_edges is not None:
        energy_edges = np.asarray(energy_edges, dtype=np.float64)
    ts = SpectralTimeSeries.from_array(
        counts,
        energy_edges=energy_edges,
        timestamps=timestamps,
        live_times=live_times,
        real_times=real_times,
    )
    return obj, ts


def _target_count_rates_from_payload(
    payload: Dict[str, Any], n_spectra: int
) -> Optional[np.ndarray]:
    """Return per-spectrum total counts (rate × live_time) or None."""
    rates_raw = payload.get("count_rates")
    if rates_raw is None:
        return None
    if torch.is_tensor(rates_raw):
        rates = rates_raw.detach().cpu().numpy().astype(np.float32)
    else:
        rates = np.asarray(rates_raw, dtype=np.float32)
    live_raw = payload.get("live_times")
    if live_raw is not None:
        if torch.is_tensor(live_raw):
            live = live_raw.detach().cpu().numpy().astype(np.float32)
        else:
            live = np.asarray(live_raw, dtype=np.float32)
        rates = rates * live
    return rates


# ---------------------------------------------------------------------------
# H5 label helpers (reused from plot_high_snr_error_and_rate.py)
# ---------------------------------------------------------------------------

def _time_unit_scale(units: str) -> float:
    u = units.strip().lower()
    if u == "us":
        return 1e-6
    if u == "ms":
        return 1e-3
    if u == "s":
        return 1.0
    raise ValueError(f"Unsupported time units '{units}'")


def _source_window_labels_from_h5(
    h5_path: Path,
    run_id: int,
    timestamps: np.ndarray,
    integration_time: float,
    time_units: str,
) -> np.ndarray:
    if h5py is None:
        return np.zeros(len(timestamps), dtype=bool)
    run_key = f"run{run_id}"
    with h5py.File(h5_path, "r") as f:
        runs = f["runs"]
        if run_key not in runs:
            return np.zeros(len(timestamps), dtype=bool)
        run_group = runs[run_key]
        listmode = run_group["listmode"]
        if "id" not in listmode:
            return np.zeros(len(timestamps), dtype=bool)
        event_ids = np.asarray(listmode["id"])
        dt = np.asarray(listmode["dt"], dtype=np.float64)
        source_group = None
        if "source" in run_group:
            source_group = run_group["source"]
        elif "sources" in run_group:
            source_group = run_group["sources"]
        if source_group is None or "id" not in source_group:
            return np.zeros(len(timestamps), dtype=bool)
        source_ids = np.asarray(source_group["id"])
        if source_ids.size == 0:
            return np.zeros(len(timestamps), dtype=bool)
        source_mask = np.isin(event_ids, source_ids)
        if not np.any(source_mask):
            return np.zeros(len(timestamps), dtype=bool)
        abs_times = np.cumsum(dt * _time_unit_scale(time_units))
        src_times = abs_times[source_mask]
    labels = np.zeros(len(timestamps), dtype=bool)
    half = float(integration_time) / 2.0
    for i, center in enumerate(timestamps):
        left = np.searchsorted(src_times, center - half, side="left")
        right = np.searchsorted(src_times, center + half, side="left")
        labels[i] = right > left
    return labels


def _label_intervals(
    timestamps: np.ndarray, labels: np.ndarray
) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    if labels.size == 0:
        return intervals
    in_seg = False
    start_t = 0.0
    for t, v in zip(timestamps, labels):
        if bool(v) and not in_seg:
            in_seg = True
            start_t = float(t)
        elif not bool(v) and in_seg:
            in_seg = False
            intervals.append((start_t, float(t)))
    if in_seg:
        intervals.append((start_t, float(timestamps[-1])))
    return intervals


def _run_id_from_name(path: Path) -> Optional[int]:
    stem = path.stem
    if not stem.startswith("run"):
        return None
    try:
        return int(stem.replace("run", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_run(
    detector: LSTMTemporalDetector,
    ts: SpectralTimeSeries,
    target_count_rates: Optional[np.ndarray],
    latent_mask_pct: float,
    mask_seed: Optional[int],
    score_type: Optional[str] = None,
) -> np.ndarray:
    scores = detector.score_time_series(
        ts,
        target_count_rates=target_count_rates,
        latent_mask_pct=float(latent_mask_pct),
        mask_seed=mask_seed,
        score_type=score_type,
    )
    return _suppress_low_count_tail_artifact(scores, target_count_rates)


def _suppress_low_count_tail_artifact(
    scores: np.ndarray,
    target_scales: Optional[np.ndarray],
    min_fraction_of_median: float = 0.05,
    min_scale_floor: float = 50.0,
) -> np.ndarray:
    """Mask trailing low-count windows that can create end-of-run score spikes."""
    if target_scales is None:
        return np.asarray(scores, dtype=np.float64)

    out = np.asarray(scores, dtype=np.float64).copy()
    scales = np.asarray(target_scales, dtype=np.float64)
    if out.shape != scales.shape:
        return out

    finite_positive = scales[np.isfinite(scales) & (scales > 0.0)]
    if finite_positive.size == 0:
        return out

    median_scale = float(np.median(finite_positive))
    cutoff = max(float(min_scale_floor), float(min_fraction_of_median) * median_scale)

    idx = len(scales) - 1
    while idx >= 0 and np.isfinite(scales[idx]) and (scales[idx] < cutoff):
        out[idx] = np.nan
        idx -= 1

    return out


def _calibrate_threshold(
    detector: LSTMTemporalDetector,
    calibration_dir: Path,
    alarms_per_hour: float,
    latent_mask_pct: float,
    mask_seed: Optional[int],
    score_type: Optional[str] = None,
) -> float:
    """Calibrate threshold to target FAR using clean background runs.

    Scores each run independently (avoiding one giant window array in memory),
    concatenates only the resulting scores and timestamps, then does a binary
    search on actual alarm-aggregation counts.
    """
    run_files = sorted(
        calibration_dir.glob("run*.pt"),
        key=lambda p: int(p.stem.replace("run", "")),
    )
    if not run_files:
        raise FileNotFoundError(f"No run*.pt files found in {calibration_dir}")

    all_scores: List[np.ndarray] = []
    all_timestamps: List[np.ndarray] = []
    t_offset = 0.0

    print(f"Scoring {len(run_files)} calibration runs ...")
    for run_path in run_files:
        payload, ts = _load_run(run_path)
        target_count_rates = _target_count_rates_from_payload(payload, ts.n_spectra)

        scores = detector.score_time_series(
            ts,
            target_count_rates=target_count_rates,
            latent_mask_pct=float(latent_mask_pct),
            mask_seed=mask_seed,
            score_type=score_type,
        )
        scores = _suppress_low_count_tail_artifact(scores, target_count_rates)

        timestamps = np.asarray(ts.timestamps, dtype=np.float64)
        real_times = np.asarray(ts.real_times, dtype=np.float64)
        timestamps = timestamps - timestamps[0] + t_offset
        t_offset = float(timestamps[-1]) + float(np.median(real_times))

        all_scores.append(np.asarray(scores, dtype=np.float64))
        all_timestamps.append(timestamps)

        del payload, ts, target_count_rates, scores, timestamps, real_times
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores_cat = np.concatenate(all_scores)
    times_cat = np.concatenate(all_timestamps)
    finite_scores = scores_cat[np.isfinite(scores_cat)]

    if finite_scores.size == 0:
        raise ValueError("No finite scores from calibration runs.")

    total_time_seconds = float(times_cat[-1] - times_cat[0])
    if total_time_seconds <= 0:
        raise ValueError(f"Invalid calibration time span: {total_time_seconds}s")
    total_time_hours = total_time_seconds / 3600.0

    print(
        f"  {finite_scores.size} finite scores over {total_time_hours:.2f} hours\n"
        f"  Score range: [{finite_scores.min():.4f}, {finite_scores.max():.4f}]\n"
        f"  Score mean +/- std: {finite_scores.mean():.4f} +/- {finite_scores.std():.4f}"
    )

    # Binary search on actual alarm aggregation counts.
    low = float(finite_scores.min())
    high = float(finite_scores.max()) * 1.5
    best_threshold = float(np.percentile(finite_scores, 99.0))
    best_far_diff = float("inf")

    for iteration in range(20):
        test_threshold = (low + high) / 2.0
        detector.threshold = test_threshold
        detector._aggregate_alarms_from_scores(
            scores=scores_cat,
            times=times_cat,
            threshold=test_threshold,
        )
        n_alarms = len(detector.alarms)
        observed_far = n_alarms / total_time_hours
        far_diff = abs(observed_far - alarms_per_hour)

        print(
            f"  Iter {iteration + 1}: threshold={test_threshold:.6f} "
            f"-> {n_alarms} alarms ({observed_far:.2f}/hr)"
        )

        if far_diff < best_far_diff:
            best_far_diff = far_diff
            best_threshold = test_threshold

        if observed_far > alarms_per_hour:
            low = test_threshold
        else:
            high = test_threshold

        if far_diff < 0.1 * alarms_per_hour or (high - low) < 1e-8:
            print(f"  Converged after {iteration + 1} iterations")
            break

    detector.threshold = best_threshold
    detector._aggregate_alarms_from_scores(
        scores=scores_cat, times=times_cat, threshold=best_threshold
    )
    final_far = len(detector.alarms) / total_time_hours
    print(
        f"\n  Threshold set: {best_threshold:.6f}\n"
        f"  Achieved FAR: {final_far:.2f} alarms/hour ({len(detector.alarms)} alarms)\n"
        f"  Target FAR: {alarms_per_hour:.2f} alarms/hour"
    )
    return best_threshold


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_run(
    run_name: str,
    timestamps: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float],
    source_intervals: Optional[Sequence[Tuple[float, float]]],
    title_extra: str,
    output_path: Path,
    log_y: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))

    finite = np.isfinite(scores)
    t_plot = timestamps[finite]
    s_plot = scores[finite]

    if log_y and np.any(s_plot > 0):
        ax.semilogy(t_plot, s_plot, color="tab:blue", linewidth=1.2, label="Score")
    else:
        ax.plot(t_plot, s_plot, color="tab:blue", linewidth=1.2, label="Score")

    if threshold is not None:
        ax.axhline(
            threshold,
            color="tab:red",
            linestyle="--",
            linewidth=1.8,
            label=f"Threshold = {threshold:.4g}",
        )

    if source_intervals:
        for i, (start_t, end_t) in enumerate(source_intervals):
            label = "Source present" if i == 0 else None
            ax.axvline(
                start_t, color="black", linestyle=":", linewidth=2.0, alpha=0.9, label=label
            )
            ax.axvline(end_t, color="black", linestyle=":", linewidth=2.0, alpha=0.9)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title(f"LSTM Val Run: {run_name}{title_extra}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LSTM model on training validation set"
    )
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Trained LSTMTemporalDetector checkpoint")
    parser.add_argument("--metrics-path", type=Path, required=True,
                        help=".metrics.json written by train_lstm_temporal_preprocessed.py")
    parser.add_argument("--preprocessed-dir", type=Path, required=True,
                        help="Directory containing run*.pt files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--h5-path", type=Path, default=None,
                        help="H5 dataset file for source labels (optional)")
    parser.add_argument("--time-units", choices=["us", "ms", "s"], default="us")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent-mask-pct", type=float, default=0.0)
    parser.add_argument("--latent-mask-seed", type=int, default=None)
    parser.add_argument("--log-y", action="store_true",
                        help="Use log scale on the score axis")
    # Threshold options
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument("--threshold", type=float, default=None,
                                 help="Explicit detection threshold")
    threshold_group.add_argument("--alarms-per-hour", type=float, default=None,
                                 help="Calibrate threshold to this FAR from --calibration-dir")
    parser.add_argument("--calibration-dir", type=Path, default=None,
                        help="Clean-background run*.pt dir for FAR threshold calibration")
    parser.add_argument("--score-type", type=str, default=None,
                        choices=["jsd", "chi2", "normalized_chi2"],
                        help="Override scoring metric (default: use model's loss_type). "
                             "'jsd': rate-invariant JSD; "
                             "'normalized_chi2': chi2/total_counts, rate-invariant shape error.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = args.model_path.resolve()
    metrics_path = args.metrics_path.resolve()
    preprocessed_dir = args.preprocessed_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed dir not found: {preprocessed_dir}")

    # Read val_runs from metrics JSON
    metrics = json.loads(metrics_path.read_text())
    val_run_names: List[str] = metrics.get("val_runs", [])
    if not val_run_names:
        raise ValueError(f"No 'val_runs' found in {metrics_path}")
    val_run_set = {Path(n).name for n in val_run_names}

    # Match to actual files
    all_run_files = sorted(
        preprocessed_dir.glob("run*.pt"),
        key=lambda p: int(p.stem.replace("run", "")),
    )
    val_files = [p for p in all_run_files if p.name in val_run_set]
    if not val_files:
        raise RuntimeError(
            f"None of the val_runs from metrics were found in {preprocessed_dir}.\n"
            f"Expected names: {sorted(val_run_set)}"
        )

    missing = val_run_set - {p.name for p in val_files}
    if missing:
        print(f"Warning: {len(missing)} val run(s) not found on disk: {sorted(missing)}")

    # Load model
    detector = LSTMTemporalDetector(device=args.device, verbose=True)
    detector.load(str(model_path))
    print(f"Loaded model: {model_path}")
    print(f"  loss_type={detector.loss_type}, seq_len={detector.seq_len}, "
          f"seq_stride={detector.seq_stride}, use_attention={detector.use_attention}")

    # Resolve threshold
    if args.threshold is not None:
        detector.threshold = float(args.threshold)
        print(f"Using explicit threshold: {detector.threshold:.6g}")
    elif args.alarms_per_hour is not None:
        if args.calibration_dir is None:
            raise ValueError("--calibration-dir is required when using --alarms-per-hour")
        cal_dir = args.calibration_dir.resolve()
        if not cal_dir.exists():
            raise FileNotFoundError(f"Calibration dir not found: {cal_dir}")
        print(f"Calibrating threshold at {args.alarms_per_hour} alarms/hour ...")
        threshold = _calibrate_threshold(
            detector=detector,
            calibration_dir=cal_dir,
            alarms_per_hour=float(args.alarms_per_hour),
            latent_mask_pct=float(args.latent_mask_pct),
            mask_seed=args.latent_mask_seed,
            score_type=args.score_type,
        )
        print(f"Calibrated threshold: {threshold:.6g}")
    else:
        if detector.threshold is not None:
            print(f"Using model's saved threshold: {detector.threshold:.6g}")
        else:
            print("No threshold set — threshold line will be omitted from plots")

    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for run_path in val_files:
        run_id = _run_id_from_name(run_path)
        print(f"Scoring {run_path.name} ...", end=" ", flush=True)

        payload, ts = _load_run(run_path)
        target_count_rates = _target_count_rates_from_payload(payload, ts.n_spectra)

        scores = _score_run(
            detector=detector,
            ts=ts,
            target_count_rates=target_count_rates,
            latent_mask_pct=float(args.latent_mask_pct),
            mask_seed=args.latent_mask_seed,
            score_type=args.score_type,
        )

        finite_scores = scores[np.isfinite(scores)]
        n_finite = int(finite_scores.size)
        score_max = float(np.max(finite_scores)) if n_finite > 0 else float("nan")
        score_mean = float(np.mean(finite_scores)) if n_finite > 0 else float("nan")

        # Source labels from H5
        source_intervals: Optional[List[Tuple[float, float]]] = None
        n_source_windows = 0
        if args.h5_path is not None and run_id is not None:
            if h5py is None:
                print("\nWarning: h5py not installed — source labels unavailable")
            else:
                integration_time = payload.get("integration_time")
                if integration_time is None:
                    integration_time = float(
                        np.median(np.asarray(ts.real_times, dtype=np.float64))
                    )
                labels = _source_window_labels_from_h5(
                    h5_path=args.h5_path.resolve(),
                    run_id=run_id,
                    timestamps=np.asarray(ts.timestamps, dtype=np.float64),
                    integration_time=float(integration_time),
                    time_units=str(args.time_units),
                )
                source_intervals = _label_intervals(
                    timestamps=np.asarray(ts.timestamps, dtype=np.float64),
                    labels=labels,
                )
                n_source_windows = int(np.sum(labels))

        alarm_events: List[Tuple[float, float]] = []
        if detector.threshold is not None:
            timestamps_np = np.asarray(ts.timestamps, dtype=np.float64)
            detector._aggregate_alarms_from_scores(
                scores=np.asarray(scores, dtype=np.float64),
                times=timestamps_np,
                threshold=float(detector.threshold),
            )
            alarm_events = [(float(a.start_time), float(a.end_time)) for a in detector.alarms]
        n_alarm_events = len(alarm_events)

        title_extra = ""
        if n_source_windows > 0:
            title_extra = f"  [{n_source_windows} source windows]"

        out_png = output_dir / f"lstm_val_{run_path.stem}.png"
        _plot_run(
            run_name=run_path.name,
            timestamps=np.asarray(ts.timestamps, dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float64),
            threshold=detector.threshold,
            source_intervals=source_intervals,
            title_extra=title_extra,
            output_path=out_png,
            log_y=bool(args.log_y),
        )

        print(f"max={score_max:.4g}  mean={score_mean:.4g}  "
              f"alarm_events={n_alarm_events}  source_windows={n_source_windows}  -> {out_png.name}")

        results.append({
            "run_name": run_path.name,
            "run_id": run_id,
            "n_spectra": ts.n_spectra,
            "n_finite_scores": n_finite,
            "score_max": score_max,
            "score_mean": score_mean,
            "n_alarm_events": n_alarm_events,
            "n_source_windows": n_source_windows,
            "n_source_intervals": 0 if source_intervals is None else len(source_intervals),
            "plot_path": str(out_png.resolve()),
        })

        del payload, ts, scores, finite_scores, alarm_events
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary: Dict[str, Any] = {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "preprocessed_dir": str(preprocessed_dir),
        "h5_path": str(args.h5_path.resolve()) if args.h5_path else None,
        "threshold": detector.threshold,
        "loss_type": detector.loss_type,
        "score_type": args.score_type,
        "seq_len": detector.seq_len,
        "seq_stride": detector.seq_stride,
        "use_attention": detector.use_attention,
        "latent_mask_pct": float(args.latent_mask_pct),
        "n_val_runs": len(val_files),
        "runs": results,
    }
    summary_path = output_dir / "eval_val_set_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
