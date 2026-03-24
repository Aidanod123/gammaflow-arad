"""Evaluate trained LSTMTemporalDetector models on preprocessed RADAI runs.

This script supports:
1) Threshold calibration on clean background runs (target FAR), and
2) Evaluation on runs with injected sources/anomalies.

It can evaluate one or many model checkpoints in a single run.

Examples:
    # Evaluate a single model with FAR-calibrated threshold
    python examples/evaluate_lstm_temporal_models.py \
      --models models/jsd-20-1-.5.pt \
      --calibration-dir preprocessed-data/no-sources-.5-0.1 \
      --eval-dir preprocessed-data/with-sources-0.5-0.1 \
      --alarms-per-hour 1.0 \
      --output-dir eval-results/jsd-20-1-.5

    # Evaluate all trained models and include label-aware metrics
    python examples/evaluate_lstm_temporal_models.py \
      --model-glob "models/*.pt" \
      --calibration-dir preprocessed-data/no-sources-.5-0.1 \
      --eval-dir preprocessed-data/with-sources-0.5-0.1 \
      --h5-path RADAI/training_v4.3.h5 \
      --output-dir eval-results/all-models
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError("PyTorch is required. Install with: pip install torch") from exc

try:
    import h5py
except ImportError:
    h5py = None


def _maybe_import_wandb(enabled: bool):
    if not enabled:
        return None
    try:
        import wandb  # type: ignore

        return wandb
    except ImportError as exc:
        raise ImportError(
            "W&B was requested but is not installed. Install with: pip install wandb"
        ) from exc

# Ensure local repo package is importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gammaflow import SpectralTimeSeries
from gammaflow.algorithms import LSTMTemporalDetector


@dataclass
class RunData:
    run_name: str
    run_id: Optional[int]
    payload: Dict[str, object]
    time_series: SpectralTimeSeries


@dataclass
class LabelMetrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    tpr: float
    fpr: float


@dataclass
class TimeOverlapMetrics:
    positive_windows: int
    positive_seconds: float
    alarms_total: int
    alarms_hit_positive: int
    detected_positive_windows: int
    detected_positive_seconds: float
    alarm_hit_rate: float
    window_coverage_rate: float
    time_coverage_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained LSTMTemporalDetector models on preprocessed runs"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Explicit model checkpoint paths (.pt)",
    )
    parser.add_argument(
        "--model-glob",
        type=str,
        default=None,
        help='Glob for model checkpoints, e.g. "models/*.pt"',
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Directory of clean background run*.pt files used to calibrate threshold",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        required=True,
        help="Directory of evaluation run*.pt files (with injected sources/anomalies)",
    )
    parser.add_argument(
        "--alarms-per-hour",
        type=float,
        default=1.0,
        help="Target FAR (alarms/hour) for threshold calibration",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed threshold. If provided, calibration is skipped.",
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        default=None,
        help="Optional RADAI HDF5 path for label-aware metrics",
    )
    parser.add_argument(
        "--time-units",
        type=str,
        default=None,
        choices=["us", "ms", "s"],
        help="Override time units when reading dt from H5 (default: infer from run payload)",
    )
    parser.add_argument(
        "--latent-mask-pct",
        type=float,
        default=0.0,
        help="Random post-encoder latent history mask percentage in [0, 1]",
    )
    parser.add_argument(
        "--latent-mask-seed",
        type=int,
        default=None,
        help="Optional random seed for latent masking",
    )
    parser.add_argument(
        "--latent-mask-file",
        type=str,
        default=None,
        help="Optional JSON map of run identifiers to explicit masked spectrum indices",
    )
    parser.add_argument(
        "--mask-alarm-feedback",
        action="store_true",
        help=(
            "Mask spectra that crossed threshold as latent history in subsequent "
            "windows (alarm-feedback masking)."
        ),
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on number of eval runs",
    )
    parser.add_argument(
        "--max-calibration-runs",
        type=int,
        default=None,
        help="Optional cap on number of calibration background runs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where evaluation JSON files are written",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce per-run logging",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="gammaflow-lstm")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Comma-separated W&B tags (example: eval,injected-sources)",
    )
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode. Use offline for air-gapped runs.",
    )
    return parser.parse_args()


def _discover_model_paths(args: argparse.Namespace) -> List[Path]:
    model_paths: List[Path] = []

    if args.models:
        model_paths.extend(Path(p) for p in args.models)

    if args.model_glob:
        root = REPO_ROOT
        model_paths.extend(sorted(root.glob(args.model_glob)))

    unique: List[Path] = []
    seen = set()
    for p in model_paths:
        rp = p if p.is_absolute() else (REPO_ROOT / p)
        rp = rp.resolve()
        if str(rp) not in seen:
            unique.append(rp)
            seen.add(str(rp))

    if not unique:
        raise ValueError("No model checkpoints resolved. Use --models and/or --model-glob.")

    missing = [str(p) for p in unique if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

    return unique


def _discover_run_files(run_dir: Path, max_runs: Optional[int] = None) -> List[Path]:
    run_files = sorted(run_dir.glob("run*.pt"), key=lambda p: int(p.stem.replace("run", "")))
    if max_runs is not None:
        run_files = run_files[:max_runs]
    if not run_files:
        raise FileNotFoundError(f"No run*.pt files found in {run_dir}")
    return run_files


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


def _run_id_from_name(path: Path) -> Optional[int]:
    stem = path.stem
    if not stem.startswith("run"):
        return None
    try:
        return int(stem.replace("run", ""))
    except ValueError:
        return None


def _load_run_data(path: Path) -> RunData:
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
    timestamps = _as_numpy_1d(obj.get("timestamps"), n_spectra, default=0.0)

    integration_time = obj.get("integration_time")
    stride_time = obj.get("stride_time")

    # If real_times are missing, use integration_time if available; else 1.0 s fallback.
    rt_default = float(integration_time) if integration_time is not None else 1.0
    real_times = _as_numpy_1d(obj.get("real_times"), n_spectra, default=rt_default)

    live_times_value = obj.get("live_times")
    live_times = None if live_times_value is None else _as_numpy_1d(live_times_value, n_spectra, default=rt_default)

    energy_edges = obj.get("energy_edges")
    if energy_edges is not None:
        energy_edges = np.asarray(energy_edges, dtype=np.float64)

    ts = SpectralTimeSeries.from_array(
        counts,
        energy_edges=energy_edges,
        timestamps=timestamps,
        live_times=live_times,
        real_times=real_times,
        integration_time=float(integration_time) if integration_time is not None else None,
        stride_time=float(stride_time) if stride_time is not None else None,
    )

    return RunData(
        run_name=path.name,
        run_id=_run_id_from_name(path),
        payload=obj,
        time_series=ts,
    )


def _build_concatenated_time_series(runs: Sequence[RunData]) -> SpectralTimeSeries:
    if not runs:
        raise ValueError("No runs provided for concatenation")

    counts_parts: List[np.ndarray] = []
    timestamps_parts: List[np.ndarray] = []
    real_times_parts: List[np.ndarray] = []
    live_times_parts: List[np.ndarray] = []
    has_any_live = False
    energy_edges = runs[0].time_series.energy_edges

    t_offset = 0.0
    for idx, run in enumerate(runs):
        ts = run.time_series
        counts_parts.append(np.asarray(ts.counts, dtype=np.float64))

        ts_times = np.asarray(ts.timestamps, dtype=np.float64)
        if ts_times.size > 0:
            ts_times = ts_times - ts_times[0] + t_offset
            timestamps_parts.append(ts_times)

            rt = np.asarray(ts.real_times, dtype=np.float64)
            t_offset = float(ts_times[-1] + rt[-1])
            real_times_parts.append(rt)

            lt_arr = ts.live_times
            if lt_arr is not None and not (lt_arr.dtype == object and lt_arr[0] is None):
                live_times_parts.append(np.asarray(lt_arr, dtype=np.float64))
                has_any_live = True

        if idx < (len(runs) - 1):
            # Insert a tiny separation to avoid equal timestamps across run boundaries.
            t_offset += 1e-6

    counts = np.vstack(counts_parts)
    timestamps = np.concatenate(timestamps_parts) if timestamps_parts else np.arange(len(counts), dtype=float)
    real_times = np.concatenate(real_times_parts) if real_times_parts else np.ones(len(counts), dtype=float)

    live_times = None
    if has_any_live and live_times_parts:
        live_times = np.concatenate(live_times_parts)

    return SpectralTimeSeries.from_array(
        counts,
        energy_edges=energy_edges,
        timestamps=timestamps,
        live_times=live_times,
        real_times=real_times,
    )


def _safe_stats(values: np.ndarray) -> Dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
        }

    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p50": float(np.percentile(finite, 50.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "p99": float(np.percentile(finite, 99.0)),
    }


def _time_unit_scale(units: str) -> float:
    u = units.strip().lower()
    if u == "us":
        return 1e-6
    if u == "ms":
        return 1e-3
    if u == "s":
        return 1.0
    raise ValueError(f"Unsupported time units '{units}'. Expected one of: us, ms, s")


def _source_window_labels_from_h5(
    h5_path: Path,
    run_id: int,
    timestamps: np.ndarray,
    integration_time: float,
    time_units: str,
) -> np.ndarray:
    if h5py is None:
        raise RuntimeError("h5py is required for H5-based labels. Install with: pip install h5py")

    run_key = f"run{run_id}"
    with h5py.File(h5_path, "r") as f:
        run_group = f["runs"][run_key]
        listmode = run_group["listmode"]

        dt = np.asarray(listmode["dt"])
        event_ids = np.asarray(listmode["id"]) if "id" in listmode else None

        source_group = None
        if "source" in run_group:
            source_group = run_group["source"]
        elif "sources" in run_group:
            source_group = run_group["sources"]

        if source_group is None or "id" not in source_group or event_ids is None:
            return np.zeros_like(timestamps, dtype=bool)

        source_ids = np.asarray(source_group["id"])
        if source_ids.size == 0:
            return np.zeros_like(timestamps, dtype=bool)

        source_event_mask = np.isin(event_ids, source_ids)
        if not np.any(source_event_mask):
            return np.zeros_like(timestamps, dtype=bool)

        scale = _time_unit_scale(time_units)
        abs_times = np.cumsum(np.asarray(dt, dtype=np.float64) * scale)
        src_event_times = abs_times[source_event_mask]

    labels = np.zeros_like(timestamps, dtype=bool)
    half = integration_time / 2.0
    for i, center in enumerate(timestamps):
        t0 = center - half
        t1 = center + half
        left = np.searchsorted(src_event_times, t0, side="left")
        right = np.searchsorted(src_event_times, t1, side="left")
        labels[i] = right > left

    return labels


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> LabelMetrics:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    tn = int(np.sum((~y_true) & (~y_pred)))
    fn = int(np.sum(y_true & (~y_pred)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return LabelMetrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        tpr=tpr,
        fpr=fpr,
    )


def _enrich_alarms_with_ground_truth(
    alarms: Sequence[Dict[str, float]],
    timestamps: np.ndarray,
    labels: np.ndarray,
) -> List[Dict[str, object]]:
    """
    Enrich each alarm event with an `is_true_positive` field based on ground truth labels.
    
    An alarm is considered a true positive if it overlaps with any positive label window.
    
    Parameters
    ----------
    alarms : sequence of dicts with 'start_time' and 'end_time'
    timestamps : array of per-spectrum timestamps
    labels : boolean array marking positive windows
    
    Returns
    -------
    List of enriched alarm dicts with added 'is_true_positive' field
    """
    enriched = []
    for alarm in alarms:
        alarm_dict = dict(alarm)  # Make a copy
        
        start = float(alarm.get("start_time", float("inf")))
        end = float(alarm.get("end_time", float("-inf")))
        
        # Check if any positive label overlaps with this alarm interval
        in_alarm = (timestamps >= start) & (timestamps <= end)
        is_tp = bool(np.any(in_alarm & labels))
        alarm_dict["is_true_positive"] = is_tp
        
        enriched.append(alarm_dict)
    
    return enriched


def _time_overlap_metrics(
    timestamps: np.ndarray,
    real_times: np.ndarray,
    labels: np.ndarray,
    alarms: Sequence[Dict[str, float]],
) -> TimeOverlapMetrics:
    # Positive windows/time from weak labels.
    positive_windows = int(np.sum(labels))
    positive_seconds = float(np.sum(real_times[labels])) if labels.size else 0.0

    # Build alarm-coverage mask by checking each timestamp against any alarm interval.
    alarm_mask = np.zeros_like(labels, dtype=bool)
    alarms_hit_positive = 0
    for alarm in alarms:
        start = float(alarm["start_time"])
        end = float(alarm["end_time"])
        in_alarm = (timestamps >= start) & (timestamps <= end)
        alarm_mask |= in_alarm
        if np.any(in_alarm & labels):
            alarms_hit_positive += 1

    detected_positive_windows = int(np.sum(alarm_mask & labels))
    detected_positive_seconds = float(np.sum(real_times[alarm_mask & labels])) if labels.size else 0.0

    alarms_total = len(alarms)
    alarm_hit_rate = (alarms_hit_positive / alarms_total) if alarms_total > 0 else 0.0
    window_coverage_rate = (
        detected_positive_windows / positive_windows if positive_windows > 0 else 0.0
    )
    time_coverage_rate = (
        detected_positive_seconds / positive_seconds if positive_seconds > 0 else 0.0
    )

    return TimeOverlapMetrics(
        positive_windows=positive_windows,
        positive_seconds=positive_seconds,
        alarms_total=alarms_total,
        alarms_hit_positive=alarms_hit_positive,
        detected_positive_windows=detected_positive_windows,
        detected_positive_seconds=detected_positive_seconds,
        alarm_hit_rate=alarm_hit_rate,
        window_coverage_rate=window_coverage_rate,
        time_coverage_rate=time_coverage_rate,
    )


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _load_explicit_mask_map(mask_file: Optional[Path]) -> Dict[str, List[int]]:
    if mask_file is None:
        return {}

    with mask_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("latent mask file must be a JSON object mapping run ids to index lists")

    out: Dict[str, List[int]] = {}
    for key, value in payload.items():
        if not isinstance(value, list):
            raise ValueError(f"Mask indices for key '{key}' must be a list")
        out[str(key)] = [int(v) for v in value if int(v) >= 0]
    return out


def _resolve_run_mask_indices(
    explicit_mask_map: Dict[str, List[int]],
    run: RunData,
) -> Optional[List[int]]:
    if not explicit_mask_map:
        return None

    candidates = [run.run_name, Path(run.run_name).stem]
    if run.run_id is not None:
        candidates.extend([str(run.run_id), f"run{run.run_id}"])

    for key in candidates:
        if key in explicit_mask_map:
            return sorted(set(int(i) for i in explicit_mask_map[key] if int(i) >= 0))

    return None


def _flatten_dict(prefix: str, payload: Dict[str, Any], out: Dict[str, Any]) -> None:
    for key, value in payload.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_dict(full_key, value, out)
        elif isinstance(value, (int, float, bool, str)) or value is None:
            out[full_key] = value


def _build_eval_wandb_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "eval_dir": args.eval_dir,
        "calibration_dir": args.calibration_dir,
        "max_calibration_runs": args.max_calibration_runs,
        "threshold": args.threshold,
        "alarms_per_hour": args.alarms_per_hour,
        "h5_path": args.h5_path,
        "time_units": args.time_units,
        "max_runs": args.max_runs,
        "latent_mask_pct": args.latent_mask_pct,
        "latent_mask_seed": args.latent_mask_seed,
        "latent_mask_file": args.latent_mask_file,
        "mask_alarm_feedback": args.mask_alarm_feedback,
    }


def _aggregate_point_metrics(rows: Sequence[LabelMetrics]) -> Optional[Dict[str, float]]:
    if not rows:
        return None

    tp = sum(r.tp for r in rows)
    fp = sum(r.fp for r in rows)
    tn = sum(r.tn for r in rows)
    fn = sum(r.fn for r in rows)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    tpr = recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
    }


def evaluate_model(
    model_path: Path,
    eval_runs: Sequence[RunData],
    threshold: float,
    h5_path: Optional[Path],
    time_units_override: Optional[str],
    quiet: bool,
    latent_mask_pct: float,
    latent_mask_seed: Optional[int],
    explicit_mask_map: Dict[str, List[int]],
    mask_alarm_feedback: bool,
) -> Dict[str, object]:
    detector = LSTMTemporalDetector(verbose=not quiet)
    detector.load(str(model_path))
    detector.set_threshold(float(threshold))

    run_results: List[Dict[str, object]] = []
    point_metric_rows: List[LabelMetrics] = []
    overlap_rows: List[TimeOverlapMetrics] = []

    for run in eval_runs:
        ts = run.time_series
        run_mask_indices = _resolve_run_mask_indices(explicit_mask_map, run)
        run_mask_seed = latent_mask_seed
        if run_mask_seed is not None and run.run_id is not None:
            run_mask_seed = int(run_mask_seed) + int(run.run_id)

        scores = detector.process_time_series(
            ts,
            mask_indices=run_mask_indices,
            latent_mask_pct=float(latent_mask_pct),
            mask_seed=run_mask_seed,
            mask_alarm_feedback=bool(mask_alarm_feedback),
        )
        alarms = detector.get_alarm_summary().get("alarm_events", [])

        finite_scores = scores[np.isfinite(scores)]
        threshold_crossings = int(np.sum(finite_scores > detector.threshold))

        run_out: Dict[str, object] = {
            "run_name": run.run_name,
            "run_id": run.run_id,
            "n_spectra": int(ts.n_spectra),
            "n_bins": int(ts.n_bins),
            "warmup_samples": int(detector.warmup_samples),
            "finite_scores": int(finite_scores.size),
            "score_stats": _safe_stats(scores),
            "threshold": float(detector.threshold),
            "threshold_crossings": threshold_crossings,
            "alarm_summary": detector.get_alarm_summary(),
            "latent_masking": {
                "latent_mask_pct": float(latent_mask_pct),
                "latent_mask_seed": run_mask_seed,
                "explicit_mask_count": int(len(run_mask_indices or [])),
                "mask_alarm_feedback": bool(mask_alarm_feedback),
            },
        }

        # Optional weak labels from H5 source event IDs.
        if h5_path is not None and run.run_id is not None:
            integration_time = run.payload.get("integration_time")
            timestamps = np.asarray(ts.timestamps, dtype=np.float64)
            real_times = np.asarray(ts.real_times, dtype=np.float64)

            if integration_time is None:
                integration_time = float(np.median(real_times))

            units = time_units_override or str(run.payload.get("time_units", "us"))
            labels = _source_window_labels_from_h5(
                h5_path=h5_path,
                run_id=run.run_id,
                timestamps=timestamps,
                integration_time=float(integration_time),
                time_units=units,
            )

            # Enrich alarm events with ground truth match information
            enriched_alarms = _enrich_alarms_with_ground_truth(
                alarms=alarms,
                timestamps=timestamps,
                labels=labels,
            )
            run_out["alarm_summary"]["alarm_events"] = enriched_alarms

            valid = np.isfinite(scores)
            y_true = labels[valid]
            y_pred = (scores[valid] > detector.threshold)

            m = _binary_metrics(y_true, y_pred)
            point_metric_rows.append(m)

            o = _time_overlap_metrics(
                timestamps=timestamps,
                real_times=real_times,
                labels=labels,
                alarms=alarms,
            )
            overlap_rows.append(o)

            run_out["label_metrics"] = {
                "tp": m.tp,
                "fp": m.fp,
                "tn": m.tn,
                "fn": m.fn,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "tpr": m.tpr,
                "fpr": m.fpr,
            }
            run_out["time_overlap_metrics"] = {
                "positive_windows": o.positive_windows,
                "positive_seconds": o.positive_seconds,
                "alarms_total": o.alarms_total,
                "alarms_hit_positive": o.alarms_hit_positive,
                "detected_positive_windows": o.detected_positive_windows,
                "detected_positive_seconds": o.detected_positive_seconds,
                "alarm_hit_rate": o.alarm_hit_rate,
                "window_coverage_rate": o.window_coverage_rate,
                "time_coverage_rate": o.time_coverage_rate,
            }

        run_results.append(run_out)

    all_scores = np.concatenate(
        [
            detector.score_time_series(
                r.time_series,
                mask_indices=_resolve_run_mask_indices(explicit_mask_map, r),
                latent_mask_pct=float(latent_mask_pct),
                mask_seed=(int(latent_mask_seed) + int(r.run_id))
                if (latent_mask_seed is not None and r.run_id is not None)
                else latent_mask_seed,
                mask_alarm_feedback=bool(mask_alarm_feedback),
                feedback_threshold=float(detector.threshold),
            )
            for r in eval_runs
        ]
    )

    summary: Dict[str, object] = {
        "model_path": str(model_path),
        "threshold": float(detector.threshold),
        "n_runs": len(eval_runs),
        "total_spectra": int(sum(r.time_series.n_spectra for r in eval_runs)),
        "score_stats": _safe_stats(all_scores),
        "latent_masking": {
            "latent_mask_pct": float(latent_mask_pct),
            "latent_mask_seed": latent_mask_seed,
            "explicit_mask_file_entries": int(len(explicit_mask_map)),
            "mask_alarm_feedback": bool(mask_alarm_feedback),
        },
        "runs": run_results,
    }

    agg_points = _aggregate_point_metrics(point_metric_rows)
    if agg_points is not None:
        summary["aggregate_label_metrics"] = agg_points

    if overlap_rows:
        summary["aggregate_time_overlap_metrics"] = {
            "positive_windows": int(sum(o.positive_windows for o in overlap_rows)),
            "positive_seconds": float(sum(o.positive_seconds for o in overlap_rows)),
            "alarms_total": int(sum(o.alarms_total for o in overlap_rows)),
            "alarms_hit_positive": int(sum(o.alarms_hit_positive for o in overlap_rows)),
            "detected_positive_windows": int(sum(o.detected_positive_windows for o in overlap_rows)),
            "detected_positive_seconds": float(sum(o.detected_positive_seconds for o in overlap_rows)),
            "alarm_hit_rate": float(np.mean([o.alarm_hit_rate for o in overlap_rows])),
            "window_coverage_rate": float(np.mean([o.window_coverage_rate for o in overlap_rows])),
            "time_coverage_rate": float(np.mean([o.time_coverage_rate for o in overlap_rows])),
        }

    return summary


def main() -> None:
    args = parse_args()

    if not (0.0 <= float(args.latent_mask_pct) <= 1.0):
        raise ValueError(f"--latent-mask-pct must be in [0, 1], got {args.latent_mask_pct}")

    wandb = _maybe_import_wandb(args.wandb)
    wb_run = None
    if wandb is not None:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=tags,
            mode=args.wandb_mode,
            job_type="eval",
            config=_build_eval_wandb_config(args),
        )

    model_paths = _discover_model_paths(args)
    eval_dir = Path(args.eval_dir).resolve()
    eval_files = _discover_run_files(eval_dir, max_runs=args.max_runs)
    eval_runs = [_load_run_data(p) for p in eval_files]

    h5_path = Path(args.h5_path).resolve() if args.h5_path else None
    if h5_path is not None and not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    explicit_mask_path = Path(args.latent_mask_file).resolve() if args.latent_mask_file else None
    if explicit_mask_path is not None and not explicit_mask_path.exists():
        raise FileNotFoundError(f"Latent mask file not found: {explicit_mask_path}")
    explicit_mask_map = _load_explicit_mask_map(explicit_mask_path)

    threshold_per_model: Dict[Path, float] = {}

    if args.threshold is not None:
        for model_path in model_paths:
            threshold_per_model[model_path] = float(args.threshold)
    else:
        if not args.calibration_dir:
            raise ValueError(
                "Provide --threshold or --calibration-dir for FAR calibration"
            )

        cal_dir = Path(args.calibration_dir).resolve()
        cal_files = _discover_run_files(cal_dir, max_runs=args.max_calibration_runs)
        cal_runs = [_load_run_data(p) for p in cal_files]
        cal_ts = _build_concatenated_time_series(cal_runs)

        for model_path in model_paths:
            detector = LSTMTemporalDetector(verbose=not args.quiet)
            detector.load(str(model_path))
            threshold = detector.set_threshold_by_far(
                cal_ts,
                alarms_per_hour=float(args.alarms_per_hour),
                verbose=not args.quiet,
                latent_mask_pct=float(args.latent_mask_pct),
                mask_seed=args.latent_mask_seed,
                mask_alarm_feedback=bool(args.mask_alarm_feedback),
            )
            threshold_per_model[model_path] = float(threshold)
            if not args.quiet:
                print(
                    f"[calibration] {model_path.name}: threshold={threshold:.6f} "
                    f"at target FAR={args.alarms_per_hour:.3f}/hr"
                )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, object]] = []
    calibration_table = None
    if wb_run is not None:
        calibration_table = wandb.Table(columns=["model", "threshold"])

    for model_path in model_paths:
        if not args.quiet:
            print(f"[evaluate] {model_path.name}")

        result = evaluate_model(
            model_path=model_path,
            eval_runs=eval_runs,
            threshold=threshold_per_model[model_path],
            h5_path=h5_path,
            time_units_override=args.time_units,
            quiet=args.quiet,
            latent_mask_pct=float(args.latent_mask_pct),
            latent_mask_seed=args.latent_mask_seed,
            explicit_mask_map=explicit_mask_map,
            mask_alarm_feedback=bool(args.mask_alarm_feedback),
        )

        all_results.append(result)

        if wb_run is not None:
            flat: Dict[str, Any] = {
                "model/name": model_path.name,
                "model/threshold": float(result["threshold"]),
                "eval/n_runs": int(result["n_runs"]),
                "eval/total_spectra": int(result["total_spectra"]),
            }
            score_stats = result.get("score_stats", {})
            if isinstance(score_stats, dict):
                _flatten_dict("eval/score_stats", score_stats, flat)

            agg_label = result.get("aggregate_label_metrics", {})
            if isinstance(agg_label, dict):
                _flatten_dict("eval/aggregate_label_metrics", agg_label, flat)

            agg_overlap = result.get("aggregate_time_overlap_metrics", {})
            if isinstance(agg_overlap, dict):
                _flatten_dict("eval/aggregate_time_overlap_metrics", agg_overlap, flat)

            wandb.log(flat)

            runs_table = wandb.Table(
                columns=[
                    "model",
                    "run_name",
                    "n_spectra",
                    "threshold_crossings",
                    "n_alarms",
                    "precision",
                    "recall",
                    "f1",
                ]
            )
            for row in result.get("runs", []):
                if not isinstance(row, dict):
                    continue
                alarm_summary = row.get("alarm_summary", {})
                label_metrics = row.get("label_metrics", {})
                runs_table.add_data(
                    model_path.name,
                    row.get("run_name"),
                    row.get("n_spectra"),
                    row.get("threshold_crossings"),
                    alarm_summary.get("n_alarms") if isinstance(alarm_summary, dict) else None,
                    label_metrics.get("precision") if isinstance(label_metrics, dict) else None,
                    label_metrics.get("recall") if isinstance(label_metrics, dict) else None,
                    label_metrics.get("f1") if isinstance(label_metrics, dict) else None,
                )
            wandb.log({f"eval/runs/{model_path.stem}": runs_table})

            if calibration_table is not None:
                calibration_table.add_data(model_path.name, float(result["threshold"]))

        out_file = output_dir / f"{model_path.stem}.evaluation.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(_to_builtin(result), f, indent=2)

        print(f"Wrote {out_file}")

    summary = {
        "models": [str(p) for p in model_paths],
        "eval_dir": str(eval_dir),
        "calibration_dir": str(Path(args.calibration_dir).resolve()) if args.calibration_dir else None,
        "max_calibration_runs": args.max_calibration_runs,
        "threshold_input": args.threshold,
        "alarms_per_hour": args.alarms_per_hour,
        "h5_path": str(h5_path) if h5_path else None,
        "latent_mask_pct": float(args.latent_mask_pct),
        "latent_mask_seed": args.latent_mask_seed,
        "latent_mask_file": str(explicit_mask_path) if explicit_mask_path else None,
        "mask_alarm_feedback": bool(args.mask_alarm_feedback),
        "results": all_results,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_to_builtin(summary), f, indent=2)

    print(f"Wrote {summary_path}")

    if wb_run is not None:
        if calibration_table is not None:
            wandb.log({"eval/calibrated_thresholds": calibration_table})

        eval_artifact = wandb.Artifact(
            name=f"lstm-temporal-eval-{output_dir.name}",
            type="evaluation",
            metadata={
                "n_models": len(model_paths),
                "eval_dir": str(eval_dir),
                "calibration_dir": str(Path(args.calibration_dir).resolve()) if args.calibration_dir else None,
                "h5_path": str(h5_path) if h5_path else None,
            },
        )
        eval_artifact.add_file(str(summary_path))
        for model_path in model_paths:
            eval_file = output_dir / f"{model_path.stem}.evaluation.json"
            if eval_file.exists():
                eval_artifact.add_file(str(eval_file))
        wandb.log_artifact(eval_artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
