"""Evaluate trained ARADDetector models on preprocessed run tensors.

This script supports:
1) Threshold calibration on clean background runs (target FAR), and
2) Evaluation on held-out/test runs.

Examples:
    # Evaluate a single ARAD model with FAR-calibrated threshold
    python examples/evaluate_arad_models.py \
      --models models/arad_chi2_2.0-1.0-256.pt \
      --calibration-dir preprocessed-data/no-sources-2.0-1.0-b256 \
      --eval-dir preprocessed-data/ACTUAL-TESTING-SET-2.0-1.0-b256 \
      --alarms-per-hour 1.0 \
      --output-dir eval-results/arad-eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    import h5py
except ImportError:
    h5py = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gammaflow.algorithms.arad import ARADDetector
from gammaflow.core.time_series import SpectralTimeSeries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained ARADDetector models on preprocessed runs"
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
        help="Directory of evaluation run*.pt files",
    )
    parser.add_argument(
        "--h5-path",
        type=str,
        default=None,
        help="Optional RADAI HDF5 path for ground-truth-aware score exports",
    )
    parser.add_argument(
        "--time-units",
        type=str,
        default=None,
        choices=["us", "ms", "s"],
        help="Override time units when reading dt from H5 (default: infer from run payload)",
    )
    parser.add_argument(
        "--ground-truth-score-run",
        type=str,
        default="random",
        help=(
            "Run selector used for detailed GT score export when --h5-path is provided. "
            "Use 'random' (default), or run file/name/index (e.g., run0.pt, run0, 0)."
        ),
    )
    parser.add_argument(
        "--ground-truth-score-seed",
        type=int,
        default=42,
        help="Seed used when --ground-truth-score-run=random",
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
        "--calibration-use-val-runs",
        action="store_true",
        help=(
            "Calibrate thresholds using validation runs listed in each model's "
            "<model>.metrics.json (val_runs) instead of all runs in --calibration-dir."
        ),
    )
    parser.add_argument(
        "--calibration-metrics-path",
        type=str,
        default=None,
        help=(
            "Optional explicit metrics JSON path containing val_runs for calibration. "
            "If provided with multiple models, the same val_runs list is applied to all."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where evaluation JSON files are written",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true", help="Reduce per-run logging")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="gammaflow-arad")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="",
        help="Comma-separated W&B tags (example: eval,arad)",
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


def _discover_model_paths(args: argparse.Namespace) -> List[Path]:
    model_paths: List[Path] = []

    if args.models:
        model_paths.extend(Path(p) for p in args.models)

    if args.model_glob:
        model_paths.extend(sorted(Path().glob(args.model_glob)))

    # De-duplicate while preserving order
    unique: Dict[str, Path] = {}
    for p in model_paths:
        unique[str(p.resolve())] = p

    resolved = [Path(k) for k in unique.keys()]
    if not resolved:
        raise ValueError("No model checkpoints resolved. Use --models and/or --model-glob.")

    for p in resolved:
        if not p.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {p}")

    return resolved


def _discover_run_files(root: Path, max_runs: Optional[int]) -> List[Path]:
    run_files = sorted(root.glob("run*.pt"), key=lambda p: int(p.stem.replace("run", "")))
    if not run_files:
        raise FileNotFoundError(f"No run*.pt files found in {root}")
    if max_runs is not None:
        run_files = run_files[: max_runs]
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


def _load_run_as_timeseries(path: Path) -> SpectralTimeSeries:
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


def _load_run_payload(path: Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "spectra" not in obj:
        raise ValueError(f"Expected dict with 'spectra' in {path}")
    return obj


def _run_id_from_name(path: Path) -> Optional[int]:
    stem = path.stem
    if not stem.startswith("run"):
        return None
    try:
        return int(stem.replace("run", ""))
    except ValueError:
        return None


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


def _resolve_gt_score_run(eval_files: Sequence[Path], selector: Optional[str]) -> Path:
    if not eval_files:
        raise ValueError("No eval files available to resolve ground-truth score run")

    if selector is None or not selector.strip():
        selector = "random"

    if selector.strip().lower() == "random":
        raise ValueError("Internal: random selection requires _resolve_gt_score_run_with_seed")

    s = selector.strip()
    normalized = s if s.endswith(".pt") else f"{s}.pt" if s.startswith("run") else s

    for p in eval_files:
        if p.name == normalized or p.stem == s:
            return p

    if s.isdigit():
        target_name = f"run{int(s)}.pt"
        for p in eval_files:
            if p.name == target_name:
                return p

    available = ", ".join(p.name for p in eval_files[:8])
    raise ValueError(
        f"Could not resolve --ground-truth-score-run='{selector}'. "
        f"Available examples: {available}{' ...' if len(eval_files) > 8 else ''}"
    )


def _resolve_gt_score_run_with_seed(
    eval_files: Sequence[Path],
    selector: Optional[str],
    seed: int,
) -> Path:
    if not eval_files:
        raise ValueError("No eval files available to resolve ground-truth score run")

    s = (selector or "random").strip().lower()
    if s == "random":
        rng = np.random.default_rng(int(seed))
        idx = int(rng.integers(0, len(eval_files)))
        return eval_files[idx]

    return _resolve_gt_score_run(eval_files, selector)


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


def _concat_runs_as_timeseries(run_files: Sequence[Path]) -> SpectralTimeSeries:
    counts_parts: List[np.ndarray] = []
    timestamps_parts: List[np.ndarray] = []
    real_times_parts: List[np.ndarray] = []
    live_times_parts: List[np.ndarray] = []
    has_any_live = False
    shared_energy_edges = None
    t_offset = 0.0

    for idx, path in enumerate(run_files):
        ts = _load_run_as_timeseries(path)

        counts_parts.append(ts.counts)
        timestamps_parts.append(ts.timestamps + t_offset)
        real_times_parts.append(ts.real_times)

        if ts.live_times is not None and ts.live_times.dtype != object:
            has_any_live = True
            live_times_parts.append(ts.live_times)

        if ts.energy_edges is not None:
            if shared_energy_edges is None:
                shared_energy_edges = ts.energy_edges
            elif not np.array_equal(shared_energy_edges, ts.energy_edges):
                raise ValueError(f"Energy edges mismatch between runs; first mismatch at {path}")

        if idx < (len(run_files) - 1):
            t_offset = float(timestamps_parts[-1][-1]) + 1e-6

    all_counts = np.vstack(counts_parts)
    all_timestamps = np.concatenate(timestamps_parts)
    all_real_times = np.concatenate(real_times_parts)
    all_live_times = np.concatenate(live_times_parts) if has_any_live and live_times_parts else None

    return SpectralTimeSeries.from_array(
        all_counts,
        energy_edges=shared_energy_edges,
        timestamps=all_timestamps,
        live_times=all_live_times,
        real_times=all_real_times,
    )


def _load_val_runs_from_metrics(metrics_path: Path) -> List[str]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    payload = json.loads(metrics_path.read_text())
    val_runs = payload.get("val_runs")
    if not isinstance(val_runs, list) or not val_runs:
        raise ValueError(
            f"Metrics file '{metrics_path}' does not contain a non-empty 'val_runs' list"
        )
    out: List[str] = []
    for item in val_runs:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    if not out:
        raise ValueError(f"Metrics file '{metrics_path}' has an empty/invalid 'val_runs' list")
    return out


def _resolve_calibration_files(
    all_calibration_files: Sequence[Path],
    model_path: Path,
    use_val_runs: bool,
    calibration_metrics_path: Optional[Path],
) -> List[Path]:
    if not use_val_runs:
        return list(all_calibration_files)

    metrics_path = calibration_metrics_path
    if metrics_path is None:
        metrics_path = model_path.with_suffix(".metrics.json")

    val_run_names = _load_val_runs_from_metrics(metrics_path)
    name_to_path = {p.name: p for p in all_calibration_files}

    selected = [name_to_path[name] for name in val_run_names if name in name_to_path]
    if not selected:
        raise ValueError(
            "No calibration files matched metrics val_runs for model "
            f"{model_path}. metrics_path={metrics_path}"
        )

    return selected


def _safe_float(x: float) -> float:
    if x is None or not np.isfinite(x):
        return float("nan")
    return float(x)


def main() -> None:
    args = parse_args()

    model_paths = _discover_model_paths(args)
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_files = _discover_run_files(eval_dir, max_runs=args.max_runs)
    h5_path = Path(args.h5_path).resolve() if args.h5_path else None
    if h5_path is not None and not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    gt_score_run = _resolve_gt_score_run_with_seed(
        eval_files,
        args.ground_truth_score_run,
        int(args.ground_truth_score_seed),
    )

    calibration_files: List[Path] = []
    if args.threshold is None:
        if args.calibration_dir is None:
            raise ValueError("--calibration-dir is required when --threshold is not provided")
        calibration_files = _discover_run_files(
            Path(args.calibration_dir), max_runs=args.max_calibration_runs
        )

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
            config={
                "models": [str(p) for p in model_paths],
                "calibration_dir": args.calibration_dir,
                "eval_dir": args.eval_dir,
                "alarms_per_hour": args.alarms_per_hour,
                "threshold": args.threshold,
                "max_runs": args.max_runs,
                "max_calibration_runs": args.max_calibration_runs,
                "calibration_use_val_runs": args.calibration_use_val_runs,
                "calibration_metrics_path": args.calibration_metrics_path,
                "device": args.device,
                "h5_path": str(h5_path) if h5_path else None,
                "time_units": args.time_units,
                "ground_truth_score_run": gt_score_run.name,
                "ground_truth_score_seed": int(args.ground_truth_score_seed),
            },
        )

    for model_path in model_paths:
        detector = ARADDetector(device=args.device, verbose=not args.quiet)
        detector.load(str(model_path))

        threshold_used: float
        calibration_used_files: List[Path] = []

        if args.threshold is not None:
            threshold_used = float(args.threshold)
            detector.set_threshold(threshold_used)
        else:
            selected_calibration_files = _resolve_calibration_files(
                all_calibration_files=calibration_files,
                model_path=model_path,
                use_val_runs=bool(args.calibration_use_val_runs),
                calibration_metrics_path=(
                    Path(args.calibration_metrics_path)
                    if args.calibration_metrics_path is not None
                    else None
                ),
            )
            calibration_used_files = selected_calibration_files
            calib_ts = _concat_runs_as_timeseries(selected_calibration_files)
            threshold_used = float(
                detector.set_threshold_by_far(
                    background_data=calib_ts,
                    alarms_per_hour=float(args.alarms_per_hour),
                    verbose=not args.quiet,
                )
            )

        per_run: List[Dict[str, Any]] = []
        all_scores: List[np.ndarray] = []
        total_alarm_events = 0
        gt_scores_written_path: Optional[Path] = None

        for run_file in eval_files:
            run_payload = _load_run_payload(run_file)
            ts = _load_run_as_timeseries(run_file)
            scores = detector.process_time_series(ts)
            alarm_summary = detector.get_alarm_summary()

            n_events = int(alarm_summary.get("n_alarms", 0))
            total_alarm_events += n_events
            all_scores.append(scores)

            run_out: Dict[str, Any] = {
                "run": run_file.name,
                "run_id": _run_id_from_name(run_file),
                "n_spectra": int(len(scores)),
                "mean_score": _safe_float(np.mean(scores)),
                "p90_score": _safe_float(np.quantile(scores, 0.90)),
                "p99_score": _safe_float(np.quantile(scores, 0.99)),
                "n_alarm_events": n_events,
                "alarm_summary": alarm_summary,
            }

            if h5_path is not None and run_out["run_id"] is not None:
                integration_time = run_payload.get("integration_time")
                if integration_time is None:
                    integration_time = float(np.median(ts.real_times))
                units = args.time_units or str(run_payload.get("time_units", "us"))
                labels = _source_window_labels_from_h5(
                    h5_path=h5_path,
                    run_id=int(run_out["run_id"]),
                    timestamps=np.asarray(ts.timestamps, dtype=np.float64),
                    integration_time=float(integration_time),
                    time_units=units,
                )
                valid = np.isfinite(scores)
                anomaly_scores = np.asarray(scores)[valid & labels]
                normal_scores = np.asarray(scores)[valid & (~labels)]
                run_out["ground_truth_score_stats"] = {
                    "n_anomalous": int(np.sum(labels)),
                    "n_normal": int(np.sum(~labels)),
                    "anomalous": _safe_stats(anomaly_scores),
                    "normal": _safe_stats(normal_scores),
                }

                if run_file.name == gt_score_run.name:
                    rows: List[Dict[str, Any]] = []
                    score_arr = np.asarray(scores, dtype=float)
                    for i in range(len(score_arr)):
                        v = score_arr[i]
                        rows.append(
                            {
                                "spectrum_index": int(i),
                                "score": None if not np.isfinite(v) else float(v),
                                "is_anomaly": bool(labels[i]),
                            }
                        )

                    gt_dump = {
                        "run": run_file.name,
                        "run_id": run_out["run_id"],
                        "threshold": float(threshold_used),
                        "n_spectra": int(len(scores)),
                        "labels_are_source_present": True,
                        "anomalous_scores": [float(x) for x in anomaly_scores.tolist()],
                        "normal_scores": [float(x) for x in normal_scores.tolist()],
                        "rows": rows,
                    }
                    gt_scores_written_path = (
                        output_dir
                        / model_path.stem
                        / f"{run_file.stem}.ground_truth_scores.json"
                    )
                    gt_scores_written_path.write_text(json.dumps(gt_dump, indent=2))

            per_run.append(run_out)

            if not args.quiet:
                print(
                    f"[{model_path.name}] {run_file.name}: "
                    f"p99={np.quantile(scores, 0.99):.6f} alarms={n_events}"
                )

        merged_scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float64)

        summary = {
            "model": str(model_path),
            "threshold": float(threshold_used),
            "alarms_per_hour_target": float(args.alarms_per_hour),
            "calibration_dir": args.calibration_dir,
            "eval_dir": args.eval_dir,
            "h5_path": str(h5_path) if h5_path else None,
            "ground_truth_score_run": gt_score_run.name,
            "ground_truth_scores_path": (
                str(gt_scores_written_path.relative_to(output_dir))
                if gt_scores_written_path is not None
                else None
            ),
            "calibration_used_files": [p.name for p in calibration_used_files],
            "calibration_use_val_runs": bool(args.calibration_use_val_runs),
            "overall": {
                "n_scores": int(len(merged_scores)),
                "p50": _safe_float(np.quantile(merged_scores, 0.50)) if len(merged_scores) else float("nan"),
                "p90": _safe_float(np.quantile(merged_scores, 0.90)) if len(merged_scores) else float("nan"),
                "p99": _safe_float(np.quantile(merged_scores, 0.99)) if len(merged_scores) else float("nan"),
                "n_alarm_events": int(total_alarm_events),
            },
        }

        model_output_dir = output_dir / model_path.stem
        model_output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = model_output_dir / "summary.json"
        evaluation_path = model_output_dir / f"{model_path.stem}.evaluation.json"

        summary_path.write_text(json.dumps(summary, indent=2))
        evaluation_path.write_text(json.dumps({"per_run": per_run}, indent=2))

        print(f"Wrote {summary_path}")
        print(f"Wrote {evaluation_path}")
        if gt_scores_written_path is not None:
            print(f"Wrote {gt_scores_written_path}")

        if wb_run is not None:
            wandb.log(
                {
                    f"{model_path.stem}/threshold": float(threshold_used),
                    f"{model_path.stem}/overall_p90": summary["overall"]["p90"],
                    f"{model_path.stem}/overall_p99": summary["overall"]["p99"],
                    f"{model_path.stem}/n_alarm_events": int(total_alarm_events),
                }
            )

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()
