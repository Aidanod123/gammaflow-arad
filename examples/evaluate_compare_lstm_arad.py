#!/usr/bin/env python3
"""Compare LSTM and ARAD on one run with per-spectrum anomaly labels.

This script scores a single preprocessed run with both models, exports a
per-spectrum table of anomaly labels and scores, and saves random visualization
comparisons for selected spectra.

Outputs
-------
- summary.json: run/model metadata and score statistics
- scores.csv: per-spectrum scores and anomaly labels
- visuals/: random reconstruction plots for LSTM, ARAD, and the combined view
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import h5py
except ImportError:
    h5py = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gammaflow import SpectralTimeSeries, Spectrum
from gammaflow.algorithms.arad import ARADDetector
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

    ts = SpectralTimeSeries.from_array(
        counts,
        energy_edges=energy_edges,
        timestamps=timestamps,
        live_times=live_times,
        real_times=real_times,
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
    return value


def _spectrum_time(spec: Spectrum) -> float:
    if spec.live_time is not None and not np.isnan(spec.live_time):
        return float(spec.live_time)
    return float(spec.real_time)


def _recon_to_target_scale_lstm(
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


def _score_and_reconstruct_lstm_at_index(
    detector: LSTMTemporalDetector,
    counts: np.ndarray,
    idx: int,
) -> Optional[Tuple[float, np.ndarray]]:
    window = detector._build_window_for_index(counts.astype(np.float32, copy=False), idx)
    if window is None:
        return None

    with torch.no_grad():
        window_tensor = torch.from_numpy(window).unsqueeze(0).to(detector.device)
        target_tensor = torch.from_numpy(counts[idx].astype(np.float32)).unsqueeze(0).to(detector.device)
        recon_raw = detector.model_(window_tensor)
        score_t = detector._score_batch(target_tensor, recon_raw)

    recon_raw_np = recon_raw.detach().cpu().numpy().squeeze(0)
    target_np = counts[idx].astype(np.float32, copy=False)
    recon_scaled = _recon_to_target_scale_lstm(detector, target_np, recon_raw_np)
    return float(score_t.item()), recon_scaled


def _resolve_run_file(eval_dir: Path, run_file: Optional[str], seed: int) -> Path:
    if run_file:
        path = Path(run_file)
        if not path.is_absolute():
            path = (eval_dir.parent / path).resolve() if not path.exists() else path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Run file not found: {path}")
        return path

    run_files = sorted(eval_dir.glob("run*.pt"), key=lambda p: int(p.stem.replace("run", "")))
    if not run_files:
        raise FileNotFoundError(f"No run*.pt files found in {eval_dir}")

    rng = np.random.default_rng(int(seed))
    return run_files[int(rng.integers(0, len(run_files)))]


def _select_random_indices(
    valid_indices: np.ndarray,
    count: int,
    seed: int,
) -> List[int]:
    if valid_indices.size == 0:
        return []
    rng = np.random.default_rng(int(seed))
    count = min(int(count), int(valid_indices.size))
    if count <= 0:
        return []
    selected = rng.choice(valid_indices, size=count, replace=False)
    return [int(i) for i in np.sort(selected)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a LSTM temporal model with an ARAD model on one run"
    )
    parser.add_argument(
        "--lstm-model-path",
        type=Path,
        default=Path("models/4-1-lstm_no_att_chi2-2layer-20-256b-v2_2.0-0.5.pt"),
        help="Path to the LSTM temporal checkpoint",
    )
    parser.add_argument(
        "--arad-model-path",
        type=Path,
        default=Path("models/arad_chi2_2.0-1.0-256.pt"),
        help="Path to the ARAD checkpoint",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("preprocessed-data/ACTUAL-TESTING-SET-2.0-1.0-b256"),
        help="Directory containing run*.pt files",
    )
    parser.add_argument(
        "--run-file",
        type=str,
        default=None,
        help="Optional explicit run*.pt file. If omitted, one is chosen at random from --eval-dir.",
    )
    parser.add_argument(
        "--run-seed",
        type=int,
        default=42,
        help="Seed used when choosing a random run file",
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=Path("RADAI/training_v4.3.h5"),
        help="Optional H5 path for anomaly labels",
    )
    parser.add_argument(
        "--time-units",
        type=str,
        default=None,
        choices=["us", "ms", "s"],
        help="Override time units when reading dt from H5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for summaries and visualizations",
    )
    parser.add_argument(
        "--num-random-viz",
        type=int,
        default=6,
        help="How many random spectra to visualize",
    )
    parser.add_argument(
        "--viz-seed",
        type=int,
        default=7,
        help="Seed used to sample visualization indices",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a log y-axis for reconstruction plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for PyTorch models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eval_dir = args.eval_dir.resolve()
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    run_path = _resolve_run_file(eval_dir, args.run_file, int(args.run_seed))
    run_payload, ts = _load_run(run_path)
    run_id = _run_id_from_name(run_path)

    h5_path = args.h5_path.resolve() if args.h5_path is not None else None
    if h5_path is not None and not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    lstm = LSTMTemporalDetector(device=args.device, verbose=False)
    lstm.load(str(args.lstm_model_path.resolve()))

    arad = ARADDetector(device=args.device, verbose=False)
    arad.load(str(args.arad_model_path.resolve()))

    counts = np.asarray(ts.counts, dtype=np.float32)
    lstm_scores = lstm.score_time_series(ts)
    arad_scores = np.asarray([arad.score_spectrum(ts[i]) for i in range(ts.n_spectra)], dtype=np.float64)

    labels: Optional[np.ndarray] = None
    if h5_path is not None and run_id is not None:
        integration_time = run_payload.get("integration_time")
        if integration_time is None:
            integration_time = float(np.median(ts.real_times))
        units = args.time_units or str(run_payload.get("time_units", "us"))
        labels = _source_window_labels_from_h5(
            h5_path=h5_path,
            run_id=run_id,
            timestamps=np.asarray(ts.timestamps, dtype=np.float64),
            integration_time=float(integration_time),
            time_units=units,
        )

    valid_indices = np.where(np.isfinite(lstm_scores) & np.isfinite(arad_scores))[0]
    selected_indices = _select_random_indices(valid_indices, int(args.num_random_viz), int(args.viz_seed))

    output_dir = args.output_dir.resolve()
    combined_dir = output_dir / "visuals" / "combined"
    lstm_dir = output_dir / "visuals" / "lstm"
    arad_dir = output_dir / "visuals" / "arad"
    combined_dir.mkdir(parents=True, exist_ok=True)
    lstm_dir.mkdir(parents=True, exist_ok=True)
    arad_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for i in range(ts.n_spectra):
        is_anomaly = bool(labels[i]) if labels is not None else None
        rows.append(
            {
                "spectrum_index": int(i),
                "timestamp": float(ts.timestamps[i]),
                "real_time": float(ts.real_times[i]),
                "is_anomaly": is_anomaly,
                "lstm_score": None if not np.isfinite(lstm_scores[i]) else float(lstm_scores[i]),
                "arad_score": None if not np.isfinite(arad_scores[i]) else float(arad_scores[i]),
            }
        )

    score_table_path = output_dir / "scores.csv"
    with score_table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["spectrum_index", "timestamp", "real_time", "is_anomaly", "lstm_score", "arad_score"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Summary statistics.
    valid_lstm = np.asarray(lstm_scores, dtype=np.float64)[np.isfinite(lstm_scores)]
    valid_arad = np.asarray(arad_scores, dtype=np.float64)[np.isfinite(arad_scores)]
    summary: Dict[str, Any] = {
        "run_file": str(run_path.resolve()),
        "run_name": run_path.name,
        "run_id": run_id,
        "n_spectra": int(ts.n_spectra),
        "n_bins": int(ts.n_bins),
        "lstm_model_path": str(args.lstm_model_path.resolve()),
        "arad_model_path": str(args.arad_model_path.resolve()),
        "h5_path": str(h5_path) if h5_path is not None else None,
        "time_units": args.time_units,
        "selected_random_indices": selected_indices,
        "score_table_path": str(score_table_path.resolve()),
        "score_stats": {
            "lstm": _safe_stats(np.asarray(lstm_scores, dtype=np.float64)),
            "arad": _safe_stats(np.asarray(arad_scores, dtype=np.float64)),
        },
        "rows": rows,
        "visual_examples": [],
    }

    if labels is not None:
        anomaly_lstm = np.asarray(lstm_scores, dtype=np.float64)[np.isfinite(lstm_scores) & labels]
        normal_lstm = np.asarray(lstm_scores, dtype=np.float64)[np.isfinite(lstm_scores) & (~labels)]
        anomaly_arad = np.asarray(arad_scores, dtype=np.float64)[np.isfinite(arad_scores) & labels]
        normal_arad = np.asarray(arad_scores, dtype=np.float64)[np.isfinite(arad_scores) & (~labels)]
        summary["ground_truth_score_stats"] = {
            "n_anomalous": int(np.sum(labels)),
            "n_normal": int(np.sum(~labels)),
            "lstm": {
                "anomalous": _safe_stats(anomaly_lstm),
                "normal": _safe_stats(normal_lstm),
            },
            "arad": {
                "anomalous": _safe_stats(anomaly_arad),
                "normal": _safe_stats(normal_arad),
            },
        }

    for idx in selected_indices:
        spec = ts[int(idx)]
        label_text = "unknown"
        if labels is not None:
            label_text = "anomaly" if bool(labels[int(idx)]) else "normal"

        lstm_pair = _score_and_reconstruct_lstm_at_index(lstm, counts, int(idx))
        if lstm_pair is None:
            continue
        lstm_score, lstm_recon = lstm_pair

        arad_score = float(arad_scores[int(idx)])
        recon_count_rate = arad.reconstruct(spec)
        t = _spectrum_time(spec)
        arad_recon = recon_count_rate * t

        lstm_recon_spec = Spectrum(
            counts=lstm_recon,
            energy_edges=spec.energy_edges,
            timestamp=spec.timestamp,
            real_time=spec.real_time,
            live_time=spec.live_time,
        )
        arad_recon_spec = Spectrum(
            counts=arad_recon,
            energy_edges=spec.energy_edges,
            timestamp=spec.timestamp,
            real_time=spec.real_time,
            live_time=spec.live_time,
        )

        combined_fig, combined_ax = plot_spectrum_comparison(
            [spec, lstm_recon_spec, arad_recon_spec],
            labels=["Input", "LSTM reconstruction", "ARAD reconstruction"],
            mode="count_rate",
            log_y=bool(args.log_y),
        )
        combined_ax.set_title(
            f"Run {run_path.name} | idx={idx} | label={label_text} | LSTM={lstm_score:.6f} | ARAD={arad_score:.6f}"
        )
        combined_png = combined_dir / f"comparison_idx{idx}.png"
        combined_fig.savefig(combined_png, dpi=160, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(combined_fig)

        lstm_fig, lstm_ax = plot_spectrum_comparison(
            [spec, lstm_recon_spec],
            labels=["Input", "LSTM reconstruction"],
            mode="count_rate",
            log_y=bool(args.log_y),
        )
        lstm_ax.set_title(
            f"LSTM | idx={idx} | label={label_text} | score={lstm_score:.6f}"
        )
        lstm_png = lstm_dir / f"lstm_idx{idx}.png"
        lstm_fig.savefig(lstm_png, dpi=160, bbox_inches="tight")
        plt.close(lstm_fig)

        arad_fig, arad_ax = plot_spectrum_comparison(
            [spec, arad_recon_spec],
            labels=["Input", "ARAD reconstruction"],
            mode="count_rate",
            log_y=bool(args.log_y),
        )
        arad_ax.set_title(
            f"ARAD | idx={idx} | label={label_text} | score={arad_score:.6f}"
        )
        arad_png = arad_dir / f"arad_idx{idx}.png"
        arad_fig.savefig(arad_png, dpi=160, bbox_inches="tight")
        plt.close(arad_fig)

        summary["visual_examples"].append(
            {
                "index": int(idx),
                "label": label_text,
                "lstm_score": float(lstm_score),
                "arad_score": float(arad_score),
                "combined_plot": str(combined_png.resolve()),
                "lstm_plot": str(lstm_png.resolve()),
                "arad_plot": str(arad_png.resolve()),
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_to_builtin(summary), indent=2))

    print(f"Run: {run_path}")
    print(f"Wrote per-spectrum scores to {score_table_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote {len(summary['visual_examples'])} random visualizations to {output_dir / 'visuals'}")


if __name__ == "__main__":
    main()
