#!/usr/bin/env python3
"""Plot reconstruction error and gross count rate for high-SNR testing runs.

This script selects top runs by a source-event SNR proxy from an H5 testing file,
then generates one figure per run with:
- Top panel: reconstruction error (model score) vs time
- Bottom panel: gross count rate vs time

Example usage
-------------
LSTM:
    /home/aodonnell/urban-data-venv/venv/bin/python examples/plot_high_snr_error_and_rate.py \
      --model-type lstm \
      --model-path models/4-1-lstm_no_att_chi2-2layer-20-256b-v2_2.0-0.5.pt \
      --preprocessed-dir preprocessed-data/with-sources-2.0-1.0-b256 \
      --h5-path /home/aodonnell/Desktop/urbandata-gammaflow/RADAI-dataset/testing_v4.3.h5 \
      --output-dir eval-results/high-snr-lstm \
      --num-runs 4 --device cuda

ARAD:
    /home/aodonnell/urban-data-venv/venv/bin/python examples/plot_high_snr_error_and_rate.py \
      --model-type arad \
      --model-path models/arad_chi2_2.0-1.0-256.pt \
      --preprocessed-dir preprocessed-data/with-sources-2.0-1.0-b256 \
      --h5-path /home/aodonnell/Desktop/urbandata-gammaflow/RADAI-dataset/testing_v4.3.h5 \
      --output-dir eval-results/high-snr-arad \
      --num-runs 4 --device cuda
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


@dataclass
class RunSNR:
    run_id: int
    run_name: str
    source_events: int
    total_events: int
    source_fraction: float
    peak_to_median_rate: float
    std_to_mean_rate: float


def _load_val_runs_from_metrics(metrics_path: Path) -> List[str]:
    payload = json.loads(metrics_path.read_text())
    val_runs = payload.get("val_runs")
    if not isinstance(val_runs, list) or not val_runs:
        raise ValueError(
            f"Metrics file '{metrics_path}' does not contain a non-empty 'val_runs' list"
        )
    out: List[str] = []
    for item in val_runs:
        name = str(item).strip()
        if name:
            out.append(Path(name).name)
    if not out:
        raise ValueError(f"Metrics file '{metrics_path}' has invalid 'val_runs' entries")
    return out


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


def _discover_runs(preprocessed_dir: Path) -> List[Path]:
    run_files = sorted(
        preprocessed_dir.glob("run*.pt"), key=lambda p: int(p.stem.replace("run", ""))
    )
    if not run_files:
        raise FileNotFoundError(f"No run*.pt files found in {preprocessed_dir}")
    return run_files


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
                source_group = None
                if "source" in run_group:
                    source_group = run_group["source"]
                elif "sources" in run_group:
                    source_group = run_group["sources"]
                if source_group is not None and "id" in source_group:
                    source_ids = np.asarray(source_group["id"])
                    if source_ids.size > 0:
                        source_events = int(np.isin(event_ids, source_ids).sum())

            # Build a robust fallback SNR proxy from listmode event density over time.
            # This remains available even when the H5 has no explicit source labels.
            scale = _time_unit_scale("us")
            if dt.size > 0:
                abs_times = np.cumsum(np.asarray(dt, dtype=np.float64) * scale)
                # Estimate gross-rate profile with fixed-width windows and summarize contrast.
                n_windows = min(512, max(64, int(np.sqrt(abs_times.size))))
                centers = np.linspace(abs_times[0], abs_times[-1], num=n_windows)
                width = max((abs_times[-1] - abs_times[0]) / max(n_windows, 1), 1e-6)
                half = width / 2.0
                counts = np.empty(n_windows, dtype=np.float64)
                for i, c in enumerate(centers):
                    left = np.searchsorted(abs_times, c - half, side="left")
                    right = np.searchsorted(abs_times, c + half, side="left")
                    counts[i] = right - left
                rates = counts / width
                median_rate = float(np.median(rates)) if rates.size else 0.0
                p95_rate = float(np.percentile(rates, 95.0)) if rates.size else 0.0
                mean_rate = float(np.mean(rates)) if rates.size else 0.0
                std_rate = float(np.std(rates)) if rates.size else 0.0
                peak_to_median = p95_rate / median_rate if median_rate > 0 else 0.0
                std_to_mean = std_rate / mean_rate if mean_rate > 0 else 0.0
            else:
                peak_to_median = 0.0
                std_to_mean = 0.0

            frac = (source_events / total_events) if total_events > 0 else 0.0
            rows.append(
                RunSNR(
                    run_id=int(run_id),
                    run_name=f"run{run_id}.pt",
                    source_events=source_events,
                    total_events=total_events,
                    source_fraction=float(frac),
                    peak_to_median_rate=float(peak_to_median),
                    std_to_mean_rate=float(std_to_mean),
                )
            )
    return rows


def _gross_count_rate_from_h5(
    h5_path: Path,
    run_id: int,
    timestamps: np.ndarray,
    integration_time: float,
    time_units: str,
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
        t0 = center - half
        t1 = center + half
        left = np.searchsorted(abs_times, t0, side="left")
        right = np.searchsorted(abs_times, t1, side="left")
        n = right - left
        rates[i] = n / integration_time if integration_time > 0 else 0.0

    return rates


def _source_times_from_h5(h5_path: Path, run_id: int) -> np.ndarray:
    """Return source injection times in seconds.

    ``sources/time`` is stored in milliseconds in the H5 file.  Each entry
    is the absolute time (within the run) at which a source injection began.
    """
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
    return times_ms / 1000.0  # ms → s


def _score_run_lstm(
    model_path: Path,
    ts: SpectralTimeSeries,
    device: str,
    latent_mask_pct: float,
    mask_seed: Optional[int],
    target_count_rates: Optional[np.ndarray],
    normalize_inputs_l1: bool,
    score_type: Optional[str] = None,
) -> np.ndarray:
    detector = LSTMTemporalDetector(device=device, verbose=False)
    detector.load(str(model_path))
    if detector.count_rate_conditioning and target_count_rates is None:
        print(
            "WARNING: model uses count-rate conditioning (CRC) but no "
            "target_count_rates found in run payload — decoder will receive "
            "zero conditioning, scores will be unreliable."
        )
    return detector.score_time_series(
        ts,
        target_count_rates=target_count_rates,
        normalize_inputs_l1=bool(normalize_inputs_l1),
        latent_mask_pct=float(latent_mask_pct),
        mask_seed=mask_seed,
        score_type=score_type,
    )


def _score_run_arad(model_path: Path, ts: SpectralTimeSeries, device: str) -> np.ndarray:
    detector = ARADDetector(device=device, verbose=False)
    detector.load(str(model_path))
    return detector.score_time_series(ts)


def _suppress_low_count_tail_artifact(
    scores: np.ndarray,
    target_scales: Optional[np.ndarray],
    min_fraction_of_median: float = 0.05,
    min_scale_floor: float = 50.0,
) -> np.ndarray:
    """Mask trailing low-count windows that create non-physical tail spikes.

    Some runs end with a partial low-count window whose target scale is orders
    of magnitude smaller than the run median. That final point can dominate the
    error plot while not being representative of detector behavior.
    """
    if target_scales is None:
        return scores

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
    masked_any = False
    while idx >= 0 and np.isfinite(scales[idx]) and (scales[idx] < cutoff):
        out[idx] = np.nan
        masked_any = True
        idx -= 1

    if masked_any and idx >= 0:
        # Keep only the contiguous trailing low-count region masked.
        pass

    return out


def _plot_run(
    run_name: str,
    timestamps: np.ndarray,
    scores: np.ndarray,
    gross_rate: np.ndarray,
    snr: RunSNR,
    model_type: str,
    source_times: Optional[np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax0, ax1 = axes
    finite = np.isfinite(scores)
    ax0.plot(timestamps[finite], scores[finite], color="tab:blue", linewidth=1.2)
    ax0.set_ylabel("Reconstruction Error")
    ax0.grid(True, alpha=0.3)
    ax0.set_title(
        f"{model_type.upper()} | {run_name}"
        f"({snr.source_events}/{snr.total_events})"
    )

    ax1.plot(timestamps, gross_rate, color="tab:orange", linewidth=1.2)
    ax1.set_ylabel("Gross Count Rate (counts/s)")
    ax1.set_xlabel("Time (s)")
    ax1.grid(True, alpha=0.3)

    if source_times is not None and len(source_times) > 0:
        for t in source_times:
            ax0.axvline(float(t), color="black", linestyle=":", linewidth=2.0, alpha=0.95)
            ax1.axvline(float(t), color="black", linestyle=":", linewidth=2.0, alpha=0.95)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reconstruction error and gross count rate for high-SNR runs"
    )
    parser.add_argument("--model-type", choices=["lstm", "arad"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--h5-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-runs", type=int, default=4)
    parser.add_argument(
        "--snr-metric",
        choices=["auto", "source_fraction", "source_events", "peak_to_median_rate", "std_to_mean_rate"],
        default="auto",
    )
    parser.add_argument("--time-units", choices=["us", "ms", "s"], default="us")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent-mask-pct", type=float, default=0.0)
    parser.add_argument("--latent-mask-seed", type=int, default=None)
    parser.add_argument(
        "--normalize-inputs-l1",
        action="store_true",
        help="L1-normalize each spectrum at scoring time (leave off for already L1-preprocessed runs)",
    )
    parser.add_argument("--run-ids", nargs="*", type=int, default=None)
    parser.add_argument(
        "--validation-runs-from-metrics",
        type=Path,
        default=None,
        help="Optional metrics JSON with val_runs list; limits plotting to those runs",
    )
    parser.add_argument(
        "--no-source-lines",
        action="store_true",
        help="Disable source-interval dotted lines even when H5 source labels exist",
    )
    parser.add_argument(
        "--score-type",
        type=str,
        default=None,
        choices=["jsd", "corrected_jsd", "chi2", "normalized_chi2", "reduced_chi2", "combined", "poisson"],
        help="Override scoring metric (default: use model's loss_type). "
             "'jsd': raw JSD; "
             "'corrected_jsd': JSD minus Poisson noise floor (rate-decoupled); "
             "'chi2': raw Pearson chi-squared; "
             "'normalized_chi2': chi2/total_counts; "
             "'reduced_chi2': (chi2 - n_bins)/n_bins (rate-decoupled); "
             "'combined': JSD + reduced_chi2, flat and count-weighted shape error together; "
             "'poisson': N-normalized Poisson NLL = KL(target || recon), rate-invariant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preprocessed_dir = args.preprocessed_dir.resolve()
    h5_path = args.h5_path.resolve()
    output_dir = args.output_dir.resolve()

    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed dir not found: {preprocessed_dir}")
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    run_files = _discover_runs(preprocessed_dir)
    file_by_id: Dict[int, Path] = {}
    for p in run_files:
        run_id = _run_id_from_name(p)
        if run_id is not None:
            file_by_id[run_id] = p

    candidate_ids = sorted(file_by_id.keys())

    if args.validation_runs_from_metrics is not None:
        metrics_path = args.validation_runs_from_metrics.resolve()
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        val_run_names = set(_load_val_runs_from_metrics(metrics_path))
        candidate_ids = [rid for rid in candidate_ids if f"run{rid}.pt" in val_run_names]

    if args.run_ids:
        requested = set(int(x) for x in args.run_ids)
        candidate_ids = [rid for rid in candidate_ids if rid in requested]

    snr_rows = _compute_snr_rows(h5_path=h5_path, run_ids=candidate_ids)
    if not snr_rows:
        raise RuntimeError("No overlapping runs found between preprocessed dir and H5")

    metric_used = str(args.snr_metric)
    if args.snr_metric == "source_events":
        snr_rows.sort(key=lambda r: (r.source_events, r.source_fraction), reverse=True)
    elif args.snr_metric == "source_fraction":
        snr_rows.sort(key=lambda r: (r.source_fraction, r.source_events), reverse=True)
    elif args.snr_metric == "peak_to_median_rate":
        snr_rows.sort(key=lambda r: (r.peak_to_median_rate, r.std_to_mean_rate), reverse=True)
    elif args.snr_metric == "std_to_mean_rate":
        snr_rows.sort(key=lambda r: (r.std_to_mean_rate, r.peak_to_median_rate), reverse=True)
    else:
        has_source_labels = any((r.source_events > 0 or r.source_fraction > 0.0) for r in snr_rows)
        if has_source_labels:
            metric_used = "source_fraction"
            snr_rows.sort(key=lambda r: (r.source_fraction, r.source_events), reverse=True)
        else:
            metric_used = "peak_to_median_rate"
            snr_rows.sort(key=lambda r: (r.peak_to_median_rate, r.std_to_mean_rate), reverse=True)

    selected = snr_rows[: max(1, int(args.num_runs))]

    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    for row in selected:
        run_path = file_by_id[row.run_id]
        payload, ts = _load_run(run_path)

        if args.model_type == "lstm":
            target_count_rates = payload.get("count_rates")
            if target_count_rates is not None and torch.is_tensor(target_count_rates):
                target_count_rates = target_count_rates.detach().cpu().numpy()
            elif target_count_rates is not None:
                target_count_rates = np.asarray(target_count_rates, dtype=np.float32)
            if target_count_rates is not None:
                live_times = payload.get("live_times")
                if live_times is not None:
                    if torch.is_tensor(live_times):
                        live_times = live_times.detach().cpu().numpy()
                    else:
                        live_times = np.asarray(live_times, dtype=np.float32)
                    target_count_rates = target_count_rates * live_times

            scores = _score_run_lstm(
                model_path=args.model_path.resolve(),
                ts=ts,
                device=args.device,
                latent_mask_pct=float(args.latent_mask_pct),
                mask_seed=args.latent_mask_seed,
                target_count_rates=target_count_rates,
                normalize_inputs_l1=bool(args.normalize_inputs_l1),
                score_type=args.score_type,
            )
            scores = _suppress_low_count_tail_artifact(scores, target_count_rates)
        else:
            scores = _score_run_arad(
                model_path=args.model_path.resolve(),
                ts=ts,
                device=args.device,
            )

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

        out_png = output_dir / f"{args.model_type}_run{row.run_id}_error_and_gross_rate.png"

        source_times: Optional[np.ndarray] = None
        if not bool(args.no_source_lines):
            source_times = _source_times_from_h5(h5_path=h5_path, run_id=row.run_id)

        _plot_run(
            run_name=run_path.name,
            timestamps=np.asarray(ts.timestamps, dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float64),
            gross_rate=gross_rate,
            snr=row,
            model_type=args.model_type,
            source_times=source_times,
            output_path=out_png,
        )

        results.append(
            {
                "run_id": row.run_id,
                "run_name": run_path.name,
                "source_fraction": row.source_fraction,
                "source_events": row.source_events,
                "total_events": row.total_events,
                "peak_to_median_rate": row.peak_to_median_rate,
                "std_to_mean_rate": row.std_to_mean_rate,
                "n_source_times": 0 if source_times is None else len(source_times),
                "plot_path": str(out_png.resolve()),
            }
        )
        print(f"Wrote {out_png}")

    summary = {
        "model_type": args.model_type,
        "model_path": str(args.model_path.resolve()),
        "preprocessed_dir": str(preprocessed_dir),
        "h5_path": str(h5_path),
        "snr_metric": args.snr_metric,
        "snr_metric_used": metric_used,
        "validation_runs_from_metrics": (
            str(args.validation_runs_from_metrics.resolve())
            if args.validation_runs_from_metrics is not None
            else None
        ),
        "source_lines_enabled": not bool(args.no_source_lines),
        "normalize_inputs_l1": bool(args.normalize_inputs_l1),
        "num_runs": len(results),
        "runs": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
