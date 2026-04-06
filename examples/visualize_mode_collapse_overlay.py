"""Overlay random reconstructions to inspect possible mode collapse.

Creates two plots for one run file using the same sampled spectrum indices:
1) LSTM temporal reconstructions overlaid
2) ARAD reconstructions overlaid
"""

from __future__ import annotations

import argparse
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

from gammaflow import SpectralTimeSeries, Spectrum
from gammaflow.algorithms.arad import ARADDetector
from gammaflow.algorithms.lstm_temporal import LSTMTemporalDetector


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


def _to_target_scale_lstm(
    detector: LSTMTemporalDetector,
    target: np.ndarray,
    recon_raw: np.ndarray,
    target_count_rate: Optional[float],
) -> np.ndarray:
    eps = 1e-8
    if detector.loss_type == "chi2":
        if target_count_rate is None:
            raise ValueError("target_count_rate is required for chi2 LSTM reconstruction scaling")
        recon_prob = np.maximum(np.asarray(recon_raw, dtype=np.float64), eps)
        recon_prob = recon_prob / np.sum(recon_prob)
        return recon_prob * float(target_count_rate)

    recon_sum = float(np.sum(recon_raw))
    target_sum = float(np.sum(target))
    if recon_sum <= eps:
        return np.zeros_like(recon_raw)
    return (recon_raw / recon_sum) * target_sum


def _reconstruct_lstm_at_index(
    detector: LSTMTemporalDetector,
    counts: np.ndarray,
    idx: int,
    target_count_rates: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    window = detector._build_window_for_index(counts.astype(np.float32, copy=False), idx)
    if window is None:
        return None

    with torch.no_grad():
        window_tensor = torch.from_numpy(window).unsqueeze(0).to(detector.device)
        target_tensor = torch.from_numpy(counts[idx].astype(np.float32)).unsqueeze(0).to(detector.device)
        recon_raw = detector.model_(window_tensor)

    recon_raw_np = recon_raw.detach().cpu().numpy().squeeze(0)
    target_np = target_tensor.detach().cpu().numpy().squeeze(0)
    target_scale = None
    if target_count_rates is not None:
        target_scale = float(target_count_rates[idx])
    return _to_target_scale_lstm(detector, target_np, recon_raw_np, target_scale)


def _spectrum_time(spec: Spectrum) -> float:
    if spec.live_time is not None and not np.isnan(spec.live_time):
        return float(spec.live_time)
    return float(spec.real_time)


def _normalize_curve(y: np.ndarray) -> np.ndarray:
    s = float(np.sum(y))
    if s <= 0:
        return np.zeros_like(y)
    return y / s


def _select_random_indices(
    valid_indices: np.ndarray,
    n: int,
    seed: int,
) -> List[int]:
    if valid_indices.size == 0:
        return []

    rng = np.random.default_rng(int(seed))
    k = min(int(n), int(valid_indices.size))
    if k <= 0:
        return []

    selected = rng.choice(valid_indices, size=k, replace=False)
    return [int(i) for i in np.sort(selected)]


def _overlay_plot(
    curves: Sequence[np.ndarray],
    indices: Sequence[int],
    title: str,
    out_path: Path,
    log_y: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(curves))))
    for i, (idx, curve) in enumerate(zip(indices, curves)):
        if log_y:
            ax.semilogy(curve, color=colors[i % len(colors)], alpha=0.9, linewidth=1.8, label=f"idx={idx}")
        else:
            ax.plot(curve, color=colors[i % len(colors)], alpha=0.9, linewidth=1.8, label=f"idx={idx}")

    ax.set_xlabel("Energy Bin")
    ax.set_ylabel("Reconstruction (normalized area)" if "normalized" in title.lower() else "Reconstruction")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay random LSTM/ARAD reconstructions")
    parser.add_argument("--lstm-model-path", type=Path, required=True)
    parser.add_argument("--arad-model-path", type=Path, required=True)
    parser.add_argument("--run-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-spectra", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each reconstruction to unit area for shape-only comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_payload, ts = _load_run(args.run_file.resolve())
    counts = np.asarray(ts.counts, dtype=np.float32)
    target_count_rates = np.sum(counts, axis=1).astype(np.float32, copy=False)
    rates_value = run_payload.get("count_rates")
    lt_value = run_payload.get("live_times")
    if rates_value is not None and lt_value is not None:
        if torch.is_tensor(rates_value):
            rates_arr = rates_value.detach().cpu().numpy()
        else:
            rates_arr = np.asarray(rates_value)

        if torch.is_tensor(lt_value):
            lt_arr = lt_value.detach().cpu().numpy()
        else:
            lt_arr = np.asarray(lt_value)

        if (
            rates_arr.ndim == 1
            and lt_arr.ndim == 1
            and rates_arr.shape[0] == counts.shape[0]
            and lt_arr.shape[0] == counts.shape[0]
        ):
            metadata_scales = rates_arr * lt_arr
            finite_positive = np.isfinite(metadata_scales) & (metadata_scales > 0)
            if bool(np.all(finite_positive)):
                target_count_rates = np.asarray(metadata_scales, dtype=np.float32)

    lstm = LSTMTemporalDetector(device=args.device, verbose=False)
    lstm.load(str(args.lstm_model_path.resolve()))

    arad = ARADDetector(device=args.device, verbose=False)
    arad.load(str(args.arad_model_path.resolve()))

    valid_lstm_indices = np.where(
        np.isfinite(lstm.score_time_series(ts, target_count_rates=target_count_rates))
    )[0]
    indices = _select_random_indices(valid_lstm_indices, int(args.num_spectra), int(args.seed))
    if not indices:
        raise RuntimeError("No valid indices available for LSTM reconstruction")

    lstm_curves: List[np.ndarray] = []
    arad_curves: List[np.ndarray] = []

    for idx in indices:
        lstm_recon = _reconstruct_lstm_at_index(lstm, counts, idx, target_count_rates)
        if lstm_recon is None:
            continue

        spec = ts[idx]
        arad_count_rate = arad.reconstruct(spec)
        arad_recon = arad_count_rate * _spectrum_time(spec)

        if args.normalize:
            lstm_curves.append(_normalize_curve(lstm_recon))
            arad_curves.append(_normalize_curve(arad_recon))
        else:
            lstm_curves.append(np.asarray(lstm_recon, dtype=float))
            arad_curves.append(np.asarray(arad_recon, dtype=float))

    if not lstm_curves or not arad_curves:
        raise RuntimeError("Failed to build reconstruction overlays")

    used_indices = indices[: len(lstm_curves)]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "normalized" if args.normalize else "raw"
    lstm_out = args.output_dir / f"lstm_overlay_{suffix}.png"
    arad_out = args.output_dir / f"arad_overlay_{suffix}.png"

    title_tail = "(normalized)" if args.normalize else "(raw scale)"
    _overlay_plot(
        curves=lstm_curves,
        indices=used_indices,
        title=f"LSTM Reconstructions Overlay - {len(used_indices)} Random Spectra {title_tail}",
        out_path=lstm_out,
        log_y=bool(args.log_y),
    )
    _overlay_plot(
        curves=arad_curves,
        indices=used_indices,
        title=f"ARAD Reconstructions Overlay - {len(used_indices)} Random Spectra {title_tail}",
        out_path=arad_out,
        log_y=bool(args.log_y),
    )

    summary: Dict[str, object] = {
        "run_file": str(args.run_file.resolve()),
        "lstm_model_path": str(args.lstm_model_path.resolve()),
        "arad_model_path": str(args.arad_model_path.resolve()),
        "num_requested": int(args.num_spectra),
        "num_used": int(len(used_indices)),
        "indices": used_indices,
        "normalize": bool(args.normalize),
        "log_y": bool(args.log_y),
        "lstm_plot": str(lstm_out.resolve()),
        "arad_plot": str(arad_out.resolve()),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Saved LSTM overlay: {lstm_out}")
    print(f"Saved ARAD overlay: {arad_out}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
