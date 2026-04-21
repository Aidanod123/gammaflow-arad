"""Training pipeline for LSTMTemporalDetector using preprocessed run tensors.

This pipeline expects each run file (``run*.pt``) to contain a dictionary with at
least:
- ``spectra``: tensor/array with shape (n_spectra, n_bins)

Optionally, for physical-domain chi-squared loss and count-rate conditioning:
- ``count_rates``: tensor/array with shape (n_spectra,) — gross count rate (counts/s)
- ``live_times``: tensor/array with shape (n_spectra,) — live time per spectrum (s)
- ``integration_time``: scalar — integration window duration (s)

The preprocessed datasets in this repository satisfy this full contract.
"""

from __future__ import annotations

import copy
import json
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for the LSTM temporal training pipeline. "
        "Install with: pip install torch"
    ) from exc

from gammaflow.algorithms.lstm_temporal import LSTMTemporalDetector
from gammaflow.training.losses import get_loss_fn


@dataclass
class RunIndexEntry:
    """Index metadata for a single run file."""

    path: Path
    n_spectra: int
    n_windows: int


class PreprocessedTemporalWindowDataset(Dataset):
    """Map-style dataset that serves causal windows from preprocessed run files.

    Each sample is a tuple ``(window, target, target_scale)`` where:
    - ``window`` has shape ``(seq_len, n_bins)``
    - ``target`` has shape ``(n_bins,)`` — final spectrum in the window
    - ``target_scale`` is a scalar (total counts) for chi2 de-normalization
      and count-rate decoder conditioning
    """

    def __init__(
        self,
        run_files: Sequence[Path],
        seq_len: int,
        seq_stride: int = 1,
        cache_size: int = 2,
    ):
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        if seq_stride < 1:
            raise ValueError(f"seq_stride must be >= 1, got {seq_stride}")
        if cache_size < 1:
            raise ValueError(f"cache_size must be >= 1, got {cache_size}")

        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride)
        self.cache_size = int(cache_size)
        self.warmup = (self.seq_len - 1) * self.seq_stride

        self._entries: List[RunIndexEntry] = []
        self._cumulative_windows: List[int] = []
        self._cache: "OrderedDict[Path, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

        total = 0
        inferred_n_bins: Optional[int] = None

        for run_path in run_files:
            spectra, _ = self._load_run_arrays(run_path)
            n_spectra, n_bins = spectra.shape

            if inferred_n_bins is None:
                inferred_n_bins = int(n_bins)
            elif int(n_bins) != inferred_n_bins:
                raise ValueError(
                    f"Inconsistent n_bins in {run_path}: {n_bins} != {inferred_n_bins}"
                )

            n_windows = max(0, n_spectra - self.warmup)
            self._entries.append(
                RunIndexEntry(path=Path(run_path), n_spectra=int(n_spectra), n_windows=int(n_windows))
            )
            total += n_windows
            self._cumulative_windows.append(total)

        self.n_bins = int(inferred_n_bins) if inferred_n_bins is not None else 0

    @staticmethod
    def _load_run_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load spectra and target_scales (total counts) from a run file.

        target_scales = count_rates * live_times when metadata is available,
        otherwise falls back to the row-sum of the spectra array.
        """
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "spectra" not in obj:
            raise ValueError(f"Expected dict with 'spectra' in {path}")

        spectra = obj["spectra"]
        if torch.is_tensor(spectra):
            arr = spectra.detach().cpu().numpy()
        else:
            arr = np.asarray(spectra)

        if arr.ndim != 2:
            raise ValueError(f"spectra in {path} must have shape (n_spectra, n_bins), got {arr.shape}")

        arr = arr.astype(np.float32, copy=False)

        # Default scale: row sums (=1.0 each for L1-normalized spectra).
        scales_arr = np.sum(arr, axis=1)

        # Preferred: count_rates * live_times gives true total counts.
        if "count_rates" in obj and "live_times" in obj:
            count_rates = obj["count_rates"]
            live_times = obj["live_times"]

            if torch.is_tensor(count_rates):
                rates_arr = count_rates.detach().cpu().numpy()
            else:
                rates_arr = np.asarray(count_rates)

            if torch.is_tensor(live_times):
                live_times_arr = live_times.detach().cpu().numpy()
            else:
                live_times_arr = np.asarray(live_times)

            if (
                rates_arr.ndim == 1
                and live_times_arr.ndim == 1
                and rates_arr.shape[0] == arr.shape[0]
                and live_times_arr.shape[0] == arr.shape[0]
            ):
                metadata_scales = rates_arr * live_times_arr
                finite_positive = np.isfinite(metadata_scales) & (metadata_scales > 0)
                if bool(np.all(finite_positive)):
                    scales_arr = metadata_scales

        return arr, np.asarray(scales_arr, dtype=np.float32)

    def __len__(self) -> int:
        return self._cumulative_windows[-1] if self._cumulative_windows else 0

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        run_idx = bisect_right(self._cumulative_windows, idx)
        prev_cum = self._cumulative_windows[run_idx - 1] if run_idx > 0 else 0
        local_idx = idx - prev_cum
        return run_idx, local_idx

    def _get_cached_spectra(self, run_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._entries[run_idx].path
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached

        run_arrays = self._load_run_arrays(path)
        self._cache[path] = run_arrays
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return run_arrays

    def __getitem__(self, idx: int):
        run_idx, local_window_idx = self._resolve_index(idx)
        spectra, target_scales = self._get_cached_spectra(run_idx)

        end_idx = self.warmup + local_window_idx
        offsets = np.arange(self.seq_len - 1, -1, -1, dtype=int) * self.seq_stride
        indices = end_idx - offsets

        window = spectra[indices]
        target = spectra[end_idx]
        target_scale = np.float32(target_scales[end_idx])

        return (
            torch.from_numpy(window),
            torch.from_numpy(target),
            torch.tensor(target_scale, dtype=torch.float32),
        )


@dataclass
class DataLoadersBundle:
    """Train/validation loaders and metadata."""

    train_loader: DataLoader
    val_loader: DataLoader
    train_run_files: List[Path]
    val_run_files: List[Path]
    n_bins: int


def _discover_run_files(preprocessed_dir: Path) -> List[Path]:
    run_files = sorted(
        preprocessed_dir.glob("run*.pt"),
        key=lambda p: int(p.stem.replace("run", "")),
    )
    if not run_files:
        raise FileNotFoundError(f"No run*.pt files found in {preprocessed_dir}")
    return run_files


def _split_run_files(
    run_files: Sequence[Path],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    ids = np.arange(len(run_files))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_val = max(1, int(round(len(ids) * val_fraction)))
    val_ids = set(ids[:n_val].tolist())

    train = [run_files[i] for i in range(len(run_files)) if i not in val_ids]
    val = [run_files[i] for i in range(len(run_files)) if i in val_ids]

    if not train:
        raise ValueError("Train split is empty. Reduce val_fraction.")

    return train, val


def build_dataloaders_from_preprocessed(
    preprocessed_dir: str,
    seq_len: int,
    seq_stride: int = 1,
    val_fraction: float = 0.2,
    seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 0,
    cache_size: int = 2,
) -> DataLoadersBundle:
    """Build train/validation DataLoaders from preprocessed run files."""
    root = Path(preprocessed_dir)
    run_files = _discover_run_files(root)
    train_files, val_files = _split_run_files(run_files, val_fraction=val_fraction, seed=seed)

    train_ds = PreprocessedTemporalWindowDataset(
        train_files,
        seq_len=seq_len,
        seq_stride=seq_stride,
        cache_size=cache_size,
    )
    val_ds = PreprocessedTemporalWindowDataset(
        val_files,
        seq_len=seq_len,
        seq_stride=seq_stride,
        cache_size=cache_size,
    )

    if train_ds.n_bins != val_ds.n_bins:
        raise ValueError(f"Train and val n_bins mismatch: {train_ds.n_bins} != {val_ds.n_bins}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return DataLoadersBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_run_files=train_files,
        val_run_files=val_files,
        n_bins=train_ds.n_bins,
    )


def _sample_history_latent_mask(
    batch_size: int,
    seq_len: int,
    latent_mask_pct: float,
    rng: np.random.Generator,
    device: torch.device,
    mask_target_timestep: bool = False,
) -> Optional[torch.Tensor]:
    """Sample a boolean mask over latent timesteps.

    History steps are randomly masked at rate ``latent_mask_pct``.
    When ``mask_target_timestep`` is True the final timestep is always masked
    to prevent the encoder from short-circuiting reconstruction.
    """
    if seq_len <= 1:
        return None
    history_mask = np.zeros((batch_size, seq_len - 1), dtype=bool)
    if latent_mask_pct > 0.0:
        history_mask = rng.random((batch_size, seq_len - 1)) < float(latent_mask_pct)

    final_col = np.full((batch_size, 1), bool(mask_target_timestep), dtype=bool)
    full_mask = np.concatenate([history_mask, final_col], axis=1)
    if not bool(full_mask.any()):
        return None
    return torch.from_numpy(full_mask).to(device=device)


def train_lstm_temporal_from_preprocessed(
    preprocessed_dir: str,
    output_model_path: str,
    seq_len: int = 20,
    seq_stride: int = 1,
    latent_dim: int = 64,
    lstm_hidden_dim: int = 128,
    lstm_layers: int = 1,
    dropout: float = 0.2,
    mask_target: bool = True,
    loss_type: str = "jsd",
    output_activation: str = "sigmoid",
    count_rate_conditioning: bool = False,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 3,
    scheduler_min_lr: float = 1e-6,
    scheduler_threshold: float = 1e-4,
    scheduler_cooldown: int = 1,
    batch_size: int = 128,
    epochs: int = 20,
    min_epochs: int = 10,
    early_stopping_patience: int = 8,
    early_stopping_min_delta: float = 1e-4,
    latent_mask_pct: float = 0.0,
    latent_mask_seed: Optional[int] = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    grad_clip_norm: float = 1.0,
    cache_size: int = 2,
    device: Optional[str] = None,
    require_cuda: bool = False,
    verbose: bool = True,
    epoch_end_callback: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, object]:
    """Train LSTMTemporalDetector on preprocessed run tensors.

    Returns a dictionary with trained detector, history, paths, and stop metadata.
    """
    # --- Input validation ---
    if not (0.0 <= float(latent_mask_pct) <= 1.0):
        raise ValueError(f"latent_mask_pct must be in [0, 1], got {latent_mask_pct}")
    if int(epochs) < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if int(min_epochs) < 1:
        raise ValueError(f"min_epochs must be >= 1, got {min_epochs}")
    if int(min_epochs) > int(epochs):
        raise ValueError(f"min_epochs ({min_epochs}) cannot exceed epochs ({epochs})")
    if not (0.0 < float(scheduler_factor) < 1.0):
        raise ValueError(f"scheduler_factor must be in (0, 1), got {scheduler_factor}")
    if int(scheduler_patience) < 0:
        raise ValueError(f"scheduler_patience must be >= 0, got {scheduler_patience}")
    if float(scheduler_min_lr) < 0.0:
        raise ValueError(f"scheduler_min_lr must be >= 0, got {scheduler_min_lr}")
    if float(scheduler_threshold) < 0.0:
        raise ValueError(f"scheduler_threshold must be >= 0, got {scheduler_threshold}")
    if int(scheduler_cooldown) < 0:
        raise ValueError(f"scheduler_cooldown must be >= 0, got {scheduler_cooldown}")
    if int(early_stopping_patience) < 1:
        raise ValueError(f"early_stopping_patience must be >= 1, got {early_stopping_patience}")
    if float(early_stopping_min_delta) < 0.0:
        raise ValueError(f"early_stopping_min_delta must be >= 0, got {early_stopping_min_delta}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    bundles = build_dataloaders_from_preprocessed(
        preprocessed_dir=preprocessed_dir,
        seq_len=seq_len,
        seq_stride=seq_stride,
        val_fraction=val_fraction,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_size=cache_size,
    )

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was required but is not available in this environment. "
            "Check your PyTorch CUDA install and GPU drivers."
        )

    if device is not None and str(device).lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{device}' but CUDA is not available. "
            "Use --device cpu or install CUDA-enabled PyTorch."
        )

    detector = LSTMTemporalDetector(
        seq_len=seq_len,
        seq_stride=seq_stride,
        latent_dim=latent_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_layers=lstm_layers,
        dropout=dropout,
        mask_target=bool(mask_target),
        loss_type=loss_type,
        output_activation=output_activation,
        count_rate_conditioning=count_rate_conditioning,
        threshold=None,
        device=device,
        verbose=verbose,
    )
    detector.initialize_model(n_bins=bundles.n_bins)

    if require_cuda and detector.device.type != "cuda":
        raise RuntimeError(
            f"CUDA was required but detector resolved to '{detector.device}'."
        )

    if verbose and detector.device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(detector.device)
        print(f"Training on GPU: {gpu_name} ({detector.device})")

    model = detector.model_

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_factor),
        patience=int(scheduler_patience),
        threshold=float(scheduler_threshold),
        cooldown=int(scheduler_cooldown),
        min_lr=float(scheduler_min_lr),
    )
    loss_fn = get_loss_fn(loss_type)

    mask_rng_seed = seed if latent_mask_seed is None else int(latent_mask_seed)
    train_mask_rng = np.random.default_rng(mask_rng_seed)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = "completed_all_epochs"
    epochs_completed = 0

    for epoch in range(1, epochs + 1):
        epochs_completed = epoch
        model.train()
        train_loss_sum = 0.0
        train_items = 0

        for windows, targets, target_scales in bundles.train_loader:
            windows = windows.to(detector.device)
            targets = targets.to(detector.device)
            target_scales = target_scales.to(detector.device)

            latent_mask = _sample_history_latent_mask(
                batch_size=int(windows.shape[0]),
                seq_len=int(windows.shape[1]),
                latent_mask_pct=float(latent_mask_pct),
                rng=train_mask_rng,
                device=detector.device,
                mask_target_timestep=bool(mask_target),
            )

            cr = target_scales if count_rate_conditioning else None

            optimizer.zero_grad(set_to_none=True)
            recon = model(windows, latent_timestep_mask=latent_mask, count_rate=cr)
            if str(loss_type).lower() == "chi2":
                loss = loss_fn(targets, recon, target_scales=target_scales)
            else:
                loss = loss_fn(targets, recon)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            batch_size_actual = int(windows.shape[0])
            train_loss_sum += float(loss.item()) * batch_size_actual
            train_items += batch_size_actual

        train_loss = train_loss_sum / max(train_items, 1)

        # Reset RNG with same seed each epoch so val masking is reproducible.
        val_mask_rng = np.random.default_rng(mask_rng_seed)
        model.eval()
        val_loss_sum = 0.0
        val_items = 0
        with torch.no_grad():
            for windows, targets, target_scales in bundles.val_loader:
                windows = windows.to(detector.device)
                targets = targets.to(detector.device)
                target_scales = target_scales.to(detector.device)

                latent_mask = _sample_history_latent_mask(
                    batch_size=int(windows.shape[0]),
                    seq_len=int(windows.shape[1]),
                    latent_mask_pct=float(latent_mask_pct),
                    rng=val_mask_rng,
                    device=detector.device,
                    mask_target_timestep=bool(mask_target),
                )

                cr = target_scales if count_rate_conditioning else None
                recon = model(windows, latent_timestep_mask=latent_mask, count_rate=cr)
                if str(loss_type).lower() == "chi2":
                    loss = loss_fn(targets, recon, target_scales=target_scales)
                else:
                    loss = loss_fn(targets, recon)
                batch_size_actual = int(windows.shape[0])
                val_loss_sum += float(loss.item()) * batch_size_actual
                val_items += batch_size_actual

        val_loss = val_loss_sum / max(val_items, 1)
        scheduler.step(val_loss)

        lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr:.2e}"
            )

        if val_loss < (best_val - float(early_stopping_min_delta)):
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch_end_callback is not None:
            epoch_end_callback(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "lr": float(lr),
                    "best_val_loss": float(best_val),
                    "best_epoch": float(best_epoch),
                    "epochs_without_improvement": float(epochs_without_improvement),
                }
            )

        if (
            epoch >= int(min_epochs)
            and epochs_without_improvement >= int(early_stopping_patience)
        ):
            stopped_early = True
            stop_reason = (
                f"early_stopping_no_improvement_for_{int(early_stopping_patience)}_epochs"
            )
            if verbose:
                print(
                    f"Early stopping at epoch {epoch:03d} "
                    f"(best_epoch={best_epoch:03d}, best_val_loss={best_val:.6f})"
                )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    out_model_path = Path(output_model_path)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(out_model_path))

    metadata = {
        "preprocessed_dir": str(Path(preprocessed_dir).resolve()),
        "output_model_path": str(out_model_path.resolve()),
        "seq_len": seq_len,
        "seq_stride": seq_stride,
        "latent_dim": latent_dim,
        "lstm_hidden_dim": lstm_hidden_dim,
        "lstm_layers": lstm_layers,
        "dropout": dropout,
        "mask_target": bool(mask_target),
        "loss_type": loss_type,
        "output_activation": output_activation,
        "count_rate_conditioning": count_rate_conditioning,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "scheduler_min_lr": scheduler_min_lr,
        "scheduler_threshold": scheduler_threshold,
        "scheduler_cooldown": scheduler_cooldown,
        "batch_size": batch_size,
        "epochs": epochs,
        "min_epochs": min_epochs,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "latent_mask_pct": latent_mask_pct,
        "latent_mask_seed": latent_mask_seed,
        "val_fraction": val_fraction,
        "seed": seed,
        "device": str(detector.device),
        "require_cuda": require_cuda,
        "n_bins": bundles.n_bins,
        "train_runs": [p.name for p in bundles.train_run_files],
        "val_runs": [p.name for p in bundles.val_run_files],
        "best_val_loss": best_val,
        "best_epoch": int(best_epoch),
        "epochs_completed": int(epochs_completed),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "history": history,
    }

    metrics_path = out_model_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "detector": detector,
        "history": history,
        "model_path": str(out_model_path),
        "metrics_path": str(metrics_path),
        "best_val_loss": best_val,
        "best_epoch": int(best_epoch),
        "epochs_completed": int(epochs_completed),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
    }
