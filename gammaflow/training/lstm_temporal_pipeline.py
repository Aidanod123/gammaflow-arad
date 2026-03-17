"""Training pipeline for LSTMTemporalDetector using preprocessed run tensors.

This pipeline expects each run file (``run*.pt``) to contain a dictionary with at
least:
- ``spectra``: tensor/array with shape (n_spectra, n_bins)

The preprocessed datasets in this repository already satisfy this contract.
"""

from __future__ import annotations

import copy
import json
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from typing import Callable

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


@dataclass
class RunIndexEntry:
    """Index metadata for a single run file."""

    path: Path
    n_spectra: int
    n_windows: int


class PreprocessedTemporalWindowDataset(Dataset):
    """Map-style dataset that serves causal windows from preprocessed run files.

    Each sample is a tuple ``(window, target)`` where:
    - ``window`` has shape ``(seq_len, n_bins)``
    - ``target`` has shape ``(n_bins,)`` and corresponds to the final spectrum
      in the window.
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
        self._cache: "OrderedDict[Path, np.ndarray]" = OrderedDict()

        total = 0
        inferred_n_bins: Optional[int] = None

        for run_path in run_files:
            spectra = self._load_spectra(run_path)
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
    def _load_spectra(path: Path) -> np.ndarray:
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

        return arr.astype(np.float32, copy=False)

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

    def _get_cached_spectra(self, run_idx: int) -> np.ndarray:
        path = self._entries[run_idx].path
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached

        spectra = self._load_spectra(path)
        self._cache[path] = spectra
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return spectra

    def __getitem__(self, idx: int):
        run_idx, local_window_idx = self._resolve_index(idx)
        spectra = self._get_cached_spectra(run_idx)

        end_idx = self.warmup + local_window_idx
        offsets = np.arange(self.seq_len - 1, -1, -1, dtype=int) * self.seq_stride
        indices = end_idx - offsets

        window = spectra[indices]
        target = spectra[end_idx]

        return torch.from_numpy(window), torch.from_numpy(target)


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


def _normalize_prob(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    x = torch.clamp(x, min=eps)
    return x / torch.clamp(x.sum(dim=-1, keepdim=True), min=eps)


def _jsd_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    p = _normalize_prob(target)
    q = _normalize_prob(recon)
    m = 0.5 * (p + q)
    kld_pm = torch.sum(p * torch.log(torch.clamp(p / m, min=1e-10)), dim=-1)
    kld_qm = torch.sum(q * torch.log(torch.clamp(q / m, min=1e-10)), dim=-1)
    return torch.sqrt(0.5 * (kld_pm + kld_qm)).mean()


def _chi2_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    recon = torch.clamp(recon, min=eps)
    return torch.mean(torch.sum((target - recon) ** 2 / recon, dim=-1))


def _get_loss_fn(loss_name: str):
    name = str(loss_name).lower()
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "jsd":
        return _jsd_loss
    if name == "chi2":
        return _chi2_loss
    raise ValueError(f"Unsupported loss '{loss_name}'. Use one of: mse, jsd, chi2")


def train_lstm_temporal_from_preprocessed(
    preprocessed_dir: str,
    output_model_path: str,
    seq_len: int = 20,
    seq_stride: int = 1,
    latent_dim: int = 64,
    lstm_hidden_dim: int = 128,
    lstm_layers: int = 1,
    dropout: float = 0.2,
    loss_type: str = "jsd",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    batch_size: int = 128,
    epochs: int = 20,
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

    Returns a dictionary containing trained detector, training history, and paths.
    """
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
        loss_type=loss_type,
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
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    loss_fn = _get_loss_fn(loss_type)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0

        for windows, targets in bundles.train_loader:
            windows = windows.to(detector.device)
            targets = targets.to(detector.device)

            optimizer.zero_grad(set_to_none=True)
            recon = model(windows)
            loss = loss_fn(targets, recon)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            batch_size_actual = int(windows.shape[0])
            train_loss_sum += float(loss.item()) * batch_size_actual
            train_items += batch_size_actual

        train_loss = train_loss_sum / max(train_items, 1)

        model.eval()
        val_loss_sum = 0.0
        val_items = 0
        with torch.no_grad():
            for windows, targets in bundles.val_loader:
                windows = windows.to(detector.device)
                targets = targets.to(detector.device)
                recon = model(windows)
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

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch_end_callback is not None:
            epoch_end_callback(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "lr": float(lr),
                    "best_val_loss": float(best_val),
                }
            )

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
        "loss_type": loss_type,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": epochs,
        "val_fraction": val_fraction,
        "seed": seed,
        "device": str(detector.device),
        "require_cuda": require_cuda,
        "n_bins": bundles.n_bins,
        "train_runs": [p.name for p in bundles.train_run_files],
        "val_runs": [p.name for p in bundles.val_run_files],
        "best_val_loss": best_val,
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
    }
