"""
Temporal LSTM anomaly detector for gamma-ray spectra.

This module adds a causal sequence model that reconstructs only the
final spectrum in each fixed-length history window.

Notes
-----
- Inputs are spectral counts directly (no count-rate requirement).
- Training pipeline is intentionally deferred so external training work
  can be integrated later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. LSTMTemporalDetector requires PyTorch. "
        "Install with: pip install torch"
    )

from gammaflow.algorithms.base import BaseDetector


@dataclass
class TemporalModelConfig:
    """Configuration used to initialize the temporal model."""

    n_bins: int
    latent_dim: int
    lstm_hidden_dim: int
    lstm_layers: int
    dropout: float


class TemporalLSTMAutoencoder(nn.Module):
    """
    LSTM autoencoder that reconstructs the final spectrum in a sequence.

    Input shape
    -----------
    (batch, seq_len, n_bins)

    Output shape
    ------------
    (batch, n_bins) representing reconstruction of the final time step.
    """

    def __init__(
        self,
        n_bins: int,
        latent_dim: int = 64,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_bins = int(n_bins)
        self.latent_dim = int(latent_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.lstm_layers = int(lstm_layers)
        self.dropout = float(dropout)

        self.encoder = nn.Sequential(
            nn.Linear(self.n_bins, self.latent_dim),
            nn.Mish(),
            nn.LayerNorm(self.latent_dim),
        )

        lstm_dropout = self.dropout if self.lstm_layers > 1 else 0.0
        self.temporal_core = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.latent_dim),
            nn.Mish(),
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.n_bins),
            nn.Softplus(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Forward pass from sequence window to final-step reconstruction."""
        encoded = self.encoder(windows)
        temporal_out, _ = self.temporal_core(encoded)
        final_embedding = temporal_out[:, -1, :]
        reconstructed_last = self.decoder(final_embedding)
        return reconstructed_last


class LSTMTemporalDetector(BaseDetector):
    """
    Temporal anomaly detector with causal LSTM context.

    The detector scores each time index ``t`` by reconstructing spectrum ``t``
    from the fixed-length causal history window ending at ``t``.

    Training pipeline implementation is intentionally deferred. Use one of:
    - ``fit(..., trainer_fn=...)`` to plug in external training code
    - ``initialize_model(n_bins)`` + ``load(path)`` for pretrained weights

    Parameters
    ----------
    seq_len : int
        Number of spectra in each causal window.
    seq_stride : int
        Step between spectra inside each window.
    latent_dim : int
        Per-spectrum latent dimension before LSTM.
    lstm_hidden_dim : int
        LSTM hidden dimension.
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout used in model blocks.
    threshold : float or None
        Detection threshold. If None, set via ``set_threshold`` or FAR calibration.
    aggregation_gap : float
        Gap in seconds for merging adjacent alarms.
    loss_type : str
        ``'jsd'`` or ``'chi2'``.
    device : str or None
        Torch device. Auto-select if None.
    verbose : bool
        Print verbose status messages.
    """

    def __init__(
        self,
        seq_len: int = 20,
        seq_stride: int = 1,
        latent_dim: int = 64,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        threshold: Optional[float] = None,
        aggregation_gap: float = 2.0,
        loss_type: str = "jsd",
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "LSTMTemporalDetector requires PyTorch. Install with: pip install torch"
            )

        super().__init__(threshold=threshold, aggregation_gap=aggregation_gap)

        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        if seq_stride < 1:
            raise ValueError(f"seq_stride must be >= 1, got {seq_stride}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if lstm_hidden_dim < 1:
            raise ValueError(f"lstm_hidden_dim must be >= 1, got {lstm_hidden_dim}")
        if lstm_layers < 1:
            raise ValueError(f"lstm_layers must be >= 1, got {lstm_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride)
        self.latent_dim = int(latent_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.lstm_layers = int(lstm_layers)
        self.dropout = float(dropout)
        self.loss_type = str(loss_type).lower()
        self.verbose = bool(verbose)

        if self.loss_type not in ("jsd", "chi2"):
            raise ValueError(f"loss_type must be 'jsd' or 'chi2', got '{loss_type}'")

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        self.model_: Optional[TemporalLSTMAutoencoder] = None
        self.n_bins_: Optional[int] = None

        if self.verbose:
            print(f"LSTMTemporalDetector using device: {self.device}")

    @property
    def warmup_samples(self) -> int:
        """Number of leading samples without enough causal context."""
        return (self.seq_len - 1) * self.seq_stride

    @property
    def is_trained(self) -> bool:
        return self.model_ is not None and self.n_bins_ is not None

    def initialize_model(self, n_bins: int) -> "LSTMTemporalDetector":
        """Initialize model architecture without running a training loop."""
        self.n_bins_ = int(n_bins)
        self.model_ = TemporalLSTMAutoencoder(
            n_bins=self.n_bins_,
            latent_dim=self.latent_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.model_.eval()
        return self

    def fit(self, background_data, **kwargs) -> "LSTMTemporalDetector":
        """
        Fit detector model.

        This method supports integration with external training code through
        ``trainer_fn`` and intentionally does not implement an internal training
        pipeline yet.

        Parameters
        ----------
        background_data : SpectralTimeSeries
            Background-only time series data.
        trainer_fn : callable, optional
            External training callback.
            Signature: ``trainer_fn(detector, background_data, **kwargs) -> detector``.
        """
        if not hasattr(background_data, "counts"):
            raise TypeError("background_data must provide a 'counts' attribute")

        counts = np.asarray(background_data.counts)
        if counts.ndim != 2:
            raise ValueError("background_data.counts must have shape (n_spectra, n_bins)")

        self.initialize_model(n_bins=counts.shape[1])

        trainer_fn = kwargs.pop("trainer_fn", None)
        if trainer_fn is None:
            raise NotImplementedError(
                "Internal training pipeline is deferred. Provide trainer_fn=... "
                "or load pretrained weights with load(path)."
            )

        trained_detector = trainer_fn(self, background_data, **kwargs)
        if trained_detector is None:
            return self
        return trained_detector

    def score_spectrum(self, spectrum) -> float:
        """
        Single-spectrum scoring is undefined without temporal context.

        Use ``process_time_series`` for causal sequence-aware scoring.
        """
        raise RuntimeError(
            "LSTMTemporalDetector requires temporal context. "
            "Use process_time_series(...) instead of score_spectrum(...)."
        )

    def _normalize_for_jsd(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        x = torch.clamp(x, min=eps)
        return x / torch.clamp(torch.sum(x, dim=-1, keepdim=True), min=eps)

    def _jsd_score(self, target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        p = self._normalize_for_jsd(target)
        q = self._normalize_for_jsd(recon)
        m = 0.5 * (p + q)
        kld_pm = torch.sum(p * torch.log(torch.clamp(p / m, min=1e-10)), dim=-1)
        kld_qm = torch.sum(q * torch.log(torch.clamp(q / m, min=1e-10)), dim=-1)
        return torch.sqrt(0.5 * (kld_pm + kld_qm))

    def _chi2_score(self, target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        recon = torch.clamp(recon, min=eps)
        return torch.sum((target - recon) ** 2 / recon, dim=-1)

    def _score_batch(self, targets: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "jsd":
            return self._jsd_score(targets, recon)
        return self._chi2_score(targets, recon)

    def _window_indices_for_end(self, end_idx: int) -> Optional[np.ndarray]:
        offsets = np.arange(self.seq_len - 1, -1, -1, dtype=int) * self.seq_stride
        indices = end_idx - offsets
        if indices[0] < 0:
            return None
        return indices

    def _build_window_for_index(
        self,
        counts: np.ndarray,
        end_idx: int,
    ) -> Optional[np.ndarray]:
        indices = self._window_indices_for_end(end_idx)
        if indices is None:
            return None
        return counts[indices]

    def score_time_series(self, time_series) -> np.ndarray:
        """
        Score time series with causal rolling windows.

        Returns one score per input spectrum. Entries without enough history
        are ``np.nan``.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Detector must be trained or loaded before scoring. "
                "Call fit(..., trainer_fn=...) or load(path)."
            )

        counts = np.asarray(time_series.counts, dtype=np.float32)
        if counts.ndim != 2:
            raise ValueError("time_series.counts must have shape (n_spectra, n_bins)")
        if counts.shape[1] != self.n_bins_:
            raise ValueError(
                f"Time series has {counts.shape[1]} bins, expected {self.n_bins_}."
            )

        metrics = np.full(counts.shape[0], np.nan, dtype=float)

        self.model_.eval()
        with torch.no_grad():
            for i in range(counts.shape[0]):
                window = self._build_window_for_index(counts, i)
                if window is None:
                    continue

                window_tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)
                target_tensor = torch.from_numpy(counts[i]).unsqueeze(0).to(self.device)
                reconstruction = self.model_(window_tensor)
                score = self._score_batch(target_tensor, reconstruction)
                metrics[i] = float(score.item())

        return metrics

    def process_time_series(self, time_series) -> np.ndarray:
        """
        Process a time series and aggregate alarms from temporal scores.

        This override exists because ``score_spectrum`` is intentionally
        unavailable for temporal models.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Detector must be trained or loaded before processing. "
                "Call fit(..., trainer_fn=...) or load(path)."
            )
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not set. Call set_threshold() or set_threshold_by_far() first."
            )

        self.reset_alarms()
        scores = self.score_time_series(time_series)
        times = self._extract_timestamps(time_series)

        for score, t in zip(scores, times):
            if np.isnan(score):
                continue

            if score > self.threshold:
                if not self._is_alarming:
                    self._start_alarm(t, float(score))
                else:
                    self._update_alarm_peak(t, float(score))
            else:
                if self._is_alarming:
                    self._end_alarm(t)

        if self._is_alarming:
            self._end_alarm(float(times[-1]))

        return scores

    def save(self, path: str) -> None:
        """Save model weights and detector config."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an uninitialized/untrained model")

        model_config = TemporalModelConfig(
            n_bins=self.n_bins_,
            latent_dim=self.latent_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
        )

        payload = {
            "model_state": self.model_.state_dict(),
            "model_config": asdict(model_config),
            "seq_len": self.seq_len,
            "seq_stride": self.seq_stride,
            "loss_type": self.loss_type,
            "threshold": self.threshold,
            "aggregation_gap": self.aggregation_gap,
        }

        torch.save(payload, path)

        if self.verbose:
            print(f"Temporal model saved to {path}")

    def load(self, path: str) -> None:
        """Load model weights and detector config."""
        payload = torch.load(Path(path), map_location=self.device)

        cfg = payload["model_config"]
        self.seq_len = int(payload.get("seq_len", self.seq_len))
        self.seq_stride = int(payload.get("seq_stride", self.seq_stride))
        self.loss_type = str(payload.get("loss_type", self.loss_type)).lower()
        self.threshold = payload.get("threshold", self.threshold)
        self.aggregation_gap = float(payload.get("aggregation_gap", self.aggregation_gap))

        self.n_bins_ = int(cfg["n_bins"])
        self.latent_dim = int(cfg["latent_dim"])
        self.lstm_hidden_dim = int(cfg["lstm_hidden_dim"])
        self.lstm_layers = int(cfg["lstm_layers"])
        self.dropout = float(cfg["dropout"])

        self.model_ = TemporalLSTMAutoencoder(
            n_bins=self.n_bins_,
            latent_dim=self.latent_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.model_.load_state_dict(payload["model_state"])
        self.model_.eval()

        if self.verbose:
            print(f"Temporal model loaded from {path}")

    def __repr__(self) -> str:
        return (
            f"LSTMTemporalDetector(seq_len={self.seq_len}, "
            f"seq_stride={self.seq_stride}, "
            f"loss_type='{self.loss_type}', "
            f"trained={self.is_trained}, alarms={len(self.alarms)})"
        )
