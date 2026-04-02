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
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
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
from gammaflow.algorithms.arad import ARADEncoderBlock, ARADDecoderBlock


def _score_batch_fn(
    targets: "torch.Tensor",
    recon: "torch.Tensor",
    loss_type: str,
) -> "torch.Tensor":
    """Lazy wrapper that imports from training.losses on first call."""
    from gammaflow.training.losses import score_batch  # noqa: deferred to avoid circular import
    return score_batch(targets, recon, loss_type)


@dataclass
class TemporalModelConfig:
    """Configuration used to initialize the temporal model."""

    n_bins: int
    latent_dim: int
    lstm_hidden_dim: int
    lstm_layers: int
    dropout: float
    use_attention: bool


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
        use_attention: bool = False,
    ):
        super().__init__()

        self.n_bins = int(n_bins)
        self.latent_dim = int(latent_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.lstm_layers = int(lstm_layers)
        self.dropout = float(dropout)
        self.use_attention = bool(use_attention)

        if self.n_bins % 32 != 0:
            raise ValueError(
                f"n_bins ({self.n_bins}) must be divisible by 32 for ARAD CNN blocks"
            )

        self._downsampled_bins = self.n_bins // 32

        self.encoder = nn.Sequential(
            ARADEncoderBlock(1, 8, 7, self.dropout),
            ARADEncoderBlock(8, 8, 5, self.dropout),
            ARADEncoderBlock(8, 8, 3, self.dropout),
            ARADEncoderBlock(8, 8, 3, self.dropout),
            ARADEncoderBlock(8, 8, 3, self.dropout),
            nn.Flatten(),
            nn.Linear(8 * self._downsampled_bins, self.latent_dim),
            nn.Mish(),
            nn.BatchNorm1d(self.latent_dim),
        )

        lstm_dropout = self.dropout if self.lstm_layers > 1 else 0.0
        self.temporal_core = nn.LSTM(
            input_size=self.latent_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.temporal_to_latent = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.latent_dim),
            nn.Mish(),
            nn.BatchNorm1d(self.latent_dim),
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(self.latent_dim, 8 * self._downsampled_bins),
            nn.Mish(),
            nn.BatchNorm1d(8 * self._downsampled_bins),
        )

        self.decoder = nn.Sequential(
            ARADDecoderBlock(8, 8, 3, self.dropout),
            ARADDecoderBlock(8, 8, 3, self.dropout),
            ARADDecoderBlock(8, 8, 3, self.dropout),
            ARADDecoderBlock(8, 8, 5, self.dropout),
            ARADDecoderBlock(8, 1, 7, self.dropout, is_output=True),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize spectra by per-sample max and add channel dimension."""
        eps = 1e-8
        max_val = torch.max(x, dim=1, keepdim=True).values
        normalized_x = x / (max_val + eps)
        return normalized_x.unsqueeze(1)

    def forward(
        self,
        windows: torch.Tensor,
        latent_timestep_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass from sequence window to final-step reconstruction.

        Parameters
        ----------
        windows : torch.Tensor
            Tensor of shape ``(batch, seq_len, n_bins)``.
        latent_timestep_mask : torch.Tensor, optional
            Boolean mask of shape ``(batch, seq_len)`` (or ``(seq_len,)``).
            True entries zero-out the corresponding latent timestep after the
            encoder and before the temporal LSTM.
        """
        batch_size, seq_len, _ = windows.shape
        flat_windows = windows.reshape(batch_size * seq_len, self.n_bins)
        normalized_windows = self._normalize_input(flat_windows)
        encoded_flat = self.encoder(normalized_windows)
        encoded = encoded_flat.view(batch_size, seq_len, self.latent_dim)

        if latent_timestep_mask is not None:
            if latent_timestep_mask.ndim == 1:
                latent_timestep_mask = latent_timestep_mask.unsqueeze(0)
            if latent_timestep_mask.shape != encoded.shape[:2]:
                raise ValueError(
                    "latent_timestep_mask must have shape "
                    f"(batch, seq_len)={tuple(encoded.shape[:2])}, "
                    f"got {tuple(latent_timestep_mask.shape)}"
                )
            mask = latent_timestep_mask.to(device=encoded.device, dtype=torch.bool)
            encoded = encoded.masked_fill(mask.unsqueeze(-1), 0.0)

        temporal_out, _ = self.temporal_core(encoded)
        if self.use_attention:
            # NOTE: When use_attention=True, external trainer_fn should pass a mask
            # with target timestep (-1) set True to prevent identity shortcut.
            query = temporal_out[:, -1, :].unsqueeze(1)
            scores = torch.sum(query * temporal_out, dim=-1)

            if latent_timestep_mask is not None:
                score_mask = mask
                scores.masked_fill_(score_mask, float("-inf"))
                fully_masked = score_mask.all(dim=1)
                if torch.any(fully_masked):
                    scores[fully_masked] = 0.0
                    score_mask = score_mask.clone()
                    score_mask[fully_masked, :] = False
                    score_mask[fully_masked, -1] = True
                    scores.masked_fill_(score_mask, float("-inf"))

            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
            context_vector = torch.sum(
                temporal_out * attention_weights.unsqueeze(-1),
                dim=1,
            )
            final_embedding = context_vector
        else:
            final_embedding = temporal_out[:, -1, :]

        decoder_latent = self.temporal_to_latent(final_embedding)
        decoded_linear = self.decoder_linear(decoder_latent)
        decoded_linear = decoded_linear.view(batch_size, 8, self._downsampled_bins)
        reconstructed_last = self.decoder(decoded_linear).view(batch_size, self.n_bins)
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
    use_attention : bool
        If True, reconstruct from attention-weighted temporal context instead
        of directly using only the final LSTM timestep embedding.
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
        use_attention: bool = False,
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
        self.use_attention = bool(use_attention)
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
            use_attention=self.use_attention,
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

    def _score_batch(self, targets: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """Per-sample anomaly scores.  Delegates to ``gammaflow.training.losses``."""
        return _score_batch_fn(targets, recon, self.loss_type)

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

    def _build_latent_mask_for_window(
        self,
        window_indices: np.ndarray,
        masked_index_set: Optional[Set[int]],
        latent_mask_pct: float,
        rng: Optional[np.random.Generator],
        mask_target_timestep: bool = False,
    ) -> Optional[np.ndarray]:
        mask = np.zeros(self.seq_len, dtype=bool)
        history_len = self.seq_len - 1

        if history_len <= 0:
            return np.array([True], dtype=bool) if mask_target_timestep else None

        if masked_index_set:
            mask[:history_len] = np.isin(
                window_indices[:history_len],
                np.fromiter(masked_index_set, dtype=int),
            )

        if latent_mask_pct > 0.0 and rng is not None:
            random_mask = rng.random(history_len) < float(latent_mask_pct)
            mask[:history_len] = np.logical_or(mask[:history_len], random_mask)

        if mask_target_timestep:
            mask[-1] = True

        return mask if bool(mask.any()) else None

    def score_time_series(
        self,
        time_series,
        mask_indices: Optional[Sequence[int]] = None,
        latent_mask_pct: float = 0.0,
        mask_seed: Optional[int] = None,
        mask_alarm_feedback: bool = False,
        feedback_threshold: Optional[float] = None,
        inference_batch_size: int = 256,
    ) -> np.ndarray:
        """
        Score time series with causal rolling windows.

        Returns one score per input spectrum. Entries without enough history
        are ``np.nan``.

        When ``mask_alarm_feedback`` is False, windows are pre-built and scored
        in batches of ``inference_batch_size`` for significantly faster GPU
        utilisation.  When alarm feedback is enabled, scoring falls back to a
        sequential loop because each score can influence later masks.

        Parameters
        ----------
        mask_indices : sequence of int, optional
            Absolute time-series indices whose latent history timestep should be
            masked (set to zero embedding) when they appear in a causal window.
        latent_mask_pct : float
            Additional random masking percentage applied per history timestep.
            Must be in ``[0, 1]``.
        mask_seed : int, optional
            Seed used for random masking reproducibility.
        mask_alarm_feedback : bool
            If True, any timestep whose score exceeds ``feedback_threshold`` is
            added to the latent mask set for subsequent windows. This supports
            deployment behavior where prior alarmed spectra are not fed back as
            normal history into the LSTM context.
        feedback_threshold : float, optional
            Threshold used for alarm-feedback masking. When ``None``, uses
            ``self.threshold``.
        inference_batch_size : int
            Number of windows per forward pass when using the batched path.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Detector must be trained or loaded before scoring. "
                "Call fit(..., trainer_fn=...) or load(path)."
            )

        if not (0.0 <= float(latent_mask_pct) <= 1.0):
            raise ValueError(f"latent_mask_pct must be in [0, 1], got {latent_mask_pct}")

        if mask_alarm_feedback:
            threshold_for_feedback = self.threshold if feedback_threshold is None else float(feedback_threshold)
            if threshold_for_feedback is None:
                raise RuntimeError(
                    "mask_alarm_feedback=True requires a threshold. "
                    "Set detector.threshold or pass feedback_threshold."
                )
        else:
            threshold_for_feedback = None

        counts = np.asarray(time_series.counts, dtype=np.float32)
        if counts.ndim != 2:
            raise ValueError("time_series.counts must have shape (n_spectra, n_bins)")
        if counts.shape[1] != self.n_bins_:
            raise ValueError(
                f"Time series has {counts.shape[1]} bins, expected {self.n_bins_}."
            )

        metrics = np.full(counts.shape[0], np.nan, dtype=float)

        masked_index_set: Optional[Set[int]] = None
        if mask_indices is not None:
            masked_index_set = {int(i) for i in mask_indices if int(i) >= 0}

        if mask_alarm_feedback:
            self._score_sequential(
                counts=counts,
                metrics=metrics,
                masked_index_set=masked_index_set,
                latent_mask_pct=float(latent_mask_pct),
                mask_seed=mask_seed,
                threshold_for_feedback=threshold_for_feedback,
            )
        else:
            self._score_batched(
                counts=counts,
                metrics=metrics,
                masked_index_set=masked_index_set,
                latent_mask_pct=float(latent_mask_pct),
                mask_seed=mask_seed,
                batch_size=max(1, int(inference_batch_size)),
            )

        return metrics

    # ------------------------------------------------------------------
    # Batched scoring (no alarm-feedback dependency between windows)
    # ------------------------------------------------------------------

    def _score_batched(
        self,
        counts: np.ndarray,
        metrics: np.ndarray,
        masked_index_set: Optional[Set[int]],
        latent_mask_pct: float,
        mask_seed: Optional[int],
        batch_size: int,
    ) -> None:
        """Pre-build all windows then score in GPU-friendly batches."""
        rng = np.random.default_rng(mask_seed) if latent_mask_pct > 0.0 else None
        static_mask_set: Set[int] = set(masked_index_set or set())

        # --- Phase 1: pre-materialise windows, targets and masks ----------
        valid_indices: List[int] = []
        windows_list: List[np.ndarray] = []
        masks_list: List[Optional[np.ndarray]] = []

        for i in range(counts.shape[0]):
            win_idx = self._window_indices_for_end(i)
            if win_idx is None:
                continue
            valid_indices.append(i)
            windows_list.append(counts[win_idx])
            masks_list.append(
                self._build_latent_mask_for_window(
                    window_indices=win_idx,
                    masked_index_set=static_mask_set,
                    latent_mask_pct=latent_mask_pct,
                    rng=rng,
                    mask_target_timestep=bool(self.use_attention),
                )
            )

        if not valid_indices:
            return

        all_windows = np.stack(windows_list)              # (N, seq_len, n_bins)
        all_targets = counts[valid_indices]                # (N, n_bins)

        has_any_mask = any(m is not None for m in masks_list)
        all_masks: Optional[np.ndarray] = None
        if has_any_mask:
            all_masks = np.zeros((len(valid_indices), self.seq_len), dtype=bool)
            for j, m in enumerate(masks_list):
                if m is not None:
                    all_masks[j] = m

        # --- Phase 2: batched forward passes ------------------------------
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(valid_indices), batch_size):
                end = min(start + batch_size, len(valid_indices))

                win_t = torch.from_numpy(all_windows[start:end]).to(self.device)
                tgt_t = torch.from_numpy(all_targets[start:end]).to(self.device)

                mask_t = None
                if all_masks is not None:
                    mask_t = torch.from_numpy(all_masks[start:end]).to(self.device)

                recon = self.model_(win_t, latent_timestep_mask=mask_t)
                scores = self._score_batch(tgt_t, recon)
                scores_np = scores.cpu().numpy()

                for j, idx in enumerate(valid_indices[start:end]):
                    metrics[idx] = float(scores_np[j])

    # ------------------------------------------------------------------
    # Sequential scoring (alarm-feedback: each score can mask later windows)
    # ------------------------------------------------------------------

    def _score_sequential(
        self,
        counts: np.ndarray,
        metrics: np.ndarray,
        masked_index_set: Optional[Set[int]],
        latent_mask_pct: float,
        mask_seed: Optional[int],
        threshold_for_feedback: Optional[float],
    ) -> None:
        """One-at-a-time scoring with alarm-feedback masking."""
        rng = np.random.default_rng(mask_seed) if latent_mask_pct > 0.0 else None
        dynamic_masked_index_set: Set[int] = set(masked_index_set or set())

        self.model_.eval()
        with torch.no_grad():
            for i in range(counts.shape[0]):
                window = self._build_window_for_index(counts, i)
                if window is None:
                    continue

                window_indices = self._window_indices_for_end(i)
                latent_mask_np = self._build_latent_mask_for_window(
                    window_indices=window_indices,
                    masked_index_set=dynamic_masked_index_set,
                    latent_mask_pct=latent_mask_pct,
                    rng=rng,
                    mask_target_timestep=bool(self.use_attention),
                )

                window_tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)
                target_tensor = torch.from_numpy(counts[i]).unsqueeze(0).to(self.device)
                latent_mask_tensor = None
                if latent_mask_np is not None:
                    latent_mask_tensor = (
                        torch.from_numpy(latent_mask_np)
                        .unsqueeze(0)
                        .to(self.device)
                    )

                reconstruction = self.model_(
                    window_tensor,
                    latent_timestep_mask=latent_mask_tensor,
                )
                score = self._score_batch(target_tensor, reconstruction)
                metrics[i] = float(score.item())

                if (
                    threshold_for_feedback is not None
                    and np.isfinite(metrics[i])
                    and metrics[i] > threshold_for_feedback
                ):
                    dynamic_masked_index_set.add(int(i))

    def process_time_series(
        self,
        time_series,
        mask_indices: Optional[Sequence[int]] = None,
        latent_mask_pct: float = 0.0,
        mask_seed: Optional[int] = None,
        mask_alarm_feedback: bool = False,
    ) -> np.ndarray:
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

        scores = self.score_time_series(
            time_series,
            mask_indices=mask_indices,
            latent_mask_pct=latent_mask_pct,
            mask_seed=mask_seed,
            mask_alarm_feedback=mask_alarm_feedback,
            feedback_threshold=self.threshold,
        )
        times = self._extract_timestamps(time_series)

        self._aggregate_alarms_from_scores(
            scores=scores,
            times=times,
            threshold=float(self.threshold),
        )

        return scores

    def _aggregate_alarms_from_scores(
        self,
        scores: np.ndarray,
        times: np.ndarray,
        threshold: float,
    ) -> None:
        """Build detector alarm intervals from precomputed scores and timestamps."""
        self.reset_alarms()

        for score, t in zip(scores, times):
            if np.isnan(score):
                continue

            if score > threshold:
                if not self._is_alarming:
                    self._start_alarm(t, float(score))
                else:
                    self._update_alarm_peak(t, float(score))
            else:
                if self._is_alarming:
                    self._end_alarm(t)

        if self._is_alarming:
            self._end_alarm(float(times[-1]))

    def set_threshold_by_far(
        self,
        background_data,
        alarms_per_hour: float = 1.0,
        max_iterations: int = 20,
        verbose: bool = False,
        mask_indices: Optional[Sequence[int]] = None,
        latent_mask_pct: float = 0.0,
        mask_seed: Optional[int] = None,
        mask_alarm_feedback: bool = False,
    ) -> float:
        """Set threshold to achieve target FAR with optional latent masking."""
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before setting threshold.")

        # During calibration bootstrap, threshold may be unset. Alarm-feedback
        # masking requires a threshold, so bootstrap score stats without feedback
        # and then enable it inside the threshold search loop.
        bootstrap_feedback = bool(mask_alarm_feedback and self.threshold is not None)
        scores = self.score_time_series(
            background_data,
            mask_indices=mask_indices,
            latent_mask_pct=latent_mask_pct,
            mask_seed=mask_seed,
            mask_alarm_feedback=bootstrap_feedback,
            feedback_threshold=self.threshold if bootstrap_feedback else None,
        )
        finite_scores = scores[np.isfinite(scores)]
        if finite_scores.size == 0:
            raise ValueError(
                "No finite scores available for threshold calibration. "
                "Check model outputs and warmup handling."
            )

        times = self._extract_timestamps(background_data)
        total_time_seconds = 0.0
        if len(times) > 1 and np.all(np.isfinite(times)):
            dt = np.diff(times)
            dt = dt[np.isfinite(dt) & (dt > 0)]

            window_width = None
            integration_time = getattr(background_data, "integration_time", None)
            if integration_time is not None and np.isfinite(integration_time):
                window_width = float(integration_time)
            elif dt.size > 0:
                window_width = float(np.median(dt))

            if window_width is not None:
                total_time_seconds = float(times[-1] - times[0] + max(window_width, 0.0))

        if total_time_seconds <= 0:
            total_time_seconds = float(np.sum(background_data.real_times))

        total_time_hours = total_time_seconds / 3600.0
        if total_time_hours <= 0:
            raise ValueError(f"Invalid observation time: {total_time_hours} hours")

        low_threshold = float(np.min(finite_scores))
        high_threshold = float(np.max(finite_scores)) * 1.5

        best_threshold = float(
            np.percentile(
                finite_scores,
                max(0.1, min(99.9, 100 * (1 - alarms_per_hour / (60 * len(finite_scores))))),
            )
        )
        best_far_diff = float("inf")
        best_observed_far = 0.0

        if verbose:
            print(
                f"Calibrating threshold for {alarms_per_hour:.2f} alarms/hour...\n"
                f"  Background: {len(finite_scores)} finite scores (of {len(scores)}) over {total_time_hours:.2f} hours\n"
                f"  Score range: [{finite_scores.min():.4f}, {finite_scores.max():.4f}]\n"
                f"  Score mean +/- std: {finite_scores.mean():.4f} +/- {finite_scores.std():.4f}"
            )

        for iteration in range(max_iterations):
            test_threshold = (low_threshold + high_threshold) / 2
            self.threshold = test_threshold

            if mask_alarm_feedback:
                self.process_time_series(
                    background_data,
                    mask_indices=mask_indices,
                    latent_mask_pct=latent_mask_pct,
                    mask_seed=mask_seed,
                    mask_alarm_feedback=mask_alarm_feedback,
                )
            else:
                # No feedback dependency: threshold search can reuse precomputed scores.
                self._aggregate_alarms_from_scores(
                    scores=scores,
                    times=times,
                    threshold=float(test_threshold),
                )
            n_alarms = len(self.alarms)
            observed_far = n_alarms / total_time_hours

            far_diff = abs(observed_far - alarms_per_hour)

            is_better = False
            if far_diff < best_far_diff:
                is_better = True
            elif far_diff == best_far_diff:
                if observed_far > best_observed_far:
                    is_better = True
                elif observed_far == best_observed_far:
                    is_better = test_threshold < best_threshold

            if is_better:
                best_far_diff = far_diff
                best_threshold = test_threshold
                best_observed_far = observed_far

            if verbose:
                print(
                    f"  Iter {iteration + 1}: threshold={test_threshold:.6f} "
                    f"-> {n_alarms} alarms ({observed_far:.2f}/hr)"
                )

            if observed_far > alarms_per_hour:
                low_threshold = test_threshold
            else:
                high_threshold = test_threshold

            if (
                far_diff < 0.1 * alarms_per_hour
                or (high_threshold - low_threshold) < 1e-8
            ):
                if verbose:
                    print(f"  Converged after {iteration + 1} iterations")
                break

        self.threshold = best_threshold

        if mask_alarm_feedback:
            self.process_time_series(
                background_data,
                mask_indices=mask_indices,
                latent_mask_pct=latent_mask_pct,
                mask_seed=mask_seed,
                mask_alarm_feedback=mask_alarm_feedback,
            )
        else:
            self._aggregate_alarms_from_scores(
                scores=scores,
                times=times,
                threshold=float(self.threshold),
            )
        final_far = len(self.alarms) / total_time_hours

        if verbose:
            print(
                f"\n  Threshold set: {self.threshold:.6f}\n"
                f"  Achieved FAR: {final_far:.2f} alarms/hour "
                f"({len(self.alarms)} alarms)\n"
                f"  Target FAR: {alarms_per_hour:.2f} alarms/hour"
            )

        return self.threshold

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
            use_attention=self.use_attention,
        )

        payload = {
            "model_state": self.model_.state_dict(),
            "model_config": asdict(model_config),
            "seq_len": self.seq_len,
            "seq_stride": self.seq_stride,
            "loss_type": self.loss_type,
            "threshold": self.threshold,
            "aggregation_gap": self.aggregation_gap,
            "use_attention": self.use_attention,
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
        self.use_attention = bool(payload.get("use_attention", cfg.get("use_attention", self.use_attention)))

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
            use_attention=self.use_attention,
        ).to(self.device)
        self.model_.load_state_dict(payload["model_state"])
        self.model_.eval()

        if self.verbose:
            print(f"Temporal model loaded from {path}")

    def __repr__(self) -> str:
        return (
            f"LSTMTemporalDetector(seq_len={self.seq_len}, "
            f"seq_stride={self.seq_stride}, "
            f"use_attention={self.use_attention}, "
            f"loss_type='{self.loss_type}', "
            f"trained={self.is_trained}, alarms={len(self.alarms)})"
        )
