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
from typing import List, Optional, Sequence, Set
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
    target_scales: Optional["torch.Tensor"] = None,
    score_type: Optional[str] = None,
) -> "torch.Tensor":
    """Lazy wrapper that imports from training.losses on first call."""
    from gammaflow.training.losses import score_batch  # noqa: deferred to avoid circular import
    return score_batch(targets, recon, loss_type, target_scales=target_scales, score_type=score_type)


_VALID_OUTPUT_ACTIVATIONS = ("sigmoid", "softmax")


@dataclass
class TemporalModelConfig:
    """Configuration used to initialize the temporal model."""

    n_bins: int
    latent_dim: int
    lstm_hidden_dim: int
    lstm_layers: int
    dropout: float
    mask_target: bool
    output_activation: str = "sigmoid"
    count_rate_conditioning: bool = False


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
        use_attention: Optional[bool] = None,
        output_activation: str = "sigmoid",
        count_rate_conditioning: bool = False,
        crc_version: int = 2,
    ):
        super().__init__()

        self.n_bins = int(n_bins)
        self.latent_dim = int(latent_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.lstm_layers = int(lstm_layers)
        self.dropout = float(dropout)
        if use_attention is not None:
            warnings.warn(
                "TemporalLSTMAutoencoder.use_attention is deprecated and ignored; "
                "attention logic has been removed.",
                DeprecationWarning,
                stacklevel=2,
            )

        output_activation = str(output_activation).lower()
        if output_activation not in _VALID_OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"output_activation must be one of {_VALID_OUTPUT_ACTIVATIONS}, "
                f"got '{output_activation}'"
            )
        self.output_activation = output_activation
        self.count_rate_conditioning = bool(count_rate_conditioning)
        self.crc_version = int(crc_version) if self.count_rate_conditioning else None

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

        # When count_rate_conditioning is enabled, log1p(total_counts) is used
        # to condition the decoder.  Two versions:
        #   v1 (legacy): raw scalar concatenated directly → decoder_in = latent_dim + 1
        #   v2 (current): projected through nn.Linear(1,4) → decoder_in = latent_dim + 4
        if self.count_rate_conditioning:
            if self.crc_version == 1:
                _cr_extra_dim = 1
            else:
                _cr_extra_dim = 4
                self.cr_embed = nn.Linear(1, _cr_extra_dim)
        else:
            _cr_extra_dim = 0
        _decoder_in_dim = self.latent_dim + _cr_extra_dim
        self.decoder_linear = nn.Sequential(
            nn.Linear(_decoder_in_dim, 8 * self._downsampled_bins),
            nn.Mish(),
            nn.BatchNorm1d(8 * self._downsampled_bins),
        )

        if self.output_activation == "softmax":
            # Shared hidden decoder blocks (no output activation).
            self.decoder = nn.Sequential(
                ARADDecoderBlock(8, 8, 3, self.dropout),
                ARADDecoderBlock(8, 8, 3, self.dropout),
                ARADDecoderBlock(8, 8, 3, self.dropout),
                ARADDecoderBlock(8, 8, 5, self.dropout),
            )
            # Separate final layer — softmax is applied over bins in forward().
            self.output_upsample = nn.Upsample(scale_factor=2, mode="nearest")
            self.output_deconv = nn.ConvTranspose1d(8, 1, 7, padding=3)
        else:
            # Default sigmoid path — identical to original ARAD decoder.
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

    def forward(
        self,
        windows: torch.Tensor,
        latent_timestep_mask: Optional[torch.Tensor] = None,
        count_rate: Optional[torch.Tensor] = None,
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
        count_rate : torch.Tensor, optional
            Total counts per target spectrum, shape ``(batch,)``.  Only used
            when ``count_rate_conditioning=True``.  ``log1p`` is applied
            internally before concatenating to the latent vector.
        """
        batch_size, seq_len, _ = windows.shape
        flat_windows = windows.reshape(batch_size * seq_len, self.n_bins)
        encoded_flat = self.encoder(flat_windows.unsqueeze(1))
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
        final_embedding = temporal_out[:, -1, :]

        decoder_latent = self.temporal_to_latent(final_embedding)

        if self.count_rate_conditioning:
            if count_rate is not None:
                cr_raw = torch.log1p(
                    count_rate.float().to(decoder_latent.device)
                ).unsqueeze(-1)  # (batch, 1)
            else:
                # No count rate available — fill with zero (log1p(0) = 0).
                cr_raw = torch.zeros(
                    decoder_latent.shape[0], 1,
                    device=decoder_latent.device,
                    dtype=decoder_latent.dtype,
                )
            if self.crc_version == 1:
                # Legacy: concatenate scalar directly.
                decoder_latent = torch.cat([decoder_latent, cr_raw], dim=-1)
            else:
                # v2: project through learned embedding before concatenating.
                cr_emb = self.cr_embed(cr_raw)  # (batch, 4)
                decoder_latent = torch.cat([decoder_latent, cr_emb], dim=-1)

        decoded_linear = self.decoder_linear(decoder_latent)
        decoded_linear = decoded_linear.view(batch_size, 8, self._downsampled_bins)

        if self.output_activation == "softmax":
            x = self.decoder(decoded_linear)
            x = self.output_deconv(self.output_upsample(x))
            return torch.softmax(x.view(batch_size, self.n_bins), dim=-1)

        return self.decoder(decoded_linear).view(batch_size, self.n_bins)


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
    mask_target : bool
        If True, always mask the final timestep latent embedding.
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
        mask_target: bool = True,
        use_attention: Optional[bool] = None,
        threshold: Optional[float] = None,
        aggregation_gap: float = 2.0,
        loss_type: str = "jsd",
        output_activation: str = "sigmoid",
        count_rate_conditioning: bool = False,
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
        if use_attention is not None:
            warnings.warn(
                "use_attention is deprecated and ignored; attention logic has been removed. "
                "Use mask_target instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if bool(use_attention):
                mask_target = True
        self.mask_target = bool(mask_target)
        # Backward compatibility for scripts that still reference this field.
        self.use_attention = False
        self.loss_type = str(loss_type).lower()
        self.verbose = bool(verbose)

        output_activation = str(output_activation).lower()
        if output_activation not in _VALID_OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"output_activation must be one of {_VALID_OUTPUT_ACTIVATIONS}, "
                f"got '{output_activation}'"
            )
        self.output_activation = output_activation

        self.count_rate_conditioning = bool(count_rate_conditioning)

        if self.loss_type not in ("jsd", "chi2", "poisson"):
            raise ValueError(f"loss_type must be 'jsd', 'chi2', or 'poisson', got '{loss_type}'")

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
            output_activation=self.output_activation,
            count_rate_conditioning=self.count_rate_conditioning,
            crc_version=2,
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

    def _score_batch(
        self,
        targets: torch.Tensor,
        recon: torch.Tensor,
        target_scales: Optional[torch.Tensor] = None,
        score_type: Optional[str] = None,
    ) -> torch.Tensor:
        """Per-sample anomaly scores.  Delegates to ``gammaflow.training.losses``."""
        return _score_batch_fn(
            targets,
            recon,
            self.loss_type,
            target_scales=target_scales,
            score_type=score_type,
        )

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
        target_count_rates: Optional[Sequence[float]] = None,
        normalize_inputs_l1: bool = False,
        mask_indices: Optional[Sequence[int]] = None,
        latent_mask_pct: float = 0.0,
        mask_seed: Optional[int] = None,
        mask_alarm_feedback: bool = False,
        feedback_threshold: Optional[float] = None,
        inference_batch_size: int = 256,
        score_type: Optional[str] = None,
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
        normalize_inputs_l1 : bool
            If True, L1-normalize each input spectrum before building windows.
            Keep False when the preprocessed inputs are already L1-normalized.
        score_type : str or None
            Override the scoring metric independently of the training loss.
            When ``None`` (default), uses the model's ``loss_type``.
            Supported: ``"jsd"``, ``"corrected_jsd"``, ``"chi2"``,
            ``"normalized_chi2"``, ``"reduced_chi2"``.
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

        if bool(normalize_inputs_l1):
            counts_row_sums = np.sum(counts, axis=1, keepdims=True)
            counts_row_sums = np.maximum(counts_row_sums, 1e-10)
            counts = counts / counts_row_sums

        effective_score_type = score_type if score_type is not None else self.loss_type
        _needs_scales = ("chi2", "normalized_chi2", "corrected_jsd", "reduced_chi2", "combined")
        if effective_score_type in _needs_scales and target_count_rates is None:
            raise ValueError(
                f"target_count_rates are required for {effective_score_type} scoring. "
                "Pass one scalar per spectrum aligned with the input time series."
            )

        count_rate_array: Optional[np.ndarray] = None
        if target_count_rates is not None:
            count_rate_array = np.asarray(target_count_rates, dtype=np.float32)
            if count_rate_array.ndim != 1 or count_rate_array.shape[0] != counts.shape[0]:
                raise ValueError(
                    "target_count_rates must have shape (n_spectra,), "
                    f"got {count_rate_array.shape} for {counts.shape[0]} spectra"
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
                target_count_rates=count_rate_array,
                latent_mask_pct=float(latent_mask_pct),
                mask_seed=mask_seed,
                threshold_for_feedback=threshold_for_feedback,
                score_type=score_type,
            )
        else:
            self._score_batched(
                counts=counts,
                metrics=metrics,
                masked_index_set=masked_index_set,
                target_count_rates=count_rate_array,
                latent_mask_pct=float(latent_mask_pct),
                mask_seed=mask_seed,
                batch_size=max(1, int(inference_batch_size)),
                score_type=score_type,
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
        target_count_rates: Optional[np.ndarray],
        latent_mask_pct: float,
        mask_seed: Optional[int],
        batch_size: int,
        score_type: Optional[str] = None,
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
                    mask_target_timestep=bool(self.mask_target),
                )
            )

        if not valid_indices:
            return

        all_windows = np.stack(windows_list)              # (N, seq_len, n_bins)
        del windows_list
        all_targets = counts[valid_indices]                # (N, n_bins)
        all_target_scales = None if target_count_rates is None else target_count_rates[valid_indices]

        has_any_mask = any(m is not None for m in masks_list)
        all_masks: Optional[np.ndarray] = None
        if has_any_mask:
            all_masks = np.zeros((len(valid_indices), self.seq_len), dtype=bool)
            for j, m in enumerate(masks_list):
                if m is not None:
                    all_masks[j] = m
        del masks_list

        # --- Phase 2: batched forward passes ------------------------------
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(valid_indices), batch_size):
                end = min(start + batch_size, len(valid_indices))

                win_t = torch.from_numpy(all_windows[start:end]).to(self.device)
                tgt_t = torch.from_numpy(all_targets[start:end]).to(self.device)
                scale_t = None
                if all_target_scales is not None:
                    scale_t = torch.from_numpy(all_target_scales[start:end]).to(self.device)

                mask_t = None
                if all_masks is not None:
                    mask_t = torch.from_numpy(all_masks[start:end]).to(self.device)

                cr_t = scale_t if self.count_rate_conditioning else None
                recon = self.model_(win_t, latent_timestep_mask=mask_t, count_rate=cr_t)
                scores = self._score_batch(tgt_t, recon, target_scales=scale_t, score_type=score_type)
                scores_np = scores.cpu().numpy()
                del win_t, tgt_t, scale_t, mask_t, recon, scores

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
        target_count_rates: Optional[np.ndarray],
        latent_mask_pct: float,
        mask_seed: Optional[int],
        threshold_for_feedback: Optional[float],
        score_type: Optional[str] = None,
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
                    mask_target_timestep=bool(self.mask_target),
                )

                window_tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)
                target_tensor = torch.from_numpy(counts[i]).unsqueeze(0).to(self.device)
                scale_tensor = None
                if target_count_rates is not None:
                    scale_tensor = torch.tensor(
                        [float(target_count_rates[i])],
                        device=self.device,
                        dtype=target_tensor.dtype,
                    )
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
                    count_rate=scale_tensor if self.count_rate_conditioning else None,
                )
                score = self._score_batch(
                    target_tensor,
                    reconstruction,
                    target_scales=scale_tensor,
                    score_type=score_type,
                )
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
        target_count_rates: Optional[Sequence[float]] = None,
        normalize_inputs_l1: bool = False,
        mask_indices: Optional[Sequence[int]] = None,
        latent_mask_pct: float = 0.0,
        mask_seed: Optional[int] = None,
        mask_alarm_feedback: bool = False,
        score_type: Optional[str] = None,
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
            target_count_rates=target_count_rates,
            normalize_inputs_l1=normalize_inputs_l1,
            mask_indices=mask_indices,
            latent_mask_pct=latent_mask_pct,
            mask_seed=mask_seed,
            mask_alarm_feedback=mask_alarm_feedback,
            feedback_threshold=self.threshold,
            score_type=score_type,
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
        target_count_rates: Optional[Sequence[float]] = None,
        normalize_inputs_l1: bool = False,
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
            target_count_rates=target_count_rates,
            normalize_inputs_l1=normalize_inputs_l1,
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
                    target_count_rates=target_count_rates,
                    normalize_inputs_l1=normalize_inputs_l1,
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
            mask_target=self.mask_target,
            output_activation=self.output_activation,
            count_rate_conditioning=self.count_rate_conditioning,
        )

        payload = {
            "model_state": self.model_.state_dict(),
            "model_config": asdict(model_config),
            "seq_len": self.seq_len,
            "seq_stride": self.seq_stride,
            "loss_type": self.loss_type,
            "threshold": self.threshold,
            "aggregation_gap": self.aggregation_gap,
            "mask_target": self.mask_target,
            "use_attention": False,
            # v2: count_rate_conditioning uses nn.Linear(1,4) embedding instead
            # of raw scalar concatenation. Older checkpoints (no key or "v1")
            # used scalar injection and cannot be loaded with this code.
            "crc_version": 2 if self.count_rate_conditioning else None,
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
        # Backward compatibility: legacy checkpoints may only have use_attention.
        if "mask_target" in payload:
            self.mask_target = bool(payload["mask_target"])
        elif "mask_target" in cfg:
            self.mask_target = bool(cfg["mask_target"])
        else:
            self.mask_target = bool(payload.get("use_attention", cfg.get("use_attention", True)))

        self.n_bins_ = int(cfg["n_bins"])
        self.latent_dim = int(cfg["latent_dim"])
        self.lstm_hidden_dim = int(cfg["lstm_hidden_dim"])
        self.lstm_layers = int(cfg["lstm_layers"])
        self.dropout = float(cfg["dropout"])
        self.output_activation = str(
            cfg.get("output_activation", "sigmoid")
        ).lower()
        self.count_rate_conditioning = bool(
            cfg.get("count_rate_conditioning", False)
        )

        # Determine CRC architecture version from checkpoint metadata.
        # v1: legacy scalar concatenation (latent_dim + 1, no cr_embed layer)
        # v2: nn.Linear(1,4) embedding (latent_dim + 4, has cr_embed layer)
        # When crc_version is missing from the payload (model trained before
        # the key was added), infer from the state dict keys.
        if self.count_rate_conditioning:
            saved_version = payload.get("crc_version")
            if saved_version is not None:
                crc_version = int(saved_version)
            elif "cr_embed.weight" in payload["model_state"]:
                crc_version = 2
            else:
                crc_version = 1
        else:
            crc_version = 2  # irrelevant when CRC is off

        self.model_ = TemporalLSTMAutoencoder(
            n_bins=self.n_bins_,
            latent_dim=self.latent_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            output_activation=self.output_activation,
            count_rate_conditioning=self.count_rate_conditioning,
            crc_version=crc_version,
        ).to(self.device)
        self.model_.load_state_dict(payload["model_state"])
        self.model_.eval()

        if self.verbose:
            print(f"Temporal model loaded from {path}")

    def __repr__(self) -> str:
        return (
            f"LSTMTemporalDetector(seq_len={self.seq_len}, "
            f"seq_stride={self.seq_stride}, "
            f"mask_target={self.mask_target}, "
            f"loss_type='{self.loss_type}', "
            f"output_activation='{self.output_activation}', "
            f"count_rate_conditioning={self.count_rate_conditioning}, "
            f"trained={self.is_trained}, alarms={len(self.alarms)})"
        )
