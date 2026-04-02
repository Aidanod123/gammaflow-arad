"""Tests for LSTMTemporalDetector."""

import numpy as np
import pytest

from gammaflow import Spectrum, SpectralTimeSeries


torch = pytest.importorskip("torch")

from gammaflow.algorithms import LSTMTemporalDetector
from gammaflow.algorithms.lstm_temporal import TemporalLSTMAutoencoder


@pytest.fixture
def temporal_counts() -> np.ndarray:
    """Small deterministic count matrix."""
    rng = np.random.default_rng(123)
    return rng.poisson(lam=20, size=(8, 64)).astype(float)


@pytest.fixture
def temporal_series(temporal_counts) -> SpectralTimeSeries:
    """Time series built from deterministic counts."""
    spectra = []
    for i, row in enumerate(temporal_counts):
        spectra.append(
            Spectrum(
                row,
                timestamp=float(i),
                real_time=1.0,
            )
        )
    return SpectralTimeSeries(spectra)


def test_invalid_sequence_parameters_raise() -> None:
    """Invalid temporal settings should fail fast."""
    with pytest.raises(ValueError, match="seq_len"):
        LSTMTemporalDetector(seq_len=1, verbose=False)

    with pytest.raises(ValueError, match="seq_stride"):
        LSTMTemporalDetector(seq_stride=0, verbose=False)


def test_score_spectrum_requires_temporal_context() -> None:
    """Single-spectrum scoring should be blocked for temporal detector."""
    detector = LSTMTemporalDetector(verbose=False)
    spectrum = Spectrum(np.ones(64), real_time=1.0)

    with pytest.raises(RuntimeError, match="requires temporal context"):
        detector.score_spectrum(spectrum)


def test_score_time_series_respects_warmup_window(temporal_series) -> None:
    """Warmup samples should be NaN and later scores should be finite."""
    detector = LSTMTemporalDetector(seq_len=3, seq_stride=1, threshold=100.0, verbose=False)
    detector.initialize_model(n_bins=temporal_series.n_bins)

    # Make outputs deterministic for reproducible assertions.
    for param in detector.model_.parameters():
        param.data.zero_()

    scores = detector.score_time_series(temporal_series)

    assert scores.shape == (temporal_series.n_spectra,)
    assert np.all(np.isnan(scores[: detector.warmup_samples]))
    assert np.all(np.isfinite(scores[detector.warmup_samples :]))


def test_process_time_series_requires_threshold(temporal_series) -> None:
    """Threshold must be set before alarm processing."""
    detector = LSTMTemporalDetector(seq_len=3, seq_stride=1, threshold=None, verbose=False)
    detector.initialize_model(n_bins=temporal_series.n_bins)

    with pytest.raises(RuntimeError, match="Threshold not set"):
        detector.process_time_series(temporal_series)


def test_save_and_load_roundtrip(tmp_path, temporal_series) -> None:
    """Persisted detector should reproduce identical scores."""
    detector = LSTMTemporalDetector(
        seq_len=4,
        seq_stride=1,
        use_attention=True,
        threshold=1.23,
        loss_type="chi2",
        verbose=False,
    )
    detector.initialize_model(n_bins=temporal_series.n_bins)

    for param in detector.model_.parameters():
        param.data.normal_(mean=0.0, std=0.01)

    before = detector.score_time_series(temporal_series)

    path = tmp_path / "temporal_model.pt"
    detector.save(str(path))

    loaded = LSTMTemporalDetector(verbose=False)
    loaded.load(str(path))
    after = loaded.score_time_series(temporal_series)

    assert loaded.seq_len == detector.seq_len
    assert loaded.seq_stride == detector.seq_stride
    assert loaded.use_attention == detector.use_attention
    assert loaded.loss_type == detector.loss_type
    assert loaded.threshold == detector.threshold
    assert np.allclose(before, after, equal_nan=True)


def test_temporal_autoencoder_applies_mask_after_encoder() -> None:
    """Forward masking should zero encoded timesteps before LSTM only."""
    model = TemporalLSTMAutoencoder(
        n_bins=64,
        latent_dim=4,
        lstm_hidden_dim=6,
        lstm_layers=1,
        dropout=0.0,
    )
    model.eval()

    windows = torch.rand(2, 5, 64)
    timestep_mask = torch.zeros(2, 5, dtype=torch.bool)
    timestep_mask[:, 1] = True
    timestep_mask[:, 3] = True

    with torch.no_grad():
        b, s, _ = windows.shape
        encoded = model.encoder(model._normalize_input(windows.reshape(b * s, model.n_bins)))
        encoded = encoded.view(b, s, model.latent_dim)
        assert torch.any(encoded[:, 1, :] != 0.0)

        masked_encoded = encoded.masked_fill(timestep_mask.unsqueeze(-1), 0.0)
        temporal_out, _ = model.temporal_core(masked_encoded)
        decoder_latent = model.temporal_to_latent(temporal_out[:, -1, :])
        decoded_linear = model.decoder_linear(decoder_latent)
        decoded_linear = decoded_linear.view(b, 8, model.n_bins // 32)
        expected = model.decoder(decoded_linear).view(b, model.n_bins)

        observed = model(windows, latent_timestep_mask=timestep_mask)

    assert torch.allclose(expected, observed, atol=1e-6)


def test_temporal_autoencoder_attention_masking_matches_manual() -> None:
    """Attention path should honor timestep masking in attention weights."""
    model = TemporalLSTMAutoencoder(
        n_bins=64,
        latent_dim=4,
        lstm_hidden_dim=6,
        lstm_layers=1,
        dropout=0.0,
        use_attention=True,
    )
    model.eval()

    windows = torch.rand(2, 5, 64)
    timestep_mask = torch.zeros(2, 5, dtype=torch.bool)
    timestep_mask[:, -1] = True
    timestep_mask[:, 1] = True

    with torch.no_grad():
        b, s, _ = windows.shape
        encoded = model.encoder(model._normalize_input(windows.reshape(b * s, model.n_bins)))
        encoded = encoded.view(b, s, model.latent_dim)
        masked_encoded = encoded.masked_fill(timestep_mask.unsqueeze(-1), 0.0)
        temporal_out, _ = model.temporal_core(masked_encoded)

        query = temporal_out[:, -1, :].unsqueeze(1)
        scores = torch.sum(query * temporal_out, dim=-1)
        scores = scores.masked_fill(timestep_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        context = torch.sum(temporal_out * attn.unsqueeze(-1), dim=1)

        decoder_latent = model.temporal_to_latent(context)
        decoded_linear = model.decoder_linear(decoder_latent)
        decoded_linear = decoded_linear.view(b, 8, model.n_bins // 32)
        expected = model.decoder(decoded_linear).view(b, model.n_bins)

        observed = model(windows, latent_timestep_mask=timestep_mask)

    assert torch.allclose(expected, observed, atol=1e-6)


def test_score_time_series_no_mask_pct_matches_default(temporal_series) -> None:
    """Explicit mask_pct=0 should preserve default scoring behavior."""
    detector = LSTMTemporalDetector(seq_len=4, seq_stride=1, verbose=False)
    detector.initialize_model(n_bins=temporal_series.n_bins)

    for param in detector.model_.parameters():
        param.data.normal_(mean=0.0, std=0.02)

    baseline = detector.score_time_series(temporal_series)
    explicit_zero = detector.score_time_series(
        temporal_series,
        latent_mask_pct=0.0,
        mask_seed=123,
    )

    assert np.allclose(baseline, explicit_zero, equal_nan=True)


def test_score_time_series_random_mask_is_seeded(temporal_series) -> None:
    """Random latent masking should be reproducible for a fixed seed."""
    detector = LSTMTemporalDetector(seq_len=4, seq_stride=1, verbose=False)
    detector.initialize_model(n_bins=temporal_series.n_bins)

    for param in detector.model_.parameters():
        param.data.normal_(mean=0.0, std=0.02)

    scores_seed_1_a = detector.score_time_series(
        temporal_series,
        latent_mask_pct=0.5,
        mask_seed=7,
    )
    scores_seed_1_b = detector.score_time_series(
        temporal_series,
        latent_mask_pct=0.5,
        mask_seed=7,
    )
    scores_seed_2 = detector.score_time_series(
        temporal_series,
        latent_mask_pct=0.5,
        mask_seed=8,
    )

    assert np.allclose(scores_seed_1_a, scores_seed_1_b, equal_nan=True)
    assert not np.allclose(scores_seed_1_a, scores_seed_2, equal_nan=True)


def test_process_time_series_alarm_feedback_masks_future_history() -> None:
    """Alarm-feedback masking should alter downstream scores when enabled."""
    detector = LSTMTemporalDetector(
        seq_len=3,
        seq_stride=1,
        threshold=0.5,
        loss_type="chi2",
        verbose=False,
    )
    detector.initialize_model(n_bins=64)

    class _StubModel(torch.nn.Module):
        def forward(self, windows, latent_timestep_mask=None):
            if latent_timestep_mask is not None:
                windows = windows.masked_fill(latent_timestep_mask.unsqueeze(-1), 0.0)
            return windows[:, :-1, :].mean(dim=1)

    detector.model_ = _StubModel().to(detector.device)

    # Craft deterministic spectra where index 3 is a strong outlier. With
    # seq_len=3, if index 3 alarms, it appears in the history of later windows
    # and should be masked when alarm-feedback mode is enabled.
    counts = np.array(
        [
            [1.0] * 64,
            [1.0] * 64,
            [1.0] * 64,
            [10.0] * 64,
            [1.0] * 64,
            [1.0] * 64,
        ],
        dtype=float,
    )
    timestamps = np.arange(counts.shape[0], dtype=float)
    ts = SpectralTimeSeries.from_array(
        counts,
        timestamps=timestamps,
        real_times=np.ones(counts.shape[0], dtype=float),
    )

    scores_no_feedback = detector.process_time_series(ts, mask_alarm_feedback=False)
    scores_feedback = detector.process_time_series(ts, mask_alarm_feedback=True)

    # t=4 window history includes index 3. Without feedback, index 3 contributes;
    # with feedback enabled, index 3 is masked after alarming at t=3.
    assert np.isfinite(scores_no_feedback[3])
    assert scores_no_feedback[3] > detector.threshold
    assert np.isfinite(scores_no_feedback[4])
    assert np.isfinite(scores_feedback[4])
    assert scores_feedback[4] < scores_no_feedback[4]
