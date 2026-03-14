"""Tests for LSTMTemporalDetector."""

import numpy as np
import pytest

from gammaflow import Spectrum, SpectralTimeSeries


torch = pytest.importorskip("torch")

from gammaflow.algorithms import LSTMTemporalDetector


@pytest.fixture
def temporal_counts() -> np.ndarray:
    """Small deterministic count matrix."""
    rng = np.random.default_rng(123)
    return rng.poisson(lam=20, size=(8, 16)).astype(float)


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
    spectrum = Spectrum(np.ones(16), real_time=1.0)

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
    assert loaded.loss_type == detector.loss_type
    assert loaded.threshold == detector.threshold
    assert np.allclose(before, after, equal_nan=True)
