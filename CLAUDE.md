# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_lstm_temporal.py

# Run a single test
pytest tests/test_lstm_temporal.py::test_function_name

# Run with coverage
pytest --cov=gammaflow

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Train ARAD from preprocessed data
python examples/train_arad_preprocessed.py --preprocessed-dir /path/to/runs --output-model model.pt

# Train LSTM from preprocessed data
python examples/train_lstm_temporal_preprocessed.py --preprocessed-dir /path/to/runs --output-model model.pt
```

## Architecture Overview

### Core Data Structures

**`Spectrum`** — a single gamma-ray energy spectrum (counts array + optional energy calibration). Supports arithmetic, slicing, rebinning, and Poisson uncertainty propagation.

**`SpectralTimeSeries`** — a time-ordered collection of spectra. Inherits from `Spectra`. Key operations: `reintegrate()` (coarser time resolution), `slice_time()`, `apply_to_each()`. Timestamps and live/real times are tracked per spectrum.

**`ListMode`** — event-by-event data (time deltas + energies) that can be converted to `SpectralTimeSeries`.

### Detection Algorithms

All detectors inherit from `BaseDetector` (gammaflow/algorithms/base.py):
- `fit(background_data)` — train on background
- `score_spectrum(spectrum)` → float
- `process_time_series(ts)` → (scores, alarms)
- `set_threshold_from_far(ts, far)` — calibrate threshold by false alarm rate
- Alarms are aggregated: consecutive threshold crossings within `aggregation_gap` seconds merge into one `AlarmEvent`

### ARAD (gammaflow/algorithms/arad.py)

CNN autoencoder for **per-spectrum** anomaly detection.

- **Encoder:** 5 Conv1d blocks (channels 1→8→8→8→8→8, kernel sizes 7/5/3/3/3, MaxPool after each → 32× spatial reduction) → flatten → Linear → latent vector
- **Decoder:** mirror with Upsample+Deconv blocks → Sigmoid output
- **Normalization:** input divided by per-sample max (Lᵢₙf) inside the encoder; decoder outputs [0, 1]
- **n_bins must be divisible by 32** (5 pooling layers)
- Spectra are converted to count rates (counts / live_time) before training
- Trained with AdamW + L1 regularization + ReduceLROnPlateau + early stopping
- Loss: JSD (Jensen-Shannon divergence) or Chi-squared reconstruction error
- Key methods: `score_spectrum()`, `reconstruct()`, `compute_saliency_map()`, `save()`/`load()`

### LSTM Temporal Detector (gammaflow/algorithms/lstm_temporal.py)

Extends ARAD to detect **temporal anomalies** — spectra that are unusual given their causal history.

- **Architecture:** same CNN spatial encoder as ARAD → LSTM over a window of encoded spectra → optional attention → same CNN spatial decoder
- **Input:** a causal window of `seq_len` spectra at stride `seq_stride`; only the final spectrum is reconstructed and scored
- **Warmup:** the first `(seq_len - 1) * seq_stride` spectra return `np.nan` (insufficient history)
- **Optional attention:** query = final LSTM timestep, keys/values = all LSTM outputs; `latent_timestep_mask` prevents identity shortcut
- Training is performed externally via `train_lstm_temporal_from_preprocessed()` in the pipeline module; `fit()` accepts a `trainer_fn` callable
- Key difference from ARAD: LSTM can score a spectrum as anomalous if it deviates from expected temporal evolution, even if it looks normal in isolation

### Training Pipeline (gammaflow/training/lstm_temporal_pipeline.py)

`train_lstm_temporal_from_preprocessed()` handles the full LSTM training loop:
- Loads preprocessed run files (`run*.pt`, each a dict with key `'spectra'` of shape `(n_spectra, n_bins)`)
- `PreprocessedTemporalWindowDataset` builds causal `(window, target)` pairs on-the-fly with file caching
- Features: ReduceLROnPlateau scheduling, gradient clipping, early stopping, latent timestep masking, per-epoch callback, best checkpoint save
- Outputs: trained `LSTMTemporalDetector`, training history dict, `metrics.json`

### Loss Functions (gammaflow/training/losses.py)

Shared by ARAD and LSTM:
- **JSD:** L1-normalizes both target and reconstruction, computes symmetric Jensen-Shannon divergence per sample
- **Chi-squared:** denormalizes reconstruction by `max(target)`, computes Pearson chi-squared per sample
- `get_loss_fn(name)` returns a batch-mean scalar loss; `score_batch()` returns per-sample scores

### Data Flow

```
Raw counts/listmode
  → SpectralTimeSeries (with calibration, timestamps, live_times)
  → detector.fit(background_ts)          # train on background
  → detector.process_time_series(test_ts) # get scores + AlarmEvents
```

Preprocessed datasets are stored as `.pt` files (PyTorch tensors), one file per acquisition run, loaded by the pipeline's dataset class.
