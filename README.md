# GammaFlow

A Python library for working with time series of gamma ray spectra and listmode data.

## Note on Similar Projects

This project (`gammaflow`) should not be confused with [GAMMA_FLOW](https://github.com/ElsevierSoftwareX/SOFTX-D-25-00403), which is a separate tool developed by a different team (no affiliation with us) that is focused on isotope classification via matrix factorization (published in Elsevier SoftwareX). 

**GammaFlow** provides comprehensive data structures and time-series analysis infrastructure for gamma-ray spectroscopy (similar to pandas/numpy for radiation detection), while **GAMMA_FLOW** is a classification-focused application for multi-label spectral analysis. Both projects serve different needs in the gamma spectroscopy community.

## Features

### Core Classes

- **Spectrum**: Comprehensive base class for gamma ray spectra
  - Optional energy calibration (uncalibrated = channel mode)
  - Arithmetic operations (add, subtract, scalar multiply/divide)
  - Energy operations (rebinning, calibration, slicing)
  - Analysis methods (integration, normalization)
  - Uncertainty propagation (Poisson statistics by default)
  - Optional live_time (falls back to real_time for count rates)
  
- **Spectra**: Base class for collections of spectra (not time-ordered)
  - Manages shared or independent energy calibration
  - Vectorized access to counts via 2D NumPy array
  - Statistical methods (mean, median, std, sum spectra)
  - Automatic detection of compatible calibrations
  - Individual Spectrum objects with metadata preserved
  - NumPy array protocol support for advanced operations

- **SpectralTimeSeries**: Time-ordered collection of spectra (inherits from Spectra)
  - All Spectra features plus time-series specific functionality
  - Auto-detection of integration_time and stride_time
  - Reintegration to coarser time scales (multiples only)
  - Create from 2D arrays or list mode data
  - Time-based slicing and filtering
  - Copy-on-write for safe modifications

- **ListMode**: Event-by-event data handling
  - Encapsulates time deltas and energies
  - Filtering and slicing operations
  - Conversion to SpectralTimeSeries

### Operations

- **Energy ROIs (Regions of Interest)**
  - Define labeled energy windows for analysis
  - Rebin spectra by integrating over ROIs
  - Support for overlapping and non-consecutive ROIs
  - Track creation method (manual, peak search, censored windows, etc.)
  - Time series analysis with ROIs

### Detection Algorithms

All detectors inherit from `BaseDetector`, which provides a shared interface for
training, threshold calibration (via false-alarm rate), time-series processing,
and alarm aggregation. Detected anomalies are returned as `AlarmEvent` objects.

- **SADDetector** (Spectral Anomaly Detection)
  - PCA-based reconstruction error
  - Learns a low-dimensional subspace from background spectra
  - Anomaly score: squared residual after projection
  - Optional per-spectrum normalization

- **ARADDetector** (Autoencoder Reconstruction Anomaly Detection, requires PyTorch)
  - 1-D convolutional autoencoder trained on background spectra
  - Scoring via Jensen-Shannon Divergence or Chi-squared reconstruction error
  - Configurable architecture (latent dim, dropout, kernel sizes)
  - GPU-accelerated training and inference

- **KSigmaDetector**
  - Streaming detector with online background estimation
  - No offline training required; background statistics update incrementally
  - Background is frozen during alarm periods
  - Threshold expressed as number of standard deviations

- **Censored Energy Window (CEW)**
  - Gradient-based optimization for optimal energy window selection
  - Maximizes signal-to-noise ratio for source detection
  - Ridge regression predictor with constrained coefficients
  - Score spectra for anomaly detection or source presence
  - Convert optimized windows to EnergyROI objects
  - Serializable predictors for deployment

### Datasets

Built-in loaders for standard gamma-ray spectroscopy benchmark datasets.

- **TopCoderDataset** -- TopCoder Urban Radiation Search challenge
  - Loads list-mode CSV data and answer keys
  - Splits: `training`, `testing`, `leaderboard`
  - Returns `SpectralTimeSeries` via configurable energy binning and integration time

- **APLStarterKitDataset** -- APL Starter Kit dataset (AIPT/DRAG)
  - Loads pre-binned `.open` spectral files
  - Multi-detector support (selectable gamma element index)
  - Splits: `background`, `source`
  - Filtering by `is-active` and `is-in-zone` flags
  - Returns `SpectralTimeSeries` with per-sample metadata

### Visualization

- **Publication-quality plotting** with matplotlib and seaborn
  - `plot_spectrum()`: Single spectrum with log scale, uncertainty bands
  - `plot_count_rate_time_series()`: Total count rate vs time
  - `plot_waterfall()`: 2D heatmap of spectral evolution (time vs energy)
  - `plot_roi_time_series()`: Time evolution of specific energy ROIs
  - `plot_spectrum_comparison()`: Compare multiple spectra
  - Flexible styling and customization options

### Performance

- **Vectorized Operations**: Leverage NumPy for performance
  - Direct array access via `.counts` property
  - Broadcasting and fancy indexing support
  - Efficient batch processing

## Installation

```bash
pip install -e .
```

## Quick Start

### Basic Spectrum Operations

```python
import numpy as np
from gammaflow import Spectrum, SpectralTimeSeries

# Create a spectrum
counts = np.random.poisson(100, size=1024)
spectrum = Spectrum(counts, real_time=10.0)

# Apply energy calibration
energy_edges = np.linspace(0, 3000, 1025)  # 0-3000 keV
calibrated = Spectrum(counts, energy_edges=energy_edges)

# Arithmetic operations
combined = spectrum1 + spectrum2
scaled = spectrum * 2.0

# Energy operations
roi = spectrum.slice_energy(e_min=500, e_max=1000)
integrated = spectrum.integrate(e_min=600, e_max=700)
normalized = spectrum.normalize(method='area')

# Count rate (uses live_time if available, else real_time)
rate = spectrum.count_rate
```

### Time Series Creation

```python
# Method 1: From list of Spectrum objects
spectra = [Spectrum(np.random.poisson(100, 1024), real_time=1.0) for _ in range(100)]
ts = SpectralTimeSeries(spectra)

# Method 2: From 2D array (most common)
counts = np.random.poisson(100, size=(100, 1024))
timestamps = np.arange(100) * 1.0
real_times = np.ones(100) * 1.0
energy_edges = np.linspace(0, 3000, 1025)

ts = SpectralTimeSeries.from_array(
    counts,
    energy_edges=energy_edges,
    timestamps=timestamps,
    real_times=real_times
)

# Method 3: From list mode data
from gammaflow import ListMode

time_deltas = np.random.exponential(0.001, size=100000)
energies = np.random.gamma(2, 500, size=100000)

# Direct from arrays
ts = SpectralTimeSeries.from_list_mode(
    time_deltas, energies,
    integration_time=10.0,
    stride_time=10.0,
    energy_bins=1024
)

# Or using ListMode object (for filtering first)
listmode = ListMode(time_deltas, energies)
filtered = listmode.filter_energy(e_min=200, e_max=800)
ts = SpectralTimeSeries.from_list_mode(filtered, integration_time=10.0)
```

### Time Series Operations

```python
# Timing is auto-detected from data
print(f"Integration time: {ts.integration_time}")  # Auto-detected from real_time
print(f"Stride time: {ts.stride_time}")            # Auto-detected from timestamps

# Reintegrate to coarser time resolution (must be even multiple)
ts_20s = ts.reintegrate(new_integration_time=20.0)  # 2x coarser
ts_40s = ts.reintegrate(new_integration_time=40.0)  # 4x coarser

# Vectorized operations
background = np.mean(ts.counts, axis=0)
ts.counts[:] -= background  # In-place modification

# Time slicing
ts_subset = ts.slice_time(t_min=100, t_max=500)

# Object-oriented access
for spec in ts:
    if spec.timestamp > 200:
        spec.metadata['processed'] = True
```

### ROI Analysis

```python
from gammaflow.operations import EnergyROI, rebin_spectrum_rois, rebin_time_series_rois

# Define regions of interest
rois = [
    EnergyROI(e_min=655, e_max=668, label="Cs-137 (661.7 keV)"),
    EnergyROI(e_min=1450, e_max=1470, label="K-40 (1460.8 keV)"),
    EnergyROI(e_min=1165, e_max=1180, label="Co-60 Peak 1"),
    EnergyROI(e_min=1325, e_max=1340, label="Co-60 Peak 2")
]

# Integrate single spectrum over ROIs
roi_counts, labels = rebin_spectrum_rois(spectrum, rois, return_labels=True)
print(f"Cs-137 counts: {roi_counts[0]}")

# Analyze time evolution of ROIs
roi_ts = rebin_time_series_rois(time_series, rois)
# Shape: (n_spectra, n_rois) - rows are time, columns are ROIs

# Plot K-40 evolution
import matplotlib.pyplot as plt
plt.plot(time_series.timestamps, roi_ts[:, 1])
plt.xlabel("Time (s)")
plt.ylabel("K-40 Peak Counts")
```

### Working with Spectra Collections

```python
from gammaflow import Spectra

# Create a collection of spectra (not necessarily time-ordered)
spectra_list = [
    Spectrum(np.random.poisson(100, 1024), metadata={'sample': 'background'}),
    Spectrum(np.random.poisson(120, 1024), metadata={'sample': 'source_1'}),
    Spectrum(np.random.poisson(110, 1024), metadata={'sample': 'source_2'})
]

collection = Spectra(spectra_list)

# Vectorized operations on all spectra
background_subtracted = collection.counts - collection.counts[0]

# Statistical analysis
mean_spectrum = collection.mean_spectrum()
median_spectrum = collection.median_spectrum()
std_spectrum = collection.std_spectrum()
total_spectrum = collection.sum_spectrum()

# Access individual spectra with metadata
for i, spec in enumerate(collection):
    print(f"Sample {spec.metadata['sample']}: {np.sum(spec.counts)} counts")

# NumPy protocol support
stacked = np.array(collection)  # (n_spectra, n_bins)
normalized = collection.counts / np.sum(collection.counts, axis=1, keepdims=True)
```

### Censored Energy Window (CEW) Algorithm

```python
from gammaflow.algorithms import optimize_cew_windows, fit_cew_predictor, CEWPredictor

# Load training data: background spectra
background_spectra = Spectra([...])  # Your background training data

# Define expected source spectrum (or use measured/simulated)
source_counts = np.loadtxt('cs137_spectrum.txt')
source_spectrum = Spectrum(source_counts)

# Get mean background for window optimization
mean_background = background_spectra.mean_spectrum()

# Step 1: Optimize window selection to maximize SNR
window_mask = optimize_cew_windows(
    source_spectrum=source_spectrum,
    background_spectrum=mean_background,
    method='gradient',  # Options: 'gradient', 'greedy', 'scipy'
    learning_rate=0.01,
    max_iterations=1000
)

print(f"Window bins: {np.sum(window_mask)} of {len(window_mask)}")

# Step 2: Fit ridge regression predictor
predictor = fit_cew_predictor(
    background_spectra=background_spectra,
    window_mask=window_mask,
    alpha=1.0  # Ridge regularization
)

# Step 3: Score new observations
test_spectrum = Spectrum(np.random.poisson(100, 1024))
score = predictor.score(test_spectrum)
print(f"CEW score: {score:.3f}")  # Higher = more anomalous

# Score time series for anomaly detection
# (monitoring for when a source appears)
time_series = SpectralTimeSeries(...)  # Time-ordered measurements
scores = predictor.score_time_series(time_series)

# Detect anomalies (times when source appeared)
threshold = np.percentile(scores, 99)  # 99th percentile
anomaly_indices = np.where(scores > threshold)[0]
anomaly_times = time_series.timestamps[anomaly_indices]
print(f"Source detected at {len(anomaly_indices)} time points")
print(f"Detection times: {anomaly_times}")

# Convert optimized windows to ROIs
rois = predictor.to_rois(energy_edges=test_spectrum.energy_edges)
for roi in rois:
    print(f"ROI {roi.label}: {roi.e_min:.1f} - {roi.e_max:.1f} keV")

# Save predictor for later use
predictor_dict = predictor.to_dict()
np.savez('cew_predictor.npz', **predictor_dict)

# Load and use
loaded_dict = np.load('cew_predictor.npz')
loaded_predictor = CEWPredictor.from_dict(loaded_dict)
```

### Advanced Features

```python
# Overlapping ROIs (useful for peak fitting)
overlapping_rois = [
    EnergyROI(e_min=650, e_max=670, label="Peak + Background"),
    EnergyROI(e_min=655, e_max=665, label="Peak Core")
]

# Non-consecutive ROIs (gaps allowed - perfect for peaks)
peak_rois = [
    EnergyROI(e_min=661, e_max=663, label="Cs-137"),
    EnergyROI(e_min=1460, e_max=1462, label="K-40")  # Large gap
]

# Create ROI collection with metadata
from gammaflow.operations import create_roi_collection

censored = create_roi_collection([
    (100, 600, "Low Window"),
    (700, 1400, "Mid Window"),
    (1500, 1900, "High Window")
], method="Censored Energy Windows")

# NumPy integration for advanced analysis
total_counts = np.sum(ts.counts, axis=1)
spectral_std = np.std(ts.counts, axis=0)
peak_locations = np.argmax(ts.counts, axis=1)
```

### Visualization

```python
from gammaflow.visualization import (
    plot_spectrum,
    plot_count_rate_time_series,
    plot_waterfall,
    plot_roi_time_series,
    plot_spectrum_comparison,
)
import matplotlib.pyplot as plt

# Plot a single spectrum (log scale, with uncertainty)
fig, ax = plot_spectrum(spectrum, log_y=True, show_uncertainty=True)
plt.show()

# Plot total count rate over time
fig, ax = plot_count_rate_time_series(time_series, show_uncertainty=True)
plt.show()

# Create waterfall plot (2D heatmap: time vs energy)
fig, ax = plot_waterfall(
    time_series,
    mode='count_rate',
    log_scale=True,
    cmap='viridis'
)
plt.show()

# Plot time evolution of specific ROIs
rois = [
    EnergyROI(655, 668, label="Cs-137"),
    EnergyROI(1450, 1470, label="K-40")
]
fig, ax = plot_roi_time_series(time_series, rois, mode='count_rate')
plt.show()

# Compare multiple spectra
fig, ax = plot_spectrum_comparison(
    [background, source],
    labels=['Background', 'Source'],
    log_y=True
)
plt.show()

# Create multi-panel figure
fig = plt.figure(figsize=(14, 10))

ax1 = plt.subplot(2, 2, 1)
plot_spectrum(spectrum, fig=fig, ax=ax1)
ax1.set_title('Spectrum')

ax2 = plt.subplot(2, 2, 2)
plot_count_rate_time_series(time_series, fig=fig, ax=ax2)
ax2.set_title('Count Rate vs Time')

ax3 = plt.subplot(2, 1, 2)
plot_waterfall(time_series, fig=fig, ax=ax3)
ax3.set_title('Spectral Evolution')

plt.tight_layout()
plt.show()
```

## Project Structure

```
gammaflow/
├── gammaflow/
│   ├── __init__.py
│   ├── core/
│   │   ├── calibration.py
│   │   ├── spectrum.py
│   │   ├── spectra.py
│   │   ├── time_series.py
│   │   └── listmode.py
│   ├── operations/
│   │   ├── energy.py
│   │   ├── temporal.py
│   │   └── roi.py
│   ├── algorithms/
│   │   ├── base.py              # BaseDetector, AlarmEvent
│   │   ├── sad.py               # SADDetector (PCA)
│   │   ├── arad.py              # ARADDetector (autoencoder, requires PyTorch)
│   │   ├── k_sigma.py           # KSigmaDetector (streaming)
│   │   └── censored_energy_window.py
│   ├── datasets/
│   │   ├── topcoder.py          # TopCoder Urban Radiation Search
│   │   └── apl_starter_kit.py   # APL Starter Kit (AIPT/DRAG)
│   ├── utils/
│   │   └── exceptions.py
│   └── visualization/
│       └── __init__.py
├── tests/
├── examples/
└── README.md
```

## Examples

See the `examples/` directory. Jupyter notebooks demonstrate end-to-end
detection workflows on real datasets; Python scripts cover core API usage.

### Detection Notebooks

- `sad_detection.ipynb` -- SAD detection on the TopCoder dataset
- `arad_detection.ipynb` -- ARAD detection on the TopCoder dataset
- `k_sigma_detection.ipynb` -- K-Sigma detection on the TopCoder dataset (full-spectrum and ROI-based)
- `sad_detection_apl_starter_kit.ipynb` -- SAD detection on the APL Starter Kit dataset
- `arad_cross_dataset.ipynb` -- ARAD trained on TopCoder, tested on APL Starter Kit

### Core API Scripts

- `basic_usage.py` -- Spectrum and time series basics
- `advanced_usage.py` -- Advanced operations and workflows
- `list_mode_example.py` -- Working with list mode data
- `listmode_class_example.py` -- ListMode class usage
- `reintegrate_example.py` -- Time resolution reintegration
- `timing_inference_example.py` -- Automatic timing detection
- `roi_example.py` -- Energy ROI analysis
- `visualization_example.py` -- Plotting spectra and time series

## Documentation

Documentation is integrated into docstrings throughout the codebase. For detailed API documentation, use Python's built-in help:

```python
from gammaflow import Spectrum, Spectra, SpectralTimeSeries
from gammaflow.algorithms import SADDetector, KSigmaDetector, BaseDetector
from gammaflow.datasets import TopCoderDataset, APLStarterKitDataset

help(SADDetector)
help(TopCoderDataset)
```

## Key Concepts

### Spectrum Collections
- **Spectra**: Base class for non-time-ordered collections
  - Shared or independent calibration modes
  - Statistical methods (mean, median, std, sum)
  - Individual metadata preservation
- **SpectralTimeSeries**: Time-ordered subclass of Spectra
  - All Spectra features plus time-specific functionality
  - Automatic timing parameter detection

### Energy Calibration
- Spectra can be calibrated (with energy_edges) or uncalibrated (channel mode)
- Shared calibration in collections for memory efficiency
- Auto-detection of compatible calibrations
- Copy-on-write for safe modifications

### Timing
- `real_time`: Clock time (including dead time)
- `live_time`: Actual counting time (excluding dead time)
- `integration_time`: Time window width for binning
- `stride_time`: Time between consecutive windows
- Auto-detection from consistent data

### ROIs (Regions of Interest)
- Define labeled energy windows for analysis
- Can overlap (for peak fitting, background estimation)
- Can be non-consecutive (for peak-only analysis)
- Track creation method and metadata

### List Mode
- Event-by-event data: time deltas + energies
- Filter and slice before binning
- Convert to time series with flexible windowing

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Current status: **367+ tests passing** ✓

Test coverage includes:
- Core functionality (Spectrum, Spectra, SpectralTimeSeries, ListMode)
- Energy operations (rebinning, calibration, ROIs)
- Time series operations (reintegration, slicing, filtering)
- Detection algorithms (SAD, ARAD, K-Sigma, CEW)
- Dataset loaders (TopCoder, APL Starter Kit)
- Shared calibration and copy-on-write behavior
- NumPy interoperability

## License

MIT

