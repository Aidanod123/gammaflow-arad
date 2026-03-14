"""
Algorithms for advanced spectral analysis.

This module provides implementations of detection algorithms for gamma-ray
spectroscopy, built on a common ``BaseDetector`` interface.

Detection algorithms:
- ``ARADDetector`` — autoencoder reconstruction anomaly detection
- ``SADDetector``  — PCA-based spectral anomaly detection
- ``KSigmaDetector`` — rolling background k-sigma detection

Base classes:
- ``BaseDetector`` — abstract base for all detection algorithms
- ``AlarmEvent``   — dataclass representing a detected anomaly event

Note: Some algorithms (like CEW) may be available in development
versions but not included in the released package.
"""

from gammaflow.algorithms.base import AlarmEvent, BaseDetector
from gammaflow.algorithms.sad import SADDetector
from gammaflow.algorithms.k_sigma import KSigmaDetector

__all__ = [
    "AlarmEvent",
    "BaseDetector",
    "SADDetector",
    "KSigmaDetector",
]

# ARAD requires PyTorch — import conditionally
try:
    from gammaflow.algorithms.arad import ARADDetector
    __all__.append("ARADDetector")
except ImportError:
    pass

# Temporal LSTM detector requires PyTorch — import conditionally
try:
    from gammaflow.algorithms.lstm_temporal import LSTMTemporalDetector
    __all__.append("LSTMTemporalDetector")
except ImportError:
    pass

# CEW may not be present in released versions
try:
    from gammaflow.algorithms.censored_energy_window import (
        optimize_cew_windows,
        fit_cew_predictor,
        CEWPredictor,
    )
    __all__.extend([
        "optimize_cew_windows",
        "fit_cew_predictor",
        "CEWPredictor",
    ])
except ImportError:
    pass
