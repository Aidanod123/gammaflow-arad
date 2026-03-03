"""
GammaFlow: A Python library for working with time series of gamma ray spectra.
"""

from gammaflow.core.spectrum import Spectrum
from gammaflow.core.spectra import Spectra
from gammaflow.core.time_series import SpectralTimeSeries
from gammaflow.core.listmode import ListMode
from gammaflow.utils.exceptions import (
    GammaFlowError,
    SpectrumError,
    CalibrationError,
    IncompatibleBinningError,
)

# Make subpackages easily accessible
from gammaflow import operations
from gammaflow import algorithms
from gammaflow import datasets

__version__ = "0.1.0"
__all__ = [
    "Spectrum",
    "Spectra",
    "SpectralTimeSeries",
    "ListMode",
    "GammaFlowError",
    "SpectrumError",
    "CalibrationError",
    "IncompatibleBinningError",
    "operations",
    "algorithms",
    "datasets",
]

