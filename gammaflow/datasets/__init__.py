"""
Dataset loaders for standard gamma-ray spectroscopy datasets.

Provides high-level interfaces for loading and working with common
benchmark datasets used in radiation detection research.

Available datasets:
- ``APLStarterKitDataset`` — APL Starter Kit dataset (pre-binned spectral data)
- ``TopCoderDataset`` — TopCoder Urban Radiation Search challenge data
"""

from gammaflow.datasets.apl_starter_kit import APLStarterKitDataset
from gammaflow.datasets.topcoder import TopCoderDataset

__all__ = ["APLStarterKitDataset", "TopCoderDataset"]
