"""
Spectral Anomaly Detection (SAD) using PCA-based reconstruction error.

This module implements the SAD algorithm which learns a low-dimensional subspace
of background spectra using PCA, then detects anomalies by measuring how poorly
new spectra can be reconstructed using only that subspace.

References
----------
Miller, K., & Dubrawski, A. (2018). Gamma-ray source detection with small sensors.
    IEEE Transactions on Nuclear Science, 65(4), 1047-1058.

"""

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from gammaflow.algorithms.base import BaseDetector


class SADDetector(BaseDetector):
    """
    Spectral Anomaly Detector using PCA-based reconstruction error.

    Learns a low-dimensional subspace of background spectra via Principal
    Component Analysis. New spectra are scored by reconstruction error:

        SAD(x) = ||(I - UU^T)x||^2

    where U is the orthonormal basis of the learned subspace.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain (default: 5).
    threshold : float or None
        SAD score threshold for declaring anomalies (default: None).
    normalize : bool
        Normalize spectra to unit integral before PCA (default: True).
    aggregation_gap : float
        Time gap (seconds) for aggregating consecutive alarms (default: 2.0).

    Attributes
    ----------
    pca : sklearn.decomposition.PCA or None
        Fitted PCA model (``None`` before training).

    Examples
    --------
    >>> detector = SADDetector(n_components=5)
    >>> detector.fit(background_time_series)
    >>> detector.set_threshold_by_far(background_time_series, alarms_per_hour=0.5)
    >>> scores = detector.process_time_series(test_time_series)
    >>> print(f"Detected {len(detector.alarms)} anomalies")
    """

    def __init__(
        self,
        n_components: int = 5,
        threshold: Optional[float] = None,
        normalize: bool = True,
        aggregation_gap: float = 2.0,
    ):
        super().__init__(threshold=threshold, aggregation_gap=aggregation_gap)
        self.n_components = n_components
        self.normalize = normalize
        self.pca: Optional[PCA] = None

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self.pca is not None

    def fit(self, background_data, **kwargs) -> "SADDetector":
        """
        Train the detector on background (source-absent) data.

        Parameters
        ----------
        background_data : SpectralTimeSeries or Spectra
            Background spectra for training.

        Returns
        -------
        self
        """
        if hasattr(background_data, "spectra"):
            spectra = background_data.spectra
        else:
            raise ValueError("background_data must have a 'spectra' attribute")

        n_bins = spectra[0].counts.shape[0]
        X = np.zeros((len(spectra), n_bins), dtype=np.float64)
        for i, spec in enumerate(spectra):
            X[i, :] = self._prepare_spectrum(spec.counts)

        self.pca = PCA(n_components=self.n_components, svd_solver="full")
        self.pca.fit(X)
        return self

    def score_spectrum(self, spectrum) -> float:
        """
        Compute SAD score (squared reconstruction error) for a single spectrum.

        Parameters
        ----------
        spectrum : Spectrum or np.ndarray
            Spectrum to score.

        Returns
        -------
        float
            SAD score.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Detector must be trained before scoring. Call fit() first."
            )

        counts = spectrum.counts if hasattr(spectrum, "counts") else spectrum
        x = self._prepare_spectrum(counts).reshape(1, -1)
        with np.errstate(all="ignore"):
            x_reconstructed = self.pca.inverse_transform(self.pca.transform(x))
        residual = np.nan_to_num(x - x_reconstructed, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.sum(residual ** 2))

    # ------------------------------------------------------------------
    # SAD-specific helpers
    # ------------------------------------------------------------------

    def _normalize_spectrum(self, counts: np.ndarray) -> np.ndarray:
        """Normalize spectrum to unit integral."""
        total = counts.sum()
        return counts / total if total > 0 else counts

    def _prepare_spectrum(self, counts: np.ndarray) -> np.ndarray:
        """Prepare spectrum for PCA (normalize if enabled)."""
        counts = np.asarray(counts, dtype=np.float64)
        if self.normalize:
            return self._normalize_spectrum(counts)
        return counts

    # ------------------------------------------------------------------
    # PCA diagnostics
    # ------------------------------------------------------------------

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Explained variance ratio for each principal component.

        Returns
        -------
        np.ndarray
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained first. Call fit().")
        return self.pca.explained_variance_ratio_

    def get_cumulative_variance_explained(self) -> float:
        """
        Total variance explained by all retained components.

        Returns
        -------
        float
            Cumulative explained variance (0-1).
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained first. Call fit().")
        return float(self.pca.explained_variance_ratio_.sum())
