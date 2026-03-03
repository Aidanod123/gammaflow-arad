"""
K-Sigma anomaly detection algorithm.

Maintains a rolling background buffer and compares each foreground
measurement against background statistics.  An alarm is declared when
the foreground exceeds the background by more than *k* standard
deviations. K-sigma can also be used with ROIs for cases where the radioisotope(s)
of interest are known.

Algorithm
---------
1. Maintain a rolling background window (e.g. last 60 seconds).
2. Compute mean and standard deviation of background count rate.
3. Compare foreground count rate to background statistics.
4. Declare alarm if ``(foreground - mean) / std > k``.
5. Aggregate consecutive alarms into single detection events.
6. Do **not** update background during alarm states.
"""

from typing import Optional, Tuple
from collections import deque

import numpy as np

from gammaflow.algorithms.base import BaseDetector


class KSigmaDetector(BaseDetector):
    """
    K-sigma anomaly detection for gamma-ray time series.

    This is a **streaming** detector: background statistics are updated
    incrementally as new samples arrive, and the background is frozen
    during alarm periods.

    Because it does not require offline training, ``fit()`` is a no-op
    and ``is_trained`` always returns ``True``.  The detection threshold
    is expressed as a number of standard deviations (``k_threshold``)
    rather than an absolute score, so the inherited ``threshold``
    attribute is kept in sync with ``k_threshold``.

    Parameters
    ----------
    k_threshold : float
        Number of standard deviations above background for alarm.
        Typical values: 3-5 sigma.
    background_window : float
        Duration of background window in seconds (default: 60).
    foreground_window : float
        Duration of foreground window in seconds (default: 1).
    aggregation_gap : float
        Maximum time gap between alarms to aggregate (seconds, default: 2).
    min_background_samples : int
        Minimum samples required before detection starts (default: 10).

    Attributes
    ----------
    last_background_mean : float or None
        Most recent background mean (for diagnostics).
    last_background_std : float or None
        Most recent background standard deviation.

    Examples
    --------
    >>> detector = KSigmaDetector(k_threshold=5.0, background_window=60.0)
    >>> metrics = detector.process_time_series(time_series)
    >>> print(f"Detected {len(detector.alarms)} alarms")
    """

    def __init__(
        self,
        k_threshold: float = 5.0,
        background_window: float = 60.0,
        foreground_window: float = 1.0,
        aggregation_gap: float = 2.0,
        min_background_samples: int = 10,
    ):
        super().__init__(threshold=k_threshold, aggregation_gap=aggregation_gap)
        self.k_threshold = k_threshold
        self.background_window = background_window
        self.foreground_window = foreground_window
        self.min_background_samples = min_background_samples

        self.background_buffer: deque = deque()
        self.last_background_mean: Optional[float] = None
        self.last_background_std: Optional[float] = None

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return True

    def fit(self, background_data=None, **kwargs) -> "KSigmaDetector":
        """No-op — K-sigma does not require offline training."""
        return self

    def score_spectrum(self, spectrum) -> float:
        """
        Compute gross count rate for a single spectrum.

        For K-sigma, the meaningful metric is the k-sigma value which
        depends on background state.  This method returns the gross count
        rate so that ``score_time_series`` still works for diagnostic use,
        but ``process_time_series`` is the primary interface.
        """
        counts = spectrum.counts if hasattr(spectrum, "counts") else spectrum
        total = float(np.sum(counts))
        lt = getattr(spectrum, "live_time", None)
        rt = getattr(spectrum, "real_time", None)
        time = lt if lt is not None else (rt if rt is not None else 1.0)
        return total / time

    # ------------------------------------------------------------------
    # Streaming processing (overrides base batch mode)
    # ------------------------------------------------------------------

    def reset_alarms(self) -> None:
        """Reset detector state including background buffer."""
        super().reset_alarms()
        self.background_buffer.clear()
        self.last_background_mean = None
        self.last_background_std = None

    def process_sample(self, time: float, count_rate: float) -> Optional[float]:
        """
        Process a single time sample.

        Parameters
        ----------
        time : float
            Time of measurement (seconds).
        count_rate : float
            Gross count rate (counts/second).

        Returns
        -------
        float or None
            K-sigma metric if detection is active, ``None`` if still
            filling the background buffer.
        """
        bg_mean, bg_std = self._compute_background_stats()
        self.last_background_mean = bg_mean
        self.last_background_std = bg_std

        if bg_mean is None or bg_std is None:
            self._update_background_buffer(time, count_rate)
            return None

        alarm_metric = (count_rate - bg_mean) / bg_std

        if alarm_metric > self.k_threshold:
            if not self._is_alarming:
                self._start_alarm(time, alarm_metric)
            else:
                self._update_alarm_peak(time, alarm_metric)
        else:
            if self._is_alarming:
                self._end_alarm(time)
            self._update_background_buffer(time, count_rate)

        return alarm_metric

    def process_time_series(self, time_series) -> np.ndarray:
        """
        Process an entire time series using streaming k-sigma detection.

        Overrides the base batch implementation because k-sigma interleaves
        scoring with background buffer updates and freezes the background
        during alarm periods.

        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to process.

        Returns
        -------
        np.ndarray
            Array of k-sigma metrics for each time point. ``NaN`` where
            detection was not yet active (buffer filling).
        """
        self.reset_alarms()

        times = self._extract_timestamps(time_series)
        count_rates = np.array([
            float(np.sum(s.counts)) / (
                s.live_time if s.live_time is not None else s.real_time
            )
            for s in time_series.spectra
        ])

        metrics = np.full(time_series.n_spectra, np.nan)
        for i, (t, rate) in enumerate(zip(times, count_rates)):
            metric = self.process_sample(t, rate)
            if metric is not None:
                metrics[i] = metric

        if self._is_alarming:
            self._end_alarm(times[-1])

        return metrics

    # ------------------------------------------------------------------
    # Background buffer helpers
    # ------------------------------------------------------------------

    def _update_background_buffer(self, time: float, count_rate: float) -> None:
        """Add sample and evict samples outside the background window."""
        self.background_buffer.append((time, count_rate))
        cutoff = time - self.background_window
        while self.background_buffer and self.background_buffer[0][0] < cutoff:
            self.background_buffer.popleft()

    def _compute_background_stats(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (mean, std) of background buffer, or (None, None) if insufficient data."""
        if len(self.background_buffer) < self.min_background_samples:
            return None, None
        rates = np.array([cr for _, cr in self.background_buffer])
        mean = float(np.mean(rates))
        std = float(np.std(rates, ddof=1))
        if std < 1e-10:
            std = 1e-10
        return mean, std

    def __repr__(self) -> str:
        return (
            f"KSigmaDetector(k={self.k_threshold}, "
            f"bg_window={self.background_window}s, "
            f"alarms={len(self.alarms)})"
        )
