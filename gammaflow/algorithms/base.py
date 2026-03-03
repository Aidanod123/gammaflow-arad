"""
Base classes for detection algorithms.

Provides the shared infrastructure for all anomaly detection algorithms:
- AlarmEvent dataclass for representing detected anomalies
- BaseDetector abstract class with alarm aggregation, threshold management,
  and time series processing

Future algorithm families (identification, directionality, localization)
should define their own base classes in this module as needed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np


@dataclass
class AlarmEvent:
    """
    Represents a detected anomaly event.

    An alarm event spans a contiguous time period during which the detection
    metric exceeded the threshold. Nearby threshold crossings are aggregated
    into a single event based on the detector's aggregation_gap parameter.

    Attributes
    ----------
    start_time : float
        Start time of alarm (seconds)
    end_time : float
        End time of alarm (seconds)
    peak_metric : float
        Peak detection score during alarm
    peak_time : float
        Time of peak detection score
    """

    start_time: float
    end_time: float
    peak_metric: float
    peak_time: float

    @property
    def duration(self) -> float:
        """Duration of alarm in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "peak_metric": self.peak_metric,
            "peak_time": self.peak_time,
            "duration": self.duration,
        }

    def __repr__(self) -> str:
        return (
            f"AlarmEvent(start={self.start_time:.2f}s, end={self.end_time:.2f}s, "
            f"peak={self.peak_metric:.2f} at {self.peak_time:.2f}s, "
            f"duration={self.duration:.2f}s)"
        )


class BaseDetector(ABC):
    """
    Abstract base class for anomaly detection algorithms.

    Provides shared infrastructure for threshold-based anomaly detection
    on gamma-ray spectral time series:

    - Alarm state management with gap-based aggregation
    - Batch time series processing (score all spectra, then aggregate alarms)
    - False alarm rate (FAR) based threshold calibration
    - Alarm summary statistics

    Subclasses must implement:
    - ``fit(background_data, **kwargs)`` — train on background data
    - ``score_spectrum(spectrum)`` — compute anomaly score for a single spectrum
    - ``is_trained`` property — whether the detector has been trained

    Streaming detectors (e.g. KSigma) that interleave scoring with alarm
    management should override ``process_time_series`` while reusing the
    alarm helper methods.

    Parameters
    ----------
    threshold : float or None
        Detection threshold. Scores above this trigger alarms.
    aggregation_gap : float
        Maximum time gap (seconds) between consecutive above-threshold
        samples that will be merged into a single alarm event. Default is 2.0.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        aggregation_gap: float = 2.0,
    ):
        self.threshold = threshold
        self.aggregation_gap = aggregation_gap

        # Alarm state
        self.alarms: List[AlarmEvent] = []
        self._is_alarming = False
        self._current_alarm_start: Optional[float] = None
        self._current_alarm_peak_metric: float = -np.inf
        self._current_alarm_peak_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, background_data, **kwargs) -> "BaseDetector":
        """
        Train the detector on background (source-absent) data.

        Parameters
        ----------
        background_data
            Training data (type depends on subclass).

        Returns
        -------
        self
        """
        ...

    @abstractmethod
    def score_spectrum(self, spectrum) -> float:
        """
        Compute an anomaly score for a single spectrum.

        Parameters
        ----------
        spectrum
            Spectrum to score.

        Returns
        -------
        float
            Anomaly score. Higher values indicate greater anomaly.
        """
        ...

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Whether the detector has been fitted / is ready for scoring."""
        ...

    # ------------------------------------------------------------------
    # Alarm state management
    # ------------------------------------------------------------------

    def reset_alarms(self) -> None:
        """Reset all alarm state (clears alarms list and in-progress alarm)."""
        self.alarms = []
        self._is_alarming = False
        self._current_alarm_start = None
        self._current_alarm_peak_metric = -np.inf
        self._current_alarm_peak_time = None

    def _start_alarm(self, time: float, metric: float) -> None:
        """Start a new alarm period."""
        self._is_alarming = True
        self._current_alarm_start = time
        self._current_alarm_peak_metric = metric
        self._current_alarm_peak_time = time

    def _update_alarm_peak(self, time: float, metric: float) -> None:
        """Update peak metric if *metric* exceeds current peak."""
        if metric > self._current_alarm_peak_metric:
            self._current_alarm_peak_metric = metric
            self._current_alarm_peak_time = time

    def _end_alarm(self, time: float) -> None:
        """
        End the current alarm period and record the event.

        If the gap between this alarm and the previous one is less than
        ``aggregation_gap``, the two are merged into a single event.
        """
        if not self._is_alarming or self._current_alarm_start is None:
            return

        if self.alarms:
            last_alarm = self.alarms[-1]
            time_since_last = self._current_alarm_start - last_alarm.end_time

            if time_since_last < self.aggregation_gap:
                last_alarm.end_time = time
                if self._current_alarm_peak_metric > last_alarm.peak_metric:
                    last_alarm.peak_metric = self._current_alarm_peak_metric
                    last_alarm.peak_time = self._current_alarm_peak_time

                self._reset_current_alarm()
                return

        alarm = AlarmEvent(
            start_time=self._current_alarm_start,
            end_time=time,
            peak_metric=self._current_alarm_peak_metric,
            peak_time=self._current_alarm_peak_time,
        )
        self.alarms.append(alarm)
        self._reset_current_alarm()

    def _reset_current_alarm(self) -> None:
        """Clear in-progress alarm state without touching the alarms list."""
        self._is_alarming = False
        self._current_alarm_start = None
        self._current_alarm_peak_metric = -np.inf
        self._current_alarm_peak_time = None

    # ------------------------------------------------------------------
    # Time series processing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_timestamps(time_series) -> np.ndarray:
        """
        Extract an array of timestamps from a SpectralTimeSeries.

        Falls back to cumulative real_times if explicit timestamps are
        unavailable.
        """
        if time_series.timestamps is not None and time_series.timestamps.dtype != object:
            if time_series.timestamps[0] is not None:
                return time_series.timestamps

        if time_series.real_times is not None:
            return np.cumsum(np.asarray(time_series.real_times, dtype=float))

        times = np.array(
            [s.real_time for s in time_series.spectra], dtype=float
        )
        return np.cumsum(times)

    def score_time_series(self, time_series) -> np.ndarray:
        """
        Score every spectrum in a time series (no alarm processing).

        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to score.

        Returns
        -------
        np.ndarray
            Array of anomaly scores, one per spectrum.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Detector must be trained before scoring. Call fit() first."
            )
        return np.array([
            self.score_spectrum(spec) for spec in time_series.spectra
        ])

    def process_time_series(self, time_series) -> np.ndarray:
        """
        Score a time series and aggregate alarms.

        This is the standard batch-mode processing pipeline:
        1. Reset alarm state
        2. Score all spectra
        3. Walk through scores and build alarm events

        Streaming detectors that interleave scoring with background updates
        should override this method.

        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to process.

        Returns
        -------
        np.ndarray
            Array of anomaly scores, one per spectrum.

        Raises
        ------
        RuntimeError
            If detector is not trained or threshold is not set.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Detector must be trained before processing. Call fit() first."
            )
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not set. Call set_threshold() or "
                "set_threshold_by_far() first."
            )

        self.reset_alarms()
        scores = self.score_time_series(time_series)
        times = self._extract_timestamps(time_series)

        for score, t in zip(scores, times):
            if score > self.threshold:
                if not self._is_alarming:
                    self._start_alarm(t, score)
                else:
                    self._update_alarm_peak(t, score)
            else:
                if self._is_alarming:
                    self._end_alarm(t)

        if self._is_alarming:
            self._end_alarm(times[-1])

        return scores

    # ------------------------------------------------------------------
    # Threshold calibration
    # ------------------------------------------------------------------

    def set_threshold(self, threshold: float) -> "BaseDetector":
        """
        Manually set the detection threshold.

        Parameters
        ----------
        threshold : float
            Score threshold for declaring alarms.

        Returns
        -------
        self
        """
        self.threshold = threshold
        return self

    def set_threshold_by_far(
        self,
        background_data,
        alarms_per_hour: float = 1.0,
        max_iterations: int = 20,
        verbose: bool = False,
    ) -> float:
        """
        Set threshold to achieve a target false alarm rate.

        Uses binary search over the threshold, running ``process_time_series``
        at each step to count aggregated alarm events. This accounts for
        alarm merging via ``aggregation_gap``.

        Parameters
        ----------
        background_data : SpectralTimeSeries
            Background data for calibration.
        alarms_per_hour : float
            Desired false alarm rate (alarms per hour). Default is 1.0.
        max_iterations : int
            Maximum binary search iterations. Default is 20.
        verbose : bool
            Print progress. Default is False.

        Returns
        -------
        float
            Calibrated threshold value.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Detector must be trained before setting threshold."
            )

        scores = self.score_time_series(background_data)

        total_time_seconds = float(np.sum(background_data.real_times))
        total_time_hours = total_time_seconds / 3600.0

        if total_time_hours <= 0:
            raise ValueError(
                f"Invalid observation time: {total_time_hours} hours"
            )

        low_threshold = float(np.min(scores))
        high_threshold = float(np.max(scores)) * 1.5

        best_threshold = float(
            np.percentile(
                scores,
                max(0.1, min(99.9, 100 * (1 - alarms_per_hour / (60 * len(scores))))),
            )
        )
        best_far_diff = float("inf")
        best_observed_far = 0.0

        if verbose:
            print(
                f"Calibrating threshold for {alarms_per_hour:.2f} alarms/hour...\n"
                f"  Background: {len(scores)} spectra over {total_time_hours:.2f} hours\n"
                f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]\n"
                f"  Score mean +/- std: {scores.mean():.4f} +/- {scores.std():.4f}"
            )

        for iteration in range(max_iterations):
            test_threshold = (low_threshold + high_threshold) / 2
            self.threshold = test_threshold

            self.process_time_series(background_data)
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

        # Final pass with best threshold
        self.process_time_series(background_data)
        final_far = len(self.alarms) / total_time_hours

        if verbose:
            print(
                f"\n  Threshold set: {self.threshold:.6f}\n"
                f"  Achieved FAR: {final_far:.2f} alarms/hour "
                f"({len(self.alarms)} alarms)\n"
                f"  Target FAR: {alarms_per_hour:.2f} alarms/hour"
            )

        return self.threshold

    # ------------------------------------------------------------------
    # Summary / utilities
    # ------------------------------------------------------------------

    def get_alarm_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of detected alarms.

        Returns
        -------
        dict
            Keys: n_alarms, total_alarm_time, mean_duration,
            max_peak_metric, alarm_events.
        """
        if not self.alarms:
            return {
                "n_alarms": 0,
                "total_alarm_time": 0.0,
                "mean_duration": 0.0,
                "max_peak_metric": 0.0,
                "alarm_events": [],
            }

        durations = [a.duration for a in self.alarms]
        return {
            "n_alarms": len(self.alarms),
            "total_alarm_time": sum(durations),
            "mean_duration": float(np.mean(durations)),
            "max_peak_metric": max(a.peak_metric for a in self.alarms),
            "alarm_events": [a.to_dict() for a in self.alarms],
        }
