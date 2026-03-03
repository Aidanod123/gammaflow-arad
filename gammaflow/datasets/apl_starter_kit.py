"""
APL Starter Kit dataset loader.

The APL Starter Kit dataset consists of pre-binned gamma-ray spectral measurements from
mobile NaI(Tl) detectors driving through a controlled urban test zone.
Each ``.open`` file is a tab-delimited collection of 1-second spectral
measurements with up to 4 gamma detector elements, each producing 1024-channel
spectra spanning 0–3000 keV.

The dataset is split into ``Background`` (source-absent) and ``Source``
(source-present at known times) subdirectories under a detector folder
(e.g. ``M.0``).

Typical usage::

    from gammaflow.datasets import APLDataset

    ds = APLDataset("/path/to/Starter Kit Data Set")

    # List available files
    bg_files = ds.list_files("background")
    src_files = ds.list_files("source")

    # Load a single file as a SpectralTimeSeries (detector element 0)
    ts, meta = ds.load_file(src_files[0], detector=0)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from gammaflow.core.spectrum import Spectrum
from gammaflow.core.time_series import SpectralTimeSeries


# Default energy calibration from M.0.config: 1024 channels, 0–3000 keV
_DEFAULT_N_CHANNELS = 1024
_DEFAULT_ENERGY_RANGE = (0.0, 3000.0)

# Columns that are not per-detector and appear once per row
_SCALAR_COLUMNS = [
    "record",
    "detector",
    "utc-time",
    "timestamp",
    "azimuth",
    "azimuth-uncertainty",
    "is-in-zone",
    "is-closest-approach",
    "is-source-present",
    "source-id",
    "source-offset",
    "latitude",
    "longitude",
    "distance-to-doca",
    "is-active",
    "heading",
]


class APLStarterKitDataset:
    """
    Loader for the APL Starter Kit dataset.

    Parameters
    ----------
    data_dir : str or Path
        Root directory of the dataset.  Expected structure::

            data_dir/
                M.0/
                    Background/    # .open files (source-absent runs)
                    Source/        # .open files (source-present runs)
                    Supplemental files/
                        Scoring Tool config files/
                            M.0.config
                            M.0_ROI.csv
                        Valid files/
                            *.valid

    detector_dir : str
        Name of the detector subdirectory (default ``'M.0'``).

    Attributes
    ----------
    data_dir : Path
        Resolved root path.
    detector_dir : str
        Name of the detector folder.
    n_channels : int
        Number of spectral channels (1024).
    energy_range : tuple of float
        Energy range in keV, ``(0.0, 3000.0)``.

    Examples
    --------
    >>> ds = APLStarterKitDataset("/path/to/data")
    >>> ts, meta = ds.load_file("filename.open", detector=0, active_only=True)
    """

    n_channels = _DEFAULT_N_CHANNELS
    energy_range = _DEFAULT_ENERGY_RANGE

    def __init__(
        self,
        data_dir: Union[str, Path],
        detector_dir: str = "M.0",
    ):
        self.data_dir = Path(data_dir)
        self.detector_dir = detector_dir

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}"
            )

        self._det_path = self.data_dir / detector_dir
        if not self._det_path.exists():
            raise FileNotFoundError(
                f"Detector directory not found: {self._det_path}"
            )

    # ------------------------------------------------------------------
    # File listing
    # ------------------------------------------------------------------

    def list_files(self, split: str = "background") -> List[str]:
        """
        List available ``.open`` file names for a given split.

        Parameters
        ----------
        split : str
            ``'background'`` or ``'source'``.

        Returns
        -------
        list of str
            Sorted list of ``.open`` filenames (not full paths).
        """
        split_dir = self._resolve_split_dir(split)
        files = sorted(
            f.name for f in split_dir.iterdir()
            if f.suffix == ".open"
        )
        return files

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_file(
        self,
        filename: str,
        split: str = "source",
        detector: int = 0,
        active_only: bool = False,
        in_zone_only: bool = False,
    ) -> Tuple[SpectralTimeSeries, pd.DataFrame]:
        """
        Load an ``.open`` file as a :class:`SpectralTimeSeries`.

        Parameters
        ----------
        filename : str
            Name of the ``.open`` file (e.g.
            ``'filename.open'``).
        split : str
            ``'background'`` or ``'source'``.
        detector : int
            Gamma detector element index (0–3). Default is 0.
        active_only : bool
            If ``True``, drop rows where ``is-active`` is ``FALSE``.
        in_zone_only : bool
            If ``True``, drop rows where ``is-in-zone`` is ``FALSE``.

        Returns
        -------
        time_series : SpectralTimeSeries
            Spectral time series for the requested detector element.
        metadata : pd.DataFrame
            Per-sample metadata (one row per spectrum).  Columns include
            ``timestamp``, ``is-source-present``, ``source-id``,
            ``latitude``, ``longitude``, etc.
        """
        split_dir = self._resolve_split_dir(split)
        filepath = split_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = self._read_open_file(filepath)

        if active_only:
            df = df[df["is-active"].astype(bool)].reset_index(drop=True)
        if in_zone_only:
            df = df[df["is-in-zone"].astype(bool)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No rows remain after filtering in {filename}"
            )

        return self._build_time_series(df, detector)

    def load_files(
        self,
        filenames: Optional[List[str]] = None,
        split: str = "source",
        detector: int = 0,
        active_only: bool = False,
        in_zone_only: bool = False,
    ):
        """
        Generator that yields ``(time_series, metadata)`` for multiple files.

        Parameters
        ----------
        filenames : list of str, optional
            Files to load.  If ``None``, loads all files in *split*.
        split : str
            ``'background'`` or ``'source'``.
        detector : int
            Gamma detector element index (0–3).
        active_only : bool
            Filter to active rows only.
        in_zone_only : bool
            Filter to in-zone rows only.

        Yields
        ------
        time_series : SpectralTimeSeries
        metadata : pd.DataFrame
        """
        if filenames is None:
            filenames = self.list_files(split)

        for fname in filenames:
            yield self.load_file(
                fname,
                split=split,
                detector=detector,
                active_only=active_only,
                in_zone_only=in_zone_only,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_split_dir(self, split: str) -> Path:
        """Resolve split name to directory path."""
        name = split.strip().lower()
        if name in ("background", "bg"):
            return self._det_path / "Background"
        elif name in ("source", "src"):
            return self._det_path / "Source"
        else:
            raise ValueError(
                f"Unknown split '{split}'. Use 'background' or 'source'."
            )

    @staticmethod
    def _read_open_file(filepath: Path) -> pd.DataFrame:
        """
        Parse a ``.open`` file into a DataFrame.

        Handles the optional ``#`` calibration header line and
        tab-delimited data.
        """
        with open(filepath, "r") as f:
            first_line = f.readline()

        skip = 1 if first_line.startswith("#") else 0

        df = pd.read_csv(
            filepath,
            sep="\t",
            skiprows=skip,
            low_memory=False,
        )
        return df

    def _build_time_series(
        self,
        df: pd.DataFrame,
        detector: int,
    ) -> Tuple[SpectralTimeSeries, pd.DataFrame]:
        """
        Extract a SpectralTimeSeries for a single detector element.
        """
        spec_col = f"spectrum-channels{detector}"
        lt_col = f"spectrum-lt{detector}"
        rt_col = f"spectrum-rt{detector}"

        if spec_col not in df.columns:
            raise ValueError(
                f"Detector {detector} not found in file "
                f"(no column '{spec_col}')"
            )

        # Parse comma-delimited spectra into a 2D array
        counts_list = []
        for row_spectrum in df[spec_col].values:
            channels = np.fromstring(str(row_spectrum), sep=",", dtype=np.float64)
            counts_list.append(channels)

        counts_array = np.array(counts_list, dtype=np.float64)
        n_spectra, n_channels = counts_array.shape

        # Energy edges
        energy_edges = np.linspace(
            self.energy_range[0],
            self.energy_range[1],
            n_channels + 1,
        )

        # Timestamps (ms → s, relative to first sample)
        timestamps_ms = df["timestamp"].values.astype(np.float64)
        timestamps = (timestamps_ms - timestamps_ms[0]) / 1000.0

        # Live and real times (ms → s)
        live_times = df[lt_col].values.astype(np.float64) / 1000.0
        real_times = df[rt_col].values.astype(np.float64) / 1000.0

        ts = SpectralTimeSeries.from_array(
            counts_array,
            energy_edges=energy_edges,
            timestamps=timestamps,
            live_times=live_times,
            real_times=real_times,
        )

        # Build metadata DataFrame with scalar columns that exist
        meta_cols = [c for c in _SCALAR_COLUMNS if c in df.columns]
        metadata = df[meta_cols].copy()

        # Add per-detector gross counts if available
        gc_col = f"gc{detector}"
        if gc_col in df.columns:
            metadata[gc_col] = df[gc_col]

        # Add neutron columns if available
        for nc in ("nc0", "nlt0", "nrt0"):
            if nc in df.columns:
                metadata[nc] = df[nc]

        return ts, metadata

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            n_bg = len(self.list_files("background"))
        except FileNotFoundError:
            n_bg = "?"
        try:
            n_src = len(self.list_files("source"))
        except FileNotFoundError:
            n_src = "?"
        return (
            f"APLStarterKitDataset(data_dir='{self.data_dir}', "
            f"detector='{self.detector_dir}', "
            f"background_files={n_bg}, source_files={n_src})"
        )
