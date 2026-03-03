"""
TopCoder Urban Radiation Search dataset loader.

The TopCoder dataset consists of simulated list-mode gamma-ray measurements
from a 2"x4"x16" NaI(Tl) detector driving through an urban environment.
Each run is a CSV file containing (time_delta_us, energy_keV) event pairs.
Ground-truth labels identify the radioactive source (if any) and the time
at which the detector passes closest to it.

For more information see:
Ghawaly Jr, J. M., Nicholson, A. D., Peplow, D. E., Anderson-Cook, C. M., 
    Myers, K. L., Archer, D. E., ... & Quiter, B. J. (2020). Data for training 
    and testing radiation detection algorithms in an urban environment. Scientific 
    Data, 7(1), 328. https://doi.org/10.1038/s41597-020-00672-2

Typical usage::

    from gammaflow.datasets import TopCoderDataset

    ds = TopCoderDataset("/path/to/topcoder")

    # Load a single run
    listmode, meta = ds.load_run(100001)

    # Iterate over all I-131 runs
    for run_id in ds.list_runs(source_id=3):
        lm, meta = ds.load_run(run_id)
        ...
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from gammaflow.core.listmode import ListMode


SOURCE_MAP: Dict[int, str] = {
    0: "Background",
    1: "HEU",
    2: "WGPu",
    3: "I-131",
    4: "Co-60",
    5: "Tc-99m",
    6: "Tc-99m + HEU",
}

SOURCE_NAME_TO_ID: Dict[str, int] = {v: k for k, v in SOURCE_MAP.items()}


class TopCoderDataset:
    """
    Loader for the TopCoder Urban Radiation Search dataset.

    Parameters
    ----------
    data_dir : str or Path
        Root directory of the TopCoder dataset.  Expected structure::

            data_dir/
                training/          # Run CSV files (e.g. 100001.csv)
                testing/           # Run CSV files
                scorer/
                    answerKey_training.csv
                    answerKey_testing.csv

    Attributes
    ----------
    data_dir : Path
        Resolved dataset root path.
    source_map : dict
        Mapping from integer SourceID to human-readable source name.

    Examples
    --------
    >>> ds = TopCoderDataset("topcoder/")
    >>> listmode, meta = ds.load_run(100001)
    >>> print(meta["SourceName"], meta["SourceTime"])
    Background 0

    >>> bg_ids = ds.list_runs(source_id=0)
    >>> print(f"{len(bg_ids)} background runs available")
    """

    source_map = SOURCE_MAP

    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir}"
            )

        self._answer_keys: Dict[str, pd.DataFrame] = {}

    # ---------------------
    # Answer key / metadata
    def get_answer_key(self, dataset: str = "training") -> pd.DataFrame:
        """
        Load the answer key for a dataset split.

        Column names are normalized to a common schema regardless of
        the on-disk format:

        - ``RunID`` (int)
        - ``SourceID`` (int)
        - ``SourceTime`` (float)
        - ``SourceName`` (str, added automatically)
        - ``Speed/Offset`` (float)

        Parameters
        ----------
        dataset : str
            ``'training'`` or ``'testing'``.

        Returns
        -------
        pd.DataFrame
        """
        if dataset not in self._answer_keys:
            path = self.data_dir / "scorer" / f"answerKey_{dataset}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Answer key not found: {path}")
            df = pd.read_csv(path)

            # The training and testing CSVs use different column names.
            # Normalize to a common schema. 
            rename = {
                "RunNumber": "RunID",
                "Source": "SourceID",
                "NearestTime": "SourceTime",
                "Type": "Part",
                "Speed/offset": "Speed/Offset",
            }
            df = df.rename(columns={
                k: v for k, v in rename.items() if k in df.columns
            })

            df["SourceName"] = df["SourceID"].map(SOURCE_MAP).fillna("Unknown")
            self._answer_keys[dataset] = df

        return self._answer_keys[dataset]

    def get_run_metadata(
        self, run_id: int, dataset: str = "training"
    ) -> Dict[str, Any]:
        """
        Get metadata for a specific run.

        Parameters
        ----------
        run_id : int
            Run identifier.
        dataset : str
            ``'training'`` or ``'testing'``.

        Returns
        -------
        dict
            Keys: RunID, SourceID, SourceName, SourceTime, Part,
            Speed/Offset.
        """
        ak = self.get_answer_key(dataset)
        row = ak[ak["RunID"] == run_id]
        if row.empty:
            raise KeyError(
                f"RunID {run_id} not found in {dataset} answer key"
            )
        return row.iloc[0].to_dict()

    # -----------------------
    # Run listing / filtering
    def list_runs(
        self,
        dataset: str = "training",
        source_id: Optional[int] = None,
        source_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        List available run IDs, optionally filtered by source.

        Parameters
        ----------
        dataset : str
            ``'training'`` or ``'testing'``.
        source_id : int, optional
            Filter by integer SourceID (0=Background, 1=HEU, etc.).
        source_name : str, optional
            Filter by source name (e.g. ``'I-131'``).  Ignored if
            *source_id* is also provided.

        Returns
        -------
        np.ndarray
            Array of integer RunID values.
        """
        ak = self.get_answer_key(dataset)

        if source_id is not None:
            ak = ak[ak["SourceID"] == source_id]
        elif source_name is not None:
            sid = SOURCE_NAME_TO_ID.get(source_name)
            if sid is None:
                raise ValueError(
                    f"Unknown source name '{source_name}'. "
                    f"Valid names: {list(SOURCE_NAME_TO_ID.keys())}"
                )
            ak = ak[ak["SourceID"] == sid]

        return ak["RunID"].values

    # ------------
    # Data loading
    def load_run(
        self,
        run_id: int,
        dataset: str = "training",
    ) -> Tuple[ListMode, Dict[str, Any]]:
        """
        Load a single run as a ListMode object with metadata.

        Parameters
        ----------
        run_id : int
            Run identifier (e.g. ``100001``).
        dataset : str
            ``'training'`` or ``'testing'``.

        Returns
        -------
        listmode : ListMode
            List-mode event data.
        metadata : dict
            Run metadata from the answer key, including ``SourceName``.
        """
        csv_path = self.data_dir / dataset / f"{run_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Run file not found: {csv_path}")

        data = pd.read_csv(
            csv_path, header=None, names=["time_delta_us", "energy_keV"]
        )
        time_deltas = data["time_delta_us"].values * 1e-6  # μs → s
        energies = data["energy_keV"].values

        listmode = ListMode(time_deltas, energies)
        metadata = self.get_run_metadata(run_id, dataset)

        return listmode, metadata

    def load_runs(
        self,
        run_ids: Optional[List[int]] = None,
        dataset: str = "training",
        source_id: Optional[int] = None,
        source_name: Optional[str] = None,
    ):
        """
        Generator that yields ``(listmode, metadata)`` for multiple runs.

        If *run_ids* is ``None``, uses ``list_runs`` with the given filters.

        Parameters
        ----------
        run_ids : list of int, optional
            Explicit run IDs to load.
        dataset : str
            ``'training'`` or ``'testing'``.
        source_id : int, optional
            Filter (only used when *run_ids* is ``None``).
        source_name : str, optional
            Filter (only used when *run_ids* is ``None``).

        Yields
        ------
        listmode : ListMode
        metadata : dict
        """
        if run_ids is None:
            run_ids = self.list_runs(
                dataset=dataset, source_id=source_id, source_name=source_name
            )

        for rid in run_ids:
            yield self.load_run(rid, dataset=dataset)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            n_train = len(self.list_runs("training"))
        except FileNotFoundError:
            n_train = "?"
        try:
            n_test = len(self.list_runs("testing"))
        except FileNotFoundError:
            n_test = "?"
        return (
            f"TopCoderDataset(data_dir='{self.data_dir}', "
            f"training={n_train}, testing={n_test})"
        )
