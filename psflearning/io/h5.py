from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf, DictConfig

import hdfdict
from dotted_dict import DottedDict
import h5py


def load(path: str | Path) -> tuple[DottedDict, DictConfig]:
    """Load data and parameters from an HDF5 file.

    Args:
        path: Path to the HDF5 file.

    Returns:
        A tuple containing the loaded data as a DottedDict and the
        parameters as a DictConfig.
    """
    with h5py.File(path, "r") as f:
        res = DottedDict(hdfdict.load(f, lazy=False))
        params = OmegaConf.create(f.attrs["params"])  # type: ignore[reportCallIssue]
    return res, params
