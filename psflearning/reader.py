"""
Handles input operations: loading images, parameters, initial pupil
state, and previously saved results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py as h5
import hdfdict
import numpy as np
from dotted_dict import DottedDict
from omegaconf import DictConfig, OmegaConf

from .io.param import RunParameters
from .dataloader import get_loader


class Reader:
    """Unified interface for all read operations in the PSF-learning pipeline."""

    # ── Image loading ────────────────────────────────────────────────────

    def read_images(
        self, param: RunParameters, frange: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Load raw image stacks from disk.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters (file paths, ...).
        frange : tuple of (int, int), optional
            Slice ``filelist[frange[0]:frange[1]]`` to restrict the files
            loaded.

        Returns
        -------
        numpy.ndarray
            Image array with axes arranged for the requested configuration.
        """
        filelist = param.filelist

        loader = get_loader(param)
        if not filelist:
            filelist = loader.get_file_list()
        if frange:
            filelist = filelist[frange[0] : frange[1]]

        images_all = loader.load(filelist)
        images = self._rearrange_axes(images_all, param)
        images = self._reshape_insitu(images, param)
        images = self._swap_xy(images, param)
        images = self._flip_if_reverse(images, param)

        logging.info("Loaded images shape: %s", images.shape)
        return images

    # ── Parameter loading ────────────────────────────────────────────────

    @staticmethod
    def read_params(path: Union[str, Path]) -> DictConfig:
        cfg = OmegaConf.load(path)
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"Expected DictConfig, got {type(cfg).__name__}")
        return cfg

    @staticmethod
    def combine_params(
        basefile: str,
        psftype: Optional[str] = None,
        sysfile: Optional[str] = None,
    ) -> RunParameters:
        """Combine a base configuration with PSF / channel / system overrides.

        Parameters
        ----------
        basefile : str
            Base configuration name (without ``.yaml``), resolved relative
            to the package ``config/`` directory.
        psftype : str, optional
            PSF type override (e.g. ``"zernike"``).
        sysfile : str, optional
            System type override (e.g. ``"M2"``, ``"TP"``).

        Returns
        -------
        RunParameters
            Merged parameters.
        """
        from .io.param import combine as _combine
        return _combine(basefile, psftype=psftype, sysfile=sysfile)

    # ── Result loading ───────────────────────────────────────────────────

    @staticmethod
    def read_results(
        path: Union[str, Path],
    ) -> Tuple[DottedDict, DictConfig]:
        """Load results and parameters from an HDF5 result file.

        Parameters
        ----------
        path : str or Path
            Path to the ``.h5`` result file.

        Returns
        -------
        tuple of (DottedDict, DictConfig)
            ``(results, params)`` where *results* contains the nested
            ``res``, ``locres`` and ``rois`` groups, and *params* is the
            original experiment configuration.
        """
        with h5.File(path, "r") as f:
            res = DottedDict(hdfdict.load(f, lazy=False))
            raw_params = OmegaConf.create(str(f.attrs["params"]))
            if not isinstance(raw_params, DictConfig):
                raise TypeError(f"Expected DictConfig, got {type(raw_params).__name__}")
            params = raw_params
        return res, params

    # ── Initial pupil loading ────────────────────────────────────────────

    @staticmethod
    def load_initial_pupil(
        param: Union[RunParameters, DictConfig],
    ) -> Optional[np.ndarray]:
        """Load initial pupil from an HDF5 result file.

        If ``param.option.model.init_pupil_file`` is falsy, returns None.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters (``option.model.init_pupil_file`` is used).

        Returns
        -------
        numpy.ndarray or None
            Initial pupil array, or None if no file is configured.
        """
        pupilfile = param.option.model.init_pupil_file
        if not pupilfile:
            return None

        with h5.File(pupilfile, "r") as f:
            return _load_single_channel_pupil(f)

    # ── Internal helpers (image transforms) ──────────────────────────────

    @staticmethod
    def _rearrange_axes(
        images_all: np.ndarray, param: Union[RunParameters, DictConfig]
    ) -> np.ndarray:
        """Transpose *images_all* so the leading axis matches the channel axis."""
        return images_all

    @staticmethod
    def _reshape_insitu(
        images: np.ndarray, param: Union[RunParameters, DictConfig]
    ) -> np.ndarray:
        """Flatten the z-position dimension for insitu (SMLM) data."""
        return images

    @staticmethod
    def _swap_xy(images: np.ndarray, param: Union[RunParameters, DictConfig]) -> np.ndarray:
        """Optionally swap the x and y axes."""
        if not param.swapxy:
            return images
        tmp = np.zeros(
            images.shape[:-2]
            + (images.shape[-1], images.shape[-2]),
            dtype=np.float32,
        )
        tmp[0:] = np.swapaxes(images[0:], -1, -2)
        return tmp

    @staticmethod
    def _flip_if_reverse(
        images: np.ndarray, param: Union[RunParameters, DictConfig]
    ) -> np.ndarray:
        """Flip the z-axis when the stage moves in reverse for bead data."""
        if param.stage_mov_dir == "reverse":
            return np.flip(images, axis=-3)
        return images

    # ── Parameter loading ────────────────────────────────────────────────


def _load_single_channel_pupil(f: h5.File) -> Optional[np.ndarray]:
    """Extract pupil array from an HDF5 result file.

    Returns the pupil array, or None if not found.
    """
    res_group: h5.Group = f["res"]  # type: ignore[assignment]
    try:
        return np.array(res_group["pupil"])  # type: ignore[index]
    except (KeyError, OSError):
        return None


def _redefine(base_param: DictConfig, user_param: DictConfig) -> DictConfig:
    """Recursively merge *user_param* into *base_param*.

    For each key in *user_param*:
    - If both values are dict-like (dict or DictConfig), recurse.
    - Otherwise, overwrite the base value.
    """
    for key, value in user_param.items():
        if (
            isinstance(value, (dict, DictConfig))
            and isinstance(base_param.get(key), (dict, DictConfig))
        ):
            _redefine(base_param[key], value)
        else:
            base_param[key] = value
    return base_param
