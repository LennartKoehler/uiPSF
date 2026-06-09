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
        self, param: Union[RunParameters, DictConfig], frange: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Load raw image stacks from disk.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters (file paths, channel type, PSF type, ...).
        frange : tuple of (int, int), optional
            Slice ``filelist[frange[0]:frange[1]]`` to restrict the files
            loaded.

        Returns
        -------
        numpy.ndarray
            Image array with axes arranged for the requested *channeltype*
            and *PSFtype*.
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
        channeltype: Optional[str] = None,
        sysfile: Optional[str] = None,
    ) -> RunParameters:
        """Combine a base configuration with PSF / channel / system overrides.

        Parameters
        ----------
        basefile : str
            Base configuration name (without ``.yaml``), resolved relative
            to the package ``config/`` directory.
        psftype : str, optional
            PSF type override (e.g. ``"zernike"``, ``"insitu_FD"``).
        channeltype : str, optional
            Channel type override (e.g. ``"1ch"``, ``"2ch"``, ``"4pi"``).
        sysfile : str, optional
            System type override (e.g. ``"M2"``, ``"TP"``).

        Returns
        -------
        RunParameters
            Merged parameters.
        """
        from .io.param import combine as _combine
        return _combine(basefile, psftype=psftype, channeltype=channeltype, sysfile=sysfile)

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
        param: Union[RunParameters, DictConfig], psf_model, dataobj
    ) -> None:
        """Load initial pupil / PSF / Zernike coefficients from an HDF5
        file into *psf_model* in-place.

        If ``param.option.model.init_pupil_file`` is falsy, this is a
        no-op.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters (``channeltype`` and
            ``option.model.init_pupil_file`` are used).
        psf_model : PSFInterface
            PSF model object to populate with initial values.
        dataobj : PreprocessedImageData
            Data object (used for channel count in multi-channel mode).
        """
        pupilfile = param.option.model.init_pupil_file
        if not pupilfile:
            return

        channeltype = param.channeltype

        with h5.File(pupilfile, "r") as f:
            if channeltype == "single":
                _load_single_channel_pupil(psf_model, f)
            else:
                _load_multi_channel_pupil(psf_model, f, channeltype, dataobj)

    # ── Internal helpers (image transforms) ──────────────────────────────

    @staticmethod
    def _rearrange_axes(
        images_all: np.ndarray, param: Union[RunParameters, DictConfig]
    ) -> np.ndarray:
        """Transpose *images_all* so the leading axis matches
        *channeltype*."""
        channeltype = param.channeltype
        psf_type = param.PSFtype

        if channeltype == "4pi":
            if "insitu" in psf_type:
                return np.transpose(images_all, (1, 0, 2, 3, 4))
            if param.varname:
                return np.transpose(images_all, (1, 0, 2, 3, 4, 5))
            return np.transpose(images_all, (1, 0, 3, 2, 4, 5))

        if channeltype == "multi":
            images = np.transpose(images_all, (1, 0, 2, 3, 4))
            images = _reorder_ref_channel(images, param)
            return images

        return images_all

    @staticmethod
    def _reshape_insitu(
        images: np.ndarray, param: Union[RunParameters, DictConfig]
    ) -> np.ndarray:
        """Flatten the z-position dimension for insitu (SMLM) data."""
        if "insitu" not in param.PSFtype:
            return images

        channeltype = param.channeltype
        if channeltype == "single":
            return images.reshape(
                -1, images.shape[-2], images.shape[-1]
            )
        return images.reshape(
            images.shape[0],
            -1,
            images.shape[-2],
            images.shape[-1],
        )

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
        if param.stage_mov_dir == "reverse" and param.datatype == "bead":
            return np.flip(images, axis=-3)
        return images

    # ── Parameter loading ────────────────────────────────────────────────


def _reorder_ref_channel(images: np.ndarray, param: Union[RunParameters, DictConfig]) -> np.ndarray:
    """Swap the first channel with *ref_channel* so the reference comes
    first."""
    ref = param.ref_channel
    n_channels = images.shape[0]

    defocus = [
        param.option.multi.defocus_offset
        + i * param.option.multi.defocus_delay
        for i in range(n_channels)
    ]
    defocus[0], defocus[ref] = defocus[ref], defocus[0]
    param.option.multi.defocus = defocus

    idx = list(range(n_channels))
    idx[0], idx[ref] = idx[ref], idx[0]
    return images[idx]


def _load_single_channel_pupil(psf_model, f: h5.File) -> None:
    res_group: h5.Group = f["res"]  # type: ignore[assignment]
    try:
        psf_model.initial_pupil = np.array(res_group["pupil"])  # type: ignore[index]
    except (KeyError, OSError):
        pass

    try:
        psf_model.z_offset = np.array(res_group["zoffset"])  # type: ignore[index]
    except (KeyError, OSError):
        pass

    try:
        psf_model.initial_psf_image = np.array(
            res_group["I_model_reverse"]  # type: ignore[index]
        ).astype(np.float32)
    except (KeyError, OSError):
        try:
            psf_model.initial_psf_image = np.array(
                res_group["I_model"]  # type: ignore[index]
            ).astype(np.float32)
        except (KeyError, OSError):
            pass

    try:
        psf_model.initial_zernike_coefficients = np.array(
            res_group["zernike_coeff"]  # type: ignore[index]
        ).astype(np.float32)
    except (KeyError, OSError):
        pass


def _load_multi_channel_pupil(
    psf_model, f: h5.File, channeltype: str, dataobj
) -> None:
    n_channels = len(dataobj.channels)
    res_group: h5.Group = f["res"]  # type: ignore[assignment]
    psf_model.initial_pupil = [None] * n_channels
    psf_model.initial_psf_image = [None] * n_channels
    if channeltype == "4pi":
        psf_model.initial_interference_amplitude = [None] * n_channels

    for k in range(n_channels):
        ch: h5.Group = res_group["channel" + str(k)]  # type: ignore[assignment]
        try:
            psf_model.initial_pupil[k] = np.array(ch["pupil"])  # type: ignore[index]
        except (KeyError, OSError):
            pass
        try:
            psf_model.z_offset = np.array(ch["zoffset"])  # type: ignore[index]
        except (KeyError, OSError):
            pass
        psf_model.initial_psf_image[k] = np.array(ch["I_model"]).astype(np.float32)  # type: ignore[index]
        if channeltype == "4pi":
            psf_model.initial_interference_amplitude[k] = np.array(ch["A_model"]).astype(  # type: ignore[index]
                np.complex64
            )


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
