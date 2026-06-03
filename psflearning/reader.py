"""
Handles all input operations: loading images, parameters, initial pupil
state, and previously saved results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import h5py as h5
import hdfdict
import numpy as np
from dotted_dict import DottedDict
from omegaconf import DictConfig, OmegaConf

from psflearning.learning.data_representation.PreprocessedImageDataSingleChannel import PreprocessedImageDataSingleChannel

from .dataloader import get_loader
# from .learning import (
#     PreprocessedImageDataMultiChannel,
#     PreprocessedImageDataMultiChannel_smlm,
#     PreprocessedImageDataSingleChannel,
#     PreprocessedImageDataSingleChannel_smlm,
# )


class Reader:
    """Unified interface for all read operations in the PSF-learning pipeline."""

    # ── Image loading ────────────────────────────────────────────────────

    def read_images(
        self, param: DictConfig, frange: Optional[Tuple[int, int]] = None
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

        print(images.shape)
        return images

    # ── Data preprocessing ───────────────────────────────────────────────

    def prep_data(self, param: DictConfig, images: np.ndarray):
        """Detect beads / localisations in *images* and return a
        :class:`PreprocessedImageData` object.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters (ROI, pixel sizes, FOV, ...).
        images : numpy.ndarray
            Image array as returned by :meth:`read_images`.

        Returns
        -------
        PreprocessedImageData
            Data object with extracted ROIs ready for PSF fitting.
        """
        roi_size = param.roi.roi_size
        fov = list(param.FOV.values())
        skew_const = param.LLS.skew_const
        is_volume = param.PSFtype == "voxel"

        images = self._crop_fov(images, fov)

        dataobj = self._create_dataobj(images, param)

        fov_param = None if fov[2] == 0 else fov
        skew_param = (
            None
            if (skew_const[0] == 0.0 and skew_const[1] == 0.0)
            else skew_const
        )

        dataobj.process(
            roi_size=roi_size,
            gaus_sigma=param.roi.gauss_sigma,
            min_border_dist=list(np.array(roi_size) // 2 + 1),
            min_center_dist=np.max(roi_size),
            FOV=fov_param,
            max_threshold=param.roi.peak_height,
            max_kernel=param.roi.max_kernel,
            pixelsize_x=param.pixel_size.x,
            pixelsize_y=param.pixel_size.y,
            pixelsize_z=param.pixel_size.z,
            bead_radius=param.roi.bead_radius,
            modulation_period=param.fpi.modulation_period,
            plot=param.plotall,
            padPSF=False,
            isVolume=is_volume,
            skew_const=skew_param,
            max_bead_number=param.roi.max_bead_number,
        )
        return dataobj

    # ── Parameter loading ────────────────────────────────────────────────

    @staticmethod
    def read_params(path: Union[str, Path]) -> DictConfig:
        """Load an OmegaConf configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        DictConfig
            Loaded parameters.
        """
        return OmegaConf.load(path)

    @staticmethod
    def combine_params(
        basefile: str,
        psftype: Optional[str] = None,
        channeltype: Optional[str] = None,
        sysfile: Optional[str] = None,
    ) -> DictConfig:
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
        DictConfig
            Merged parameters.
        """
        import os

        this_path = os.path.dirname(os.path.abspath(__file__))
        pkg_path = os.path.dirname(this_path)

        fparam = OmegaConf.load(
            pkg_path + "/config/" + basefile + ".yaml"
        ).Params

        if psftype is not None:
            psf_param = OmegaConf.load(
                pkg_path + "/config/psftype/" + psftype + ".yaml"
            ).Params
            fparam = _redefine(fparam, psf_param)

        if channeltype is not None:
            ch_param = OmegaConf.load(
                pkg_path + "/config/channeltype/" + channeltype + ".yaml"
            ).Params
            fparam = _redefine(fparam, ch_param)

        if sysfile is not None:
            sys_param = OmegaConf.load(
                pkg_path + "/config/systemtype/" + sysfile + ".yaml"
            ).Params
            fparam = _redefine(fparam, sys_param)

        if psftype == "zernike" and channeltype == "4pi":
            fparam.PSFtype = "zernike"

        if "insitu" in (psftype or ""):
            fparam.roi.gauss_sigma[-1] = max(
                [4, fparam.roi.gauss_sigma[-1]]
            )
            fparam.roi.gauss_sigma[-2] = max(
                [4, fparam.roi.gauss_sigma[-2]]
            )
            fparam.roi.max_kernel[-1] = max(
                [5, fparam.roi.max_kernel[-1]]
            )
            fparam.roi.max_kernel[-2] = max(
                [5, fparam.roi.max_kernel[-2]]
            )

        if "FD" in (psftype or ""):
            fparam.option.model.bin = 1

        return fparam

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
            params = OmegaConf.create(f.attrs["params"])
        return res, params

    # ── Initial pupil loading ────────────────────────────────────────────

    @staticmethod
    def load_initial_pupil(
        param: DictConfig, psfobj, dataobj
    ) -> None:
        """Load initial pupil / PSF / Zernike coefficients from an HDF5
        file into *psfobj* in-place.

        If ``param.option.model.init_pupil_file`` is falsy, this is a
        no-op.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters (``channeltype`` and
            ``option.model.init_pupil_file`` are used).
        psfobj : PSFInterface
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
                _load_single_channel_pupil(psfobj, f)
            else:
                _load_multi_channel_pupil(psfobj, f, channeltype, dataobj)

    # ── Internal helpers (image transforms) ──────────────────────────────

    @staticmethod
    def _rearrange_axes(
        images_all: np.ndarray, param: DictConfig
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
        images: np.ndarray, param: DictConfig
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
    def _swap_xy(images: np.ndarray, param: DictConfig) -> np.ndarray:
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
        images: np.ndarray, param: DictConfig
    ) -> np.ndarray:
        """Flip the z-axis when the stage moves in reverse for bead data."""
        if param.stage_mov_dir == "reverse" and param.datatype == "bead":
            return np.flip(images, axis=-3)
        return images

    @staticmethod
    def _crop_fov(images: np.ndarray, fov: list) -> np.ndarray:
        """Slice *images* along the z-axis according to *fov*."""
        zstart = fov[-3]
        zend = images.shape[-3] + fov[-2]
        zstep = fov[-1]
        zind = range(zstart, zend, zstep)
        ims = np.swapaxes(images, 0, -3)
        ims = ims[zind]
        return np.swapaxes(ims, 0, -3)

    @staticmethod
    def _create_dataobj(images: np.ndarray, param: DictConfig):
        """Instantiate the correct :class:`PreprocessedImageData`
        subclass."""
        channeltype = param.channeltype
        is_insitu = "insitu" in param.PSFtype

        if channeltype == "single":
            cls = PreprocessedImageDataSingleChannel
            # cls = (
            #     PreprocessedImageDataSingleChannel_smlm
            #     if is_insitu
            #     else PreprocessedImageDataSingleChannel
            # )
            return cls(images)

        # if channeltype == "4pi":
        #     if is_insitu:
        #         return PreprocessedImageDataMultiChannel_smlm(
        #             images,
        #             PreprocessedImageDataSingleChannel_smlm,
        #             is4pi=True,
        #         )
        #     return PreprocessedImageDataMultiChannel(
        #         images,
        #         PreprocessedImageDataSingleChannel,
        #         is4pi=True,
        #     )
        #
        # if is_insitu:
        #     return PreprocessedImageDataMultiChannel_smlm(
        #         images, PreprocessedImageDataSingleChannel_smlm
        #     )
        # return PreprocessedImageDataMultiChannel(
        #     images, PreprocessedImageDataSingleChannel
        # )


# ── Module-level helpers ─────────────────────────────────────────────────


def _reorder_ref_channel(images: np.ndarray, param: DictConfig) -> np.ndarray:
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


def _load_single_channel_pupil(psfobj, f: h5.File) -> None:
    """Populate initial PSF state for a single-channel model."""
    try:
        psfobj.initpupil = np.array(f["res"]["pupil"])
    except (KeyError, OSError):
        pass

    try:
        psfobj.Zoffset = np.array(f["res"]["zoffset"])
    except (KeyError, OSError):
        pass

    try:
        psfobj.initpsf = np.array(
            f["res"]["I_model_reverse"]
        ).astype(np.float32)
    except (KeyError, OSError):
        try:
            psfobj.initpsf = np.array(
                f["res"]["I_model"]
            ).astype(np.float32)
        except (KeyError, OSError):
            pass

    try:
        psfobj.initzcoeff = np.array(
            f["res"]["zernike_coeff"]
        ).astype(np.float32)
    except (KeyError, OSError):
        pass


def _load_multi_channel_pupil(
    psfobj, f: h5.File, channeltype: str, dataobj
) -> None:
    """Populate initial PSF state for a multi-channel / 4pi model."""
    n_channels = len(dataobj.channels)
    psfobj.initpupil = [None] * n_channels
    psfobj.initpsf = [None] * n_channels
    if channeltype == "4pi":
        psfobj.initA = [None] * n_channels

    for k in range(n_channels):
        ch = f["res"]["channel" + str(k)]
        try:
            psfobj.initpupil[k] = np.array(ch["pupil"])
        except (KeyError, OSError):
            pass
        try:
            psfobj.Zoffset = np.array(ch["zoffset"])
        except (KeyError, OSError):
            pass
        psfobj.initpsf[k] = np.array(ch["I_model"]).astype(np.float32)
        if channeltype == "4pi":
            psfobj.initA[k] = np.array(ch["A_model"]).astype(
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
