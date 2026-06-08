"""
Handles all output operations: serialising PSF fitting results to HDF5
and generating cubic-spline coefficients for downstream localisation.
"""

from __future__ import annotations

from typing import Optional, Union

import h5py as h5
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .learning import psf2cspline_np
from .learning.psf_variables import LocResResult, PSFResult, ROIsResult
from .learning.psfs.PSFZernikeBased import ZernikePSFResult
from .learning.psfs.PSFInterface import PSFInterface
from .learning.data_representation.PreprocessedImageDataInterface import PreprocessedImageDataInterface
from .io.param import RunParameters


class Writer:
    """Unified interface for all write operations in the PSF-learning
    pipeline."""

    # ── Full save pipeline ───────────────────────────────────────────────

    def save_result(
        self,
        param: Union[RunParameters, DictConfig],
        psf_model: PSFInterface,
        dataobj: PreprocessedImageDataInterface,
        learning_result: ZernikePSFResult,
        loc_result: LocResResult,
        fourier_domain_positions=None,
        forward_images: Optional[np.ndarray] = None,
    ) -> str:
        """Save fitting results, localisation results, and ROI data to an
        HDF5 file.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters.
        psf_model : PSFInterface
            Fitted PSF model.
        dataobj : PreprocessedImageData
            Data object with extracted ROIs.
        learning_result : ZernikePSFResult
            Fitting output as returned by :func:`fitting.learn_psf`.
        loc_result : LocalizationResult or LocResResult
            Localisation output.
        fourier_domain_positions, optional
            Fourier-domain localisation result, or ``None``.
        forward_images : np.ndarray, optional
            Forward images from the learning step. Used for ``modeled_roi_images`` in
            the saved ROIs result.

        Returns
        -------
        str
            Path to the written HDF5 file.
        """
        toc = loc_result.toc
        pbar = tqdm(
            desc="6/6: saving results",
            bar_format="{desc}: [{elapsed}s] {postfix[0]}{postfix[1][time]:>4.2f}s",
            postfix=["total time: ", dict(time=toc)],
        )

        savename = (
            param.savename + "_" + param.PSFtype + "_" + param.channeltype
        )
        psf_result = psf_model.res2dict(learning_result)

        coeff_reverse = self.generate_cspline(
            param, psf_result, psf_model, keyname="psf_model_image_reversed"
        )
        coeff = self.generate_cspline(param, psf_result, psf_model)

        locres = self._build_locres(
            loc_result, coeff, coeff_reverse, fourier_domain_positions
        )

        img, _, centers, file_idxs = dataobj.get_image_data()
        img = np.stack(img)
        modeled_forward_images = forward_images if forward_images is not None else np.array([])
        rois = ROIsResult(
            roi_centers=np.stack(centers),
            source_file_indices=np.stack(file_idxs),
            measured_roi_images=dataobj.measured_roi_images,
            modeled_roi_images=modeled_forward_images,
            full_image_size=img.shape,
        )

        resfile = savename + ".h5"
        self.write_h5(param, resfile, psf_result, locres, rois)

        pbar.postfix[1]["time"] = toc + pbar._time() - pbar.start_t
        pbar.update()
        pbar.close()
        return resfile

    # ── HDF5 I/O ────────────────────────────────────────────────────────

    @staticmethod
    def write_h5(
        param,
        filename: str,
        res: PSFResult,
        locres: LocResResult,
        rois: ROIsResult,
    ) -> None:
        """Write result dataclasses to an HDF5 file.

        Parameters
        ----------
        param : RunParameters or DictConfig
            Experiment parameters (serialised as a JSON attribute).
        filename : str
            Output path.
        res : PSFResult
            PSF fitting result.
        locres : LocResResult
            Localization result.
        rois : ROIsResult
            ROI data.
        """
        from .io.param import RunParameters
        if isinstance(param, RunParameters):
            param_dict = param.to_dict()
        else:
            param_dict = OmegaConf.to_container(param)
        with h5.File(filename, "w") as f:
            f.attrs["params"] = json.dumps(param_dict)
            _write_group(f.create_group("locres"), locres.to_dict())
            _write_group(f.create_group("res"), res.to_dict())
            _write_group(f.create_group("rois"), rois.to_dict())

    # ── Cubic-spline generation ─────────────────────────────────────────

    def generate_cspline(
        self,
        param: Union[RunParameters, DictConfig],
        res: PSFResult,
        psf_model,
        keyname: str = "psf_model_image",
    ):
        """Generate cubic-spline coefficients from the fitted PSF model.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters (``channeltype`` is used).
        res : PSFResult
            Result produced by ``psf_model.res2dict``.
        psf_model : PSFInterface
            PSF model object (used for ``sub_psfs`` in multi-channel
            mode).
        keyname : str
            Key to look up in *res* (``"psf_model_image"`` or
            ``"psf_model_image_reversed"``).

        Returns
        -------
        numpy.ndarray or list
            Cubic-spline coefficients (shape depends on channel type).
        """
        channeltype = param.channeltype

        if channeltype == "single":
            return _gencspline_single(res, keyname)
        if channeltype == "multi":
            return _gencspline_multi(res, psf_model, keyname)
        if channeltype == "4pi":
            return _gencspline_4pi(res, psf_model, keyname)

        return []

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_locres(
        loc_result, coeff, coeff_reverse, fourier_domain_positions
    ) -> LocResResult:
        """Assemble the localization result for HDF5 storage."""
        from .learning.psf_variables import Positions

        loc = loc_result.positions
        if isinstance(loc, dict):
            loc = Positions(
                x=loc.get("x"), y=loc.get("y"), z=loc.get("z"),
                zast=loc.get("zast"),
            )

        loc_fd_obj = None
        if fourier_domain_positions is not None:
            if isinstance(fourier_domain_positions, dict):
                loc_fd_obj = Positions(
                    x=fourier_domain_positions.get("x"), y=fourier_domain_positions.get("y"), z=fourier_domain_positions.get("z"),
                )
            else:
                loc_fd_obj = fourier_domain_positions

        return LocResResult(
            mle_parameters=loc_result.parameters,
            cramer_rao_bounds=loc_result.crlb,
            log_likelihoods=loc_result.log_likelihood,
            spline_coefficients=coeff,
            spline_coefficients_per_bead=loc_result.spline_coefficients,
            localized_positions=loc,
            spline_coefficients_reversed=coeff_reverse,
            fourier_domain_positions=loc_fd_obj,
        )


# ── Module-level helpers ─────────────────────────────────────────────────


def _write_group(group: h5.Group, data: dict) -> None:
    """Recursively write a dict into an HDF5 group."""
    for k, v in data.items():
        if isinstance(v, dict):
            sub = group.create_group(k)
            for ki, vi in v.items():
                sub[ki] = vi
        else:
            group[k] = v


def _gencspline_single(res: PSFResult, keyname: str):
    if keyname not in res:
        return []
    model_image = res[keyname]
    offset = np.min(model_image)
    Imd = model_image - offset
    normf = np.median(np.sum(Imd, axis=(-1, -2)))
    Imd = Imd / normf
    coeff = psf2cspline_np(Imd)
    return coeff.astype(np.float32)


def _gencspline_multi(res: PSFResult, psf_model, keyname: str):
    ch0 = res.get("channel0")
    if ch0 is None or keyname not in ch0:
        return []
    n_channel = len(psf_model.sub_psfs)
    model_image = np.stack(
        [
            res["channel" + str(i)][keyname]
            for i in range(n_channel)
        ]
    )
    offset = np.min(model_image)
    Imd = model_image - offset
    normf = np.max(
        np.median(np.sum(Imd, axis=(-1, -2)), axis=-1)
    )
    Imd = Imd / normf
    Iall = [psf2cspline_np(Imd[i]) for i in range(n_channel)]
    return np.stack(Iall).astype(np.float32)


def _gencspline_4pi(res: PSFResult, psf_model, keyname: str):
    ch0 = res.get("channel0")
    if ch0 is None or keyname not in ch0:
        return []
    n_channel = len(psf_model.sub_psfs)

    model_image_list = []
    A_model_list = []
    for i in range(n_channel):
        ch = res["channel" + str(i)]
        model_image_list.append(ch[keyname])
        a_key = (
            "A_model" if keyname == "psf_model_image" else "A_model_reverse"
        )
        A_model_list.append(ch[a_key])

    model_image = np.stack(model_image_list)
    A_model = np.stack(A_model_list)

    offset = np.min(model_image - 2 * np.abs(A_model))
    Imd = model_image - offset
    normf = (
        np.max(
            np.median(np.sum(Imd[:, 1:-1], axis=(-1, -2)), axis=-1)
        )
        * 2.0
    )
    Imd = Imd / normf
    Amd = A_model / normf

    IABall = []
    for i in range(n_channel):
        Ii = Imd[i]
        Ai = 2 * np.real(Amd[i])
        Bi = -2 * np.imag(Amd[i])
        IAB = np.stack(
            [
                psf2cspline_np(Ai),
                psf2cspline_np(Bi),
                psf2cspline_np(Ii),
            ]
        )
        IABall.append(IAB)

    return np.stack(IABall).astype(np.float32)
