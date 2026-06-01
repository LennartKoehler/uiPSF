"""
Handles all output operations: serialising PSF fitting results to HDF5
and generating cubic-spline coefficients for downstream localisation.
"""

from __future__ import annotations

from typing import Optional

import h5py as h5
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .learning import psf2cspline_np


class Writer:
    """Unified interface for all write operations in the PSF-learning
    pipeline."""

    # ── Full save pipeline ───────────────────────────────────────────────

    def save_result(
        self,
        param: DictConfig,
        psfobj,
        dataobj,
        fitter,
        learning_result: list,
        loc_result: list,
        loc_FD=None,
    ) -> str:
        """Save fitting results, localisation results, and ROI data to an
        HDF5 file.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters.
        psfobj : PSFInterface
            Fitted PSF model.
        dataobj : PreprocessedImageData
            Data object with extracted ROIs.
        fitter : Fitter
            Fitter object (used for ``rois`` and ``forward_images``).
        learning_result : list
            Fitting output as returned by :func:`fitting.learn_psf`.
        loc_result : list
            Localisation output.
        loc_FD, optional
            Fourier-domain localisation result, or ``None``.

        Returns
        -------
        str
            Path to the written HDF5 file.
        """
        toc = loc_result[-2]
        pbar = tqdm(
            desc="6/6: saving results",
            bar_format="{desc}: [{elapsed}s] {postfix[0]}{postfix[1][time]:>4.2f}s",
            postfix=["total time: ", dict(time=toc)],
        )

        savename = (
            param.savename + "_" + param.PSFtype + "_" + param.channeltype
        )
        res_dict = psfobj.res2dict(learning_result)

        coeff_reverse = self.generate_cspline(
            param, res_dict, psfobj, keyname="I_model_reverse"
        )
        coeff = self.generate_cspline(param, res_dict, psfobj)

        locres_dict = self._build_locres_dict(
            loc_result, coeff, coeff_reverse, loc_FD
        )

        img, _, centers, file_idxs = dataobj.get_image_data()
        img = np.stack(img)
        rois_dict = dict(
            cor=np.stack(centers),
            fileID=np.stack(file_idxs),
            psf_data=fitter.rois,
            psf_fit=fitter.forward_images,
            image_size=img.shape,
        )

        resfile = savename + ".h5"
        self.write_h5(param, resfile, res_dict, locres_dict, rois_dict)

        pbar.postfix[1]["time"] = toc + pbar._time() - pbar.start_t
        pbar.update()
        pbar.close()
        return resfile

    # ── HDF5 I/O ────────────────────────────────────────────────────────

    @staticmethod
    def write_h5(
        param: DictConfig,
        filename: str,
        res_dict: dict,
        locres_dict: dict,
        rois_dict: dict,
    ) -> None:
        """Write result dictionaries to an HDF5 file.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters (serialised as a JSON attribute).
        filename : str
            Output path.
        res_dict, locres_dict, rois_dict : dict
            Data to write into the ``res``, ``locres`` and ``rois``
            groups.
        """
        with h5.File(filename, "w") as f:
            f.attrs["params"] = json.dumps(
                OmegaConf.to_container(param)
            )
            _write_group(f.create_group("locres"), locres_dict)
            _write_group(f.create_group("res"), res_dict)
            _write_group(f.create_group("rois"), rois_dict)

    # ── Cubic-spline generation ─────────────────────────────────────────

    def generate_cspline(
        self,
        param: DictConfig,
        res_dict: dict,
        psfobj,
        keyname: str = "I_model",
    ):
        """Generate cubic-spline coefficients from the fitted PSF model.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters (``channeltype`` is used).
        res_dict : dict
            Result dictionary produced by ``psfobj.res2dict``.
        psfobj : PSFInterface
            PSF model object (used for ``sub_psfs`` in multi-channel
            mode).
        keyname : str
            Key to look up in *res_dict* (``"I_model"`` or
            ``"I_model_reverse"``).

        Returns
        -------
        numpy.ndarray or list
            Cubic-spline coefficients (shape depends on channel type).
        """
        channeltype = param.channeltype

        if channeltype == "single":
            return _gencspline_single(res_dict, keyname)
        if channeltype == "multi":
            return _gencspline_multi(res_dict, psfobj, keyname)
        if channeltype == "4pi":
            return _gencspline_4pi(res_dict, psfobj, keyname)

        return []

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _build_locres_dict(
        loc_result: list, coeff, coeff_reverse, loc_FD
    ) -> dict:
        """Assemble the localisation-result dictionary for HDF5 storage."""
        d = dict(
            P=loc_result[0],
            CRLB=loc_result[1],
            LL=loc_result[2],
            coeff=coeff,
            coeff_bead=loc_result[3],
            loc=loc_result[-1],
            coeff_reverse=coeff_reverse,
        )
        if loc_FD is not None:
            d["loc_FD"] = loc_FD
        return d


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


def _gencspline_single(res_dict: dict, keyname: str):
    if keyname not in res_dict:
        return []
    I_model = res_dict[keyname]
    offset = np.min(I_model)
    Imd = I_model - offset
    normf = np.median(np.sum(Imd, axis=(-1, -2)))
    Imd = Imd / normf
    coeff = psf2cspline_np(Imd)
    return coeff.astype(np.float32)


def _gencspline_multi(res_dict: dict, psfobj, keyname: str):
    if keyname not in res_dict.get("channel0", {}):
        return []
    n_channel = len(psfobj.sub_psfs)
    I_model = np.stack(
        [
            res_dict["channel" + str(i)][keyname]
            for i in range(n_channel)
        ]
    )
    offset = np.min(I_model)
    Imd = I_model - offset
    normf = np.max(
        np.median(np.sum(Imd, axis=(-1, -2)), axis=-1)
    )
    Imd = Imd / normf
    Iall = [psf2cspline_np(Imd[i]) for i in range(n_channel)]
    return np.stack(Iall).astype(np.float32)


def _gencspline_4pi(res_dict: dict, psfobj, keyname: str):
    if keyname not in res_dict.get("channel0", {}):
        return []
    n_channel = len(psfobj.sub_psfs)

    I_model_list = []
    A_model_list = []
    for i in range(n_channel):
        ch = res_dict["channel" + str(i)]
        I_model_list.append(ch[keyname])
        a_key = (
            "A_model" if keyname == "I_model" else "A_model_reverse"
        )
        A_model_list.append(ch[a_key])

    I_model = np.stack(I_model_list)
    A_model = np.stack(A_model_list)

    offset = np.min(I_model - 2 * np.abs(A_model))
    Imd = I_model - offset
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
