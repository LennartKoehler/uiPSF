"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu

Saving results: serialising the PSF fitting output to HDF5 and
generating cubic-spline coefficients for downstream localisation.
"""

import numpy as np
import h5py as h5
import json
from tqdm import tqdm
from omegaconf import OmegaConf

from .learning import psf2cspline_np
from .learning.psf_variables import LocResResult, Positions, PSFResult, ROIsResult


# ── Cubic-spline generation ────────────────────────────────────────────

def gencspline(param, res: PSFResult, psfobj, keyname="I_model"):
    """Generate cubic-spline coefficients from the fitted PSF model.

    Parameters
    ----------
    param : OmegaConf
        Experiment parameters (``channeltype`` is used).
    res : PSFResult
        Result produced by ``psfobj.res2dict``.
    psfobj : PSFInterface
        PSF model object (used for ``sub_psfs`` in multi-channel mode).
    keyname : str
        Key to look up in *res* (``"I_model"`` or
        ``"I_model_reverse"``).

    Returns
    -------
    numpy.ndarray or list
        Cubic-spline coefficients (shape depends on channel type).
    """
    channeltype = param.channeltype

    if channeltype == "single":
        return _gencspline_single(res, keyname)
    if channeltype == "multi":
        return _gencspline_multi(res, psfobj, keyname)
    if channeltype == "4pi":
        return _gencspline_4pi(res, psfobj, keyname)

    return []


def _gencspline_single(res: PSFResult, keyname):
    if keyname not in res:
        return []
    I_model = res[keyname]
    offset = np.min(I_model)
    Imd = I_model - offset
    normf = np.median(np.sum(Imd, axis=(-1, -2)))
    Imd = Imd / normf
    coeff = psf2cspline_np(Imd)
    return coeff.astype(np.float32)


def _gencspline_multi(res: PSFResult, psfobj, keyname):
    ch0 = res.get("channel0")
    if ch0 is None or keyname not in ch0:
        return []
    n_channel = len(psfobj.sub_psfs)
    I_model = np.stack([res["channel" + str(i)][keyname]
                        for i in range(n_channel)])
    offset = np.min(I_model)
    Imd = I_model - offset
    normf = np.max(np.median(np.sum(Imd, axis=(-1, -2)), axis=-1))
    Imd = Imd / normf
    Iall = [psf2cspline_np(Imd[i]) for i in range(n_channel)]
    return np.stack(Iall).astype(np.float32)


def _gencspline_4pi(res: PSFResult, psfobj, keyname):
    ch0 = res.get("channel0")
    if ch0 is None or keyname not in ch0:
        return []
    n_channel = len(psfobj.sub_psfs)

    I_model_list = []
    A_model_list = []
    for i in range(n_channel):
        ch = res["channel" + str(i)]
        I_model_list.append(ch[keyname])
        a_key = "A_model" if keyname == "I_model" else "A_model_reverse"
        A_model_list.append(ch[a_key])

    I_model = np.stack(I_model_list)
    A_model = np.stack(A_model_list)

    offset = np.min(I_model - 2 * np.abs(A_model))
    Imd = I_model - offset
    normf = np.max(np.median(np.sum(Imd[:, 1:-1], axis=(-1, -2)), axis=-1)) * 2.0
    Imd = Imd / normf
    Amd = A_model / normf

    IABall = []
    for i in range(n_channel):
        Ii = Imd[i]
        Ai = 2 * np.real(Amd[i])
        Bi = -2 * np.imag(Amd[i])
        IAB = np.stack([psf2cspline_np(Ai), psf2cspline_np(Bi), psf2cspline_np(Ii)])
        IABall.append(IAB)

    return np.stack(IABall).astype(np.float32)


# ── Result saving ──────────────────────────────────────────────────────

def save_result(param, psfobj, dataobj, fitter, learning_result, loc_result,
                loc_FD=None):
    """Save fitting results, localisation results, and ROI data to an HDF5 file.

    Parameters
    ----------
    param : OmegaConf
        Experiment parameters.
    psfobj : PSFInterface
        Fitted PSF model.
    dataobj : PreprocessedImageData
        Data object with extracted ROIs.
    fitter : Fitter
        Fitter object (used for ``rois`` and ``forward_images``).
    learning_result, loc_result : list
        Fitting / localisation output as returned by :func:`fitting.learn_psf`.
    loc_FD, optional
        Fourier-domain localisation result, or ``None``.

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

    savename = param.savename + "_" + param.PSFtype + "_" + param.channeltype
    res = psfobj.res2dict(learning_result)

    coeff_reverse = gencspline(param, res, psfobj, keyname="I_model_reverse")
    coeff = gencspline(param, res, psfobj)

    locres = _build_locres(loc_result, coeff, coeff_reverse, loc_FD)

    img, _, centers, file_idxs = dataobj.get_image_data()
    img = np.stack(img)
    rois = ROIsResult(
        cor=np.stack(centers),
        fileID=np.stack(file_idxs),
        psf_data=fitter.rois,
        psf_fit=fitter.forward_images,
        image_size=img.shape,
    )

    resfile = savename + ".h5"
    writeh5file(param, resfile, res, locres, rois)

    pbar.postfix[1]["time"] = toc + pbar._time() - pbar.start_t
    pbar.update()
    pbar.close()
    return resfile


def _build_locres(loc_result, coeff, coeff_reverse, loc_FD):
    """Assemble the localization result for HDF5 storage."""
    loc = loc_result.positions
    if isinstance(loc, dict):
        loc = Positions(
            x=loc.get("x"), y=loc.get("y"), z=loc.get("z"),
            zast=loc.get("zast"),
        )

    loc_fd_obj = None
    if loc_FD is not None:
        if isinstance(loc_FD, dict):
            loc_fd_obj = Positions(
                x=loc_FD.get("x"), y=loc_FD.get("y"), z=loc_FD.get("z"),
            )
        else:
            loc_fd_obj = loc_FD

    return LocResResult(
        P=loc_result.parameters,
        CRLB=loc_result.crlb,
        LL=loc_result.log_likelihood,
        coeff=coeff,
        coeff_bead=loc_result.spline_coefficients,
        loc=loc,
        coeff_reverse=coeff_reverse,
        loc_FD=loc_fd_obj,
    )


# ── HDF5 I/O ───────────────────────────────────────────────────────────

def writeh5file(param, filename, res: PSFResult, locres: LocResResult, rois: ROIsResult):
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


def _write_group(group, data):
    """Recursively write a dict into an HDF5 group."""
    for k, v in data.items():
        if isinstance(v, dict):
            sub = group.create_group(k)
            for ki, vi in v.items():
                sub[ki] = vi
        else:
            group[k] = v
