"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu

Registry mapping PSF type strings to their implementation classes and
corresponding loss functions.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from .learning import (
    PSFVolumeBased,
    PSFPupilBased,
    PSFZernikeBased,
    PSFZernikeBased_FD,
    PSFVolumeBased4pi,
    PSFPupilBased4pi,
    PSFZernikeBased4pi,
    PSFMultiChannel,
    PSFMultiChannel_smlm,
    PSFMultiChannel4pi,
    PSFZernikeBased_vector_smlm,
    PSFPupilBased_vector_smlm,
    PSFZernikeBased_FD_smlm,
    PSFMultiChannel4pi_smlm,
    PSFZernikeBased4pi_smlm,
    mse_real,
    mse_real_zernike,
    mse_real_zernike_FD,
    mse_real_zernike_smlm,
    mse_real_pupil_smlm,
    mse_real_zernike_FD_smlm,
    mse_real_4pi,
    mse_zernike_4pi,
    mse_zernike_4pi_smlm,
    mse_real_pupil,
    mse_pupil_4pi,
    mse_real_All,
    mse_real_4pi_All,
)


# ── Standard (non-4pi) PSF models ──────────────────────────────────────
PSF_CLASS: dict[str, type[Any]] = {
    "voxel": PSFVolumeBased,
    "pupil": PSFPupilBased,
    "pupil_vector": PSFPupilBased,
    "zernike": PSFZernikeBased,
    "zernike_vector": PSFZernikeBased,
    "zernike_FD": PSFZernikeBased_FD,
    "zernike_vector_FD": PSFZernikeBased_FD,
    "insitu_zernike": PSFZernikeBased_vector_smlm,
    "insitu_pupil": PSFPupilBased_vector_smlm,
    "insitu_FD": PSFZernikeBased_FD_smlm,
}

LOSS_FUNCTION: dict[str, Any] = {
    "voxel": mse_real,
    "pupil": mse_real_pupil,
    "pupil_vector": mse_real_pupil,
    "zernike": mse_real_zernike,
    "zernike_vector": mse_real_zernike,
    "zernike_FD": mse_real_zernike_FD,
    "zernike_vector_FD": mse_real_zernike_FD,
    "insitu_zernike": mse_real_zernike_smlm,
    "insitu_pupil": mse_real_pupil_smlm,
    "insitu_FD": mse_real_zernike_FD_smlm,
}


# ── 4pi PSF models ─────────────────────────────────────────────────────
PSF_CLASS_4PI: dict[str, type[Any]] = {
    "voxel": PSFVolumeBased4pi,
    "pupil": PSFPupilBased4pi,
    "zernike": PSFZernikeBased4pi,
    "insitu_zernike": PSFZernikeBased4pi_smlm,
}

LOSS_FUNCTION_4PI: dict[str, Any] = {
    "voxel": mse_real_4pi,
    "pupil": mse_pupil_4pi,
    "zernike": mse_zernike_4pi,
    "insitu_zernike": mse_zernike_4pi_smlm,
}


# ── Multi-channel PSF classes ──────────────────────────────────────────
MULTI_CHANNEL_CLASS: dict[str, type[Any]] = {
    "standard": PSFMultiChannel,
    "insitu": PSFMultiChannel_smlm,
}

MULTI_CHANNEL_4PI_CLASS: dict[str, type[Any]] = {
    "standard": PSFMultiChannel4pi,
    "insitu": PSFMultiChannel4pi_smlm,
}

MULTI_LOSS: dict[str, Any] = {
    "multi": mse_real_All,
    "4pi": mse_real_4pi_All,
}


def get_psf_info(param: DictConfig) -> dict[str, Any]:
    """Resolve *param* into the concrete PSF class, multi-channel class,
    loss function, and multi-channel loss function.

    Returns
    -------
    dict
        Keys: ``psf_class``, ``psf_class_multi``, ``loss_fun``,
        ``loss_fun_multi``.
    """
    psf_type = param.PSFtype
    channel_type = param.channeltype

    if channel_type == "4pi":
        psf_class = PSF_CLASS_4PI[psf_type]
        loss_fun = LOSS_FUNCTION_4PI[psf_type]
        multi_key = "insitu" if "insitu" in psf_type else "standard"
        psf_class_multi = MULTI_CHANNEL_4PI_CLASS[multi_key]
        loss_fun_multi = MULTI_LOSS["4pi"]
    elif channel_type == "multi":
        psf_class = PSF_CLASS[psf_type]
        loss_fun = LOSS_FUNCTION[psf_type]
        multi_key = "insitu" if "insitu" in psf_type else "standard"
        psf_class_multi = MULTI_CHANNEL_CLASS[multi_key]
        loss_fun_multi = MULTI_LOSS["multi"]
    else:  # single
        psf_class = PSF_CLASS[psf_type]
        loss_fun = LOSS_FUNCTION[psf_type]
        psf_class_multi = None
        loss_fun_multi = None

    return {
        "psf_class": psf_class,
        "psf_class_multi": psf_class_multi,
        "loss_fun": loss_fun,
        "loss_fun_multi": loss_fun_multi,
    }
