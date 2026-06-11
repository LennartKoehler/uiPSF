"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu

Post-fit analysis: PSF model generation from fitted parameters, Strehl
ratio computation, and FWHM measurement.

This module contains **only** computation — all visualisation is handled
by :class:`~psflearning.plotter.Plotter`.
"""

from __future__ import annotations

import logging

from typing import Tuple

import numpy as np
from dotted_dict import DottedDict
from omegaconf import DictConfig
from typing import Union

from .psf_registry import PSFInfo
from .io.param import RunParameters
from .learning.psfs.IPSFModel import IPSFModel
from .learning.psfs.PSFZernikeBase import PSFContext


def genpsf(
    param: Union[RunParameters, DictConfig],
    f,
    psf_info: PSFInfo,
    Nz: int = 21,
    xsz: int = 21,
    stagepos: float = 1.0,
) -> tuple:
    """Generate a PSF model from fitted parameters.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    f : DottedDict
        Fitting result with ``res`` and ``rois`` attributes.
    psf_info : PSFInfo
        As returned by :func:`psf_registry.get_psf_info`.
    Nz : int
        Number of z-slices.
    xsz : int
        Lateral size of the generated PSF.
    stagepos : float
        Stage position for PSF model generation.

    Returns
    -------
    tuple of (DottedDict, IPSFModel)
        Updated *f* and the PSF model object.
    """
    from .fitting import initialize_psf

    p = param
    dataobj = DottedDict(
        pixelsize_x=p.pixel_size.x,
        pixelsize_y=p.pixel_size.y,
        pixelsize_z=p.pixel_size.z,
        image_size=list(f.rois.full_image_size),
        rois=np.zeros((Nz, xsz, xsz)),
    )

    psf_model = initialize_psf(param, psf_info)

    _genpsf_single(psf_model, dataobj, f, p, Nz, stagepos)

    return f, psf_model


def _genpsf_single(psf_model, dataobj, f, p, Nz, stagepos):
    sigma = f.res.gaussian_blur_sigma
    Zcoeff_magnitude = f.res.zernike_coefficients[0].reshape(
        f.res.zernike_coefficients[0].shape + (1, 1)
    )
    Zcoeff_phase = f.res.zernike_coefficients[1].reshape(
        f.res.zernike_coefficients[1].shape + (1, 1)
    )

    pupil_field = IPSFModel.compute_pupil_field(dataobj, p.option, "scalar", Nz=Nz)
    context = PSFContext(params=p.option, pupil_field=pupil_field)
    f.res.psf_model_image, _ = psf_model.genpsfmodel(sigma, context, Zcoeff_magnitude=Zcoeff_magnitude, Zcoeff_phase=Zcoeff_phase)


def calstrehlratio(
    param: Union[RunParameters, DictConfig], f, psf_info: PSFInfo, xsz: int = 31
):
    """Compute the Strehl ratio of the fitted PSF.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    f : DottedDict
        Fitting result.
    psf_info : PSFInfo
        As returned by :func:`psf_registry.get_psf_info`.
    xsz : int
        Lateral size used for generating the PSF.

    Returns
    -------
    float
        Strehl ratio.
    """
    f1 = f.copy()
    p = param

    return _strehl_single(p, f1, f, psf_info, xsz)


def _strehl_single(p, f1, f, psf_info, xsz):
    f1.res.zernike_coefficients[1, :, 0:4] = 0.0
    f1, _ = genpsf(p, f1, psf_info, Nz=1, xsz=xsz)
    I1 = f1.res.psf_model_image[0, xsz // 2, xsz // 2] / np.sum(
        f1.res.psf_model_image
    )

    f1.res.zernike_coefficients = np.zeros(
        f1.res.zernike_coefficients.shape, dtype=np.float32
    )
    f1.res.zernike_coefficients[0, :, 0] = 1
    f1, _ = genpsf(p, f1, psf_info, Nz=1, xsz=xsz)
    I0 = f1.res.psf_model_image[0, xsz // 2, xsz // 2] / np.sum(
        f1.res.psf_model_image
    )

    strehlratio = np.float32(I1 / I0)
    logging.info("Strehl ratio: %s", strehlratio)
    return strehlratio


def calfwhm(
    param: Union[RunParameters, DictConfig], f, psf_info: PSFInfo
) -> Tuple:
    """Compute the full-width at half-maximum of the fitted PSF.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    f : DottedDict
        Fitting result.
    psf_info : PSFInfo
        As returned by :func:`psf_registry.get_psf_info`.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        ``(fwhmx, fwhmy, fwhmz)`` in nanometres.
    """
    f1 = f.copy()
    p = param

    return _fwhm_single(p, f, f1, psf_info)


def _fwhm_single(p, f, f1, psf_info):
    psf_model_image = f.res.psf_model_image
    Ix, xh, Iy, yh, Iz, zh = getfwhm(psf_model_image)
    fwhmx = np.diff(xh) * p.pixel_size.x * 1e3
    fwhmy = np.diff(yh) * p.pixel_size.y * 1e3
    fwhmz = np.diff(zh) * p.pixel_size.z * 1e3

    return fwhmx, fwhmy, fwhmz


def getfwhm(psf_model_image: np.ndarray) -> Tuple:
    """Compute the FWHM along x, y, and z of a 3-D PSF volume.

    Returns
    -------
    tuple
        ``(Ix, xh, Iy, yh, Iz, zh)`` where each ``*h`` is a 2-element
        array with the half-maximum crossing positions.
    """
    cor = np.unravel_index(np.argmax(psf_model_image), psf_model_image.shape)

    Ix = psf_model_image[cor[0], cor[1]]
    xh = get1dfwhm(Ix, cor[2])
    Iy = psf_model_image[cor[0], :, cor[2]]
    yh = get1dfwhm(Iy, cor[1])

    Iz = psf_model_image[:, cor[1], cor[2]]
    zh = get1dfwhm(Iz, cor[0])

    return Ix, xh, Iy, yh, Iz, zh


def get1dfwhm(I: np.ndarray, cor: int) -> np.ndarray:
    """Find the half-maximum crossings of 1-D intensity profile *I*
    around the peak at index *cor*.

    Returns
    -------
    numpy.ndarray
        Two-element array ``[left_crossing, right_crossing]``.
    """
    Imaxh = np.max(I) / 2

    x1 = np.argsort(np.abs(I[:cor] - Imaxh))[0]
    x1 = [x1, x1 - 1] if I[x1] > Imaxh else [x1, x1 + 1]

    x2 = np.argsort(np.abs(I[cor:] - Imaxh))[0] + cor
    x2 = [x2, x2 + 1] if I[x2] > Imaxh else [x2, x2 - 1]

    xh1 = _interpolate_crossing(x1, I, Imaxh)
    xh2 = _interpolate_crossing(x2, I, Imaxh)

    return np.hstack([xh1, xh2])


def _interpolate_crossing(
    x_pair: list, I: np.ndarray, half_max: float
) -> float:
    """Linearly interpolate the half-max crossing between two sample
    points."""
    g = np.diff(x_pair) / np.diff(I[x_pair])
    xh = g * (half_max - I[x_pair[0]]) + x_pair[0]
    x_arr = np.array(x_pair, dtype=np.float64)
    return np.clip(xh, np.min(x_arr), np.max(x_arr))
