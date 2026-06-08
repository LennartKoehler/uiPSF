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

from typing import Optional, Tuple

import numpy as np
from dotted_dict import DottedDict
from omegaconf import DictConfig
from typing import Union

from .learning.psf_variables import PSFInfo
from .io.param import RunParameters


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
        Stage position for insitu models.

    Returns
    -------
    tuple of (DottedDict, PSFInterface)
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

    if p.channeltype == "single":
        _genpsf_single(psf_model, dataobj, f, p, Nz, stagepos)
    elif p.channeltype == "multi":
        _genpsf_multi(psf_model, dataobj, f, p, Nz, stagepos)

    return f, psf_model


def _genpsf_single(psf_model, dataobj, f, p, Nz, stagepos):
    sigma = f.res.gaussian_blur_sigma
    Zcoeff_magnitude = f.res.zernike_coefficients[0].reshape(
        f.res.zernike_coefficients[0].shape + (1, 1)
    )
    Zcoeff_phase = f.res.zernike_coefficients[1].reshape(
        f.res.zernike_coefficients[1].shape + (1, 1)
    )
    psf_model.data = dataobj

    if "insitu" in p.PSFtype:
        psf_model.stagepos = stagepos / p.pixel_size.z
        psf_model.estzoffset(Nz=Nz)
    elif psf_model.psftype == "vector":
        psf_model.calpupilfield("vector", Nz=Nz)
    else:
        psf_model.calpupilfield("scalar", Nz=Nz)

    if "FD" in p.PSFtype:
        f.res.psf_model_image = _genpsf_fd(psf_model, f, sigma)
    else:
        f.res.psf_model_image, _ = psf_model.genpsfmodel(sigma, Zcoeff_magnitude=Zcoeff_magnitude, Zcoeff_phase=Zcoeff_phase)


def _genpsf_fd(psf_model, f, sigma):
    """Generate a field-dependent PSF model (batched over positions)."""
    img_size = f.rois.full_image_size
    zmap = f.res.zernike_map
    dx = img_size[-1] / zmap.shape[-1] / 2
    dy = img_size[-2] / zmap.shape[-2] / 2
    xrange = np.linspace(
        dx, img_size[-1] - dx, zmap.shape[-1], dtype=np.float32
    )
    yrange = np.linspace(
        dy, img_size[-2] - dy, zmap.shape[-2], dtype=np.float32
    )
    xx, yy = np.meshgrid(xrange, yrange)
    cor = np.vstack((xx.flatten(), yy.flatten())).transpose()

    batchsize = 200
    ind = list(
        np.int32(
            np.linspace(0, cor.shape[0], cor.shape[0] // batchsize + 2)
        )
    )
    psf_model_image = None
    for i in range(len(ind) - 1):
        I0, _, _ = psf_model.genpsfmodel(
            sigma, Zmap=zmap, cor=cor[ind[i] : ind[i + 1]]
        )
        psf_model_image = I0 if psf_model_image is None else np.vstack((psf_model_image, I0))
    return psf_model_image


def _genpsf_multi(psf_model, dataobj, f, p, Nz, stagepos):
    n_channel = f.rois.roi_centers.shape[0]
    psf_model.sub_psfs = [None] * n_channel

    for i in range(n_channel):
        sub_psf_model = psf_model.psftype(options=psf_model.options)
        sub_psf_model.psftype = psf_model.PSFtype
        psf_model.sub_psfs[i] = sub_psf_model

        sigma = f.res["channel" + str(i)].gaussian_blur_sigma
        Zcoeff_magnitude = f.res["channel" + str(i)].zernike_coefficients[0].reshape(
            f.res["channel" + str(i)].zernike_coefficients[0].shape + (1, 1)
        )
        Zcoeff_phase = f.res["channel" + str(i)].zernike_coefficients[1].reshape(
            f.res["channel" + str(i)].zernike_coefficients[1].shape + (1, 1)
        )
        sub_psf_model.data = dataobj

        if "insitu" in p.PSFtype:
            sub_psf_model.stagepos = stagepos / p.pixel_size.z
            sub_psf_model.estzoffset(Nz=Nz)
        elif sub_psf_model.psftype == "vector":
            sub_psf_model.calpupilfield("vector", Nz=Nz)
        else:
            sub_psf_model.calpupilfield("scalar", Nz=Nz)

        psf_model_image, _ = sub_psf_model.genpsfmodel(sigma, Zcoeff_magnitude=Zcoeff_magnitude, Zcoeff_phase=Zcoeff_phase)
        f.res["channel" + str(i)].psf_model_image = psf_model_image


def calstrehlratio(
    param: Union[RunParameters, DictConfig], f, psf_info: PSFInfo, xsz: int = 31
):
    """Compute the Strehl ratio of the fitted PSF.

    For field-dependent (FD) PSFs a Strehl-ratio *map* is produced;
    otherwise a single scalar value is returned.

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
    float or numpy.ndarray
        Strehl ratio (scalar or map).
    """
    f1 = f.copy()
    p = param

    if p.channeltype == "single":
        return _strehl_single(p, f1, f, psf_info, xsz)
    if p.channeltype == "multi":
        return _strehl_multi(p, f1, psf_info, xsz)
    if p.channeltype == "4pi":
        return _strehl_4pi(f)

    return None


def _strehl_single(p, f1, f, psf_info, xsz):
    if "FD" in p.PSFtype:
        f1.res.zernike_map = f.res.zernike_map.copy()
        f1.res.zernike_map[1, 0:4] = 0.0
        f1, _ = genpsf(p, f1, psf_info, Nz=1, xsz=xsz)
        psf_model_image = f1.res.psf_model_image / np.sum(
            f1.res.psf_model_image, axis=(-1, -2), keepdims=True
        )
        I1 = psf_model_image[:, 0, xsz // 2, xsz // 2]

        f1.res.zernike_map = np.zeros(
            f1.res.zernike_map.shape, dtype=np.float32
        )
        f1.res.zernike_map[0, 0] = 1
        f1, _ = genpsf(p, f1, psf_info, Nz=1, xsz=xsz)
        psf_model_image = f1.res.psf_model_image / np.sum(
            f1.res.psf_model_image, axis=(-1, -2), keepdims=True
        )
        I0 = psf_model_image[:, 0, xsz // 2, xsz // 2]

        strehlratio = np.float32(I1 / I0)
        strehlratio_map = np.reshape(
            strehlratio,
            (f.res.zernike_map.shape[-2], f.res.zernike_map.shape[-1]),
        )
        return strehlratio_map

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
    print("Strehl ratio: ", strehlratio)
    return strehlratio


def _strehl_multi(p, f1, psf_info, xsz):
    n_channel = f1.rois.roi_centers.shape[0]
    I1, I0 = [], []

    for i in range(n_channel):
        f1.res["channel" + str(i)].zernike_coefficients[1, :, 0:4] = 0.0

    f1, _ = genpsf(p, f1, psf_info, Nz=1, xsz=xsz)
    coeff = np.zeros(
        f1.res.channel0.zernike_coefficients.shape, dtype=np.float32
    )
    coeff[0, :, 0] = 1

    for i in range(n_channel):
        psf_model_image = f1.res["channel" + str(i)].psf_model_image
        psf_model_image = psf_model_image / np.sum(psf_model_image)
        I1.append(psf_model_image[0, xsz // 2, xsz // 2])
        f1.res["channel" + str(i)].zernike_coefficients = coeff

    f1, _ = genpsf(p, f1, psf_info, Nz=1, xsz=31)
    for i in range(n_channel):
        psf_model_image = f1.res["channel" + str(i)].psf_model_image
        I0.append(psf_model_image[0, xsz // 2, xsz // 2] / np.sum(psf_model_image))

    strehlratio = np.float32(np.stack(I1) / np.stack(I0))
    print("Strehl ratio: ", strehlratio)
    return strehlratio


def _strehl_4pi(f):
    n_channel = f.rois.roi_centers.shape[0]
    mdepth = np.array(
        [
            f.res["channel" + str(i)].modulation_depth
            for i in range(n_channel)
        ]
    )
    print("modulation depth: ", np.round(mdepth, 2))
    return mdepth


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

    if p.channeltype == "single":
        return _fwhm_single(p, f, f1, psf_info)
    if p.channeltype == "multi":
        return _fwhm_multi(p, f)

    return None, None, None


def _fwhm_single(p, f, f1, psf_info):
    if "FD" in p.PSFtype:
        psfsize = f.res.psf_model_image_with_bead.shape
        f1.res.zernike_map = f.res.zernike_map.copy()
        f1.res.zernike_map[1, 0:4] = 0.0
        f1, _ = genpsf(p, f1, psf_info, Nz=psfsize[-3], xsz=psfsize[-1])
        psf_model_image = f1.res.psf_model_image

        fwhmx = np.zeros(psf_model_image.shape[0])
        fwhmy = np.zeros(psf_model_image.shape[0])
        fwhmz = np.zeros(psf_model_image.shape[0])
        for i, psfi in enumerate(psf_model_image):
            Ix, xh, Iy, yh, Iz, zh = getfwhm(psfi)
            fwhmx[i] = np.diff(xh) * p.pixel_size.x * 1e3
            fwhmy[i] = np.diff(yh) * p.pixel_size.y * 1e3
            fwhmz[i] = np.diff(zh) * p.pixel_size.z * 1e3

        shape = f.res.zernike_map.shape[-2:]
        fwhmx_map = np.reshape(fwhmx, shape)
        fwhmy_map = np.reshape(fwhmy, shape)
        fwhmz_map = np.reshape(fwhmz, shape)

        return fwhmx_map, fwhmy_map, fwhmz_map

    psf_model_image = f.res.psf_model_image
    Ix, xh, Iy, yh, Iz, zh = getfwhm(psf_model_image)
    fwhmx = np.diff(xh) * p.pixel_size.x * 1e3
    fwhmy = np.diff(yh) * p.pixel_size.y * 1e3
    fwhmz = np.diff(zh) * p.pixel_size.z * 1e3

    return fwhmx, fwhmy, fwhmz


def _fwhm_multi(p, f):
    n_channel = f.rois.roi_centers.shape[0]
    fwhmx, fwhmy, fwhmz = [], [], []

    for i in range(n_channel):
        psf_model_image = f.res["channel" + str(i)].psf_model_image
        Ix, xh, Iy, yh, Iz, zh = getfwhm(psf_model_image)
        fwhmxi = np.diff(xh) * p.pixel_size.x * 1e3
        fwhmyi = np.diff(yh) * p.pixel_size.y * 1e3
        fwhmzi = np.diff(zh) * p.pixel_size.z * 1e3
        fwhmx.append(fwhmxi)
        fwhmy.append(fwhmyi)
        fwhmz.append(fwhmzi)

    return (
        np.stack(fwhmx),
        np.stack(fwhmy) if fwhmy else None,
        np.stack(fwhmz),
    )


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
