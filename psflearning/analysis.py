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


def genpsf(
    param: DictConfig,
    f,
    psf_info: dict,
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
    psf_info : dict
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
        image_size=list(f.rois.image_size),
        rois=np.zeros((Nz, xsz, xsz)),
    )

    psfobj = initialize_psf(param, psf_info)

    if p.channeltype == "single":
        _genpsf_single(psfobj, dataobj, f, p, Nz, stagepos)
    elif p.channeltype == "multi":
        _genpsf_multi(psfobj, dataobj, f, p, Nz, stagepos)

    return f, psfobj


def _genpsf_single(psfobj, dataobj, f, p, Nz, stagepos):
    sigma = f.res.sigma
    Zcoeff = f.res.zernike_coeff.reshape(
        f.res.zernike_coeff.shape + (1, 1)
    )
    psfobj.data = dataobj

    if "insitu" in p.PSFtype:
        psfobj.stagepos = stagepos / p.pixel_size.z
        psfobj.estzoffset(Nz=Nz)
    elif psfobj.psftype == "vector":
        psfobj.calpupilfield("vector", Nz=Nz)
    else:
        psfobj.calpupilfield("scalar", Nz=Nz)

    if "FD" in p.PSFtype:
        f.res.I_model = _genpsf_fd(psfobj, f, sigma)
    else:
        f.res.I_model, _ = psfobj.genpsfmodel(sigma, Zcoeff)


def _genpsf_fd(psfobj, f, sigma):
    """Generate a field-dependent PSF model (batched over positions)."""
    img_size = f.rois.image_size
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
    I_model = None
    for i in range(len(ind) - 1):
        I0, _, _ = psfobj.genpsfmodel(
            sigma, Zmap=zmap, cor=cor[ind[i] : ind[i + 1]]
        )
        I_model = I0 if I_model is None else np.vstack((I_model, I0))
    return I_model


def _genpsf_multi(psfobj, dataobj, f, p, Nz, stagepos):
    n_channel = f.rois.cor.shape[0]
    psfobj.sub_psfs = [None] * n_channel

    for i in range(n_channel):
        psf = psfobj.psftype(options=psfobj.options)
        psf.psftype = psfobj.PSFtype
        psfobj.sub_psfs[i] = psf

        sigma = f.res["channel" + str(i)].sigma
        Zcoeff = f.res["channel" + str(i)].zernike_coeff
        Zcoeff = Zcoeff.reshape(Zcoeff.shape + (1, 1))
        psf.data = dataobj

        if "insitu" in p.PSFtype:
            psf.stagepos = stagepos / p.pixel_size.z
            psf.estzoffset(Nz=Nz)
        elif psf.psftype == "vector":
            psf.calpupilfield("vector", Nz=Nz)
        else:
            psf.calpupilfield("scalar", Nz=Nz)

        I_model, _ = psf.genpsfmodel(sigma, Zcoeff)
        f.res["channel" + str(i)].I_model = I_model


def calstrehlratio(
    param: DictConfig, f, psf_info: dict, xsz: int = 31
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
    psf_info : dict
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
        I_model = f1.res.I_model / np.sum(
            f1.res.I_model, axis=(-1, -2), keepdims=True
        )
        I1 = I_model[:, 0, xsz // 2, xsz // 2]

        f1.res.zernike_map = np.zeros(
            f1.res.zernike_map.shape, dtype=np.float32
        )
        f1.res.zernike_map[0, 0] = 1
        f1, _ = genpsf(p, f1, psf_info, Nz=1, xsz=xsz)
        I_model = f1.res.I_model / np.sum(
            f1.res.I_model, axis=(-1, -2), keepdims=True
        )
        I0 = I_model[:, 0, xsz // 2, xsz // 2]

        strehlratio = np.float32(I1 / I0)
        strehlratio_map = np.reshape(
            strehlratio,
            (f.res.zernike_map.shape[-2], f.res.zernike_map.shape[-1]),
        )
        return strehlratio_map

    f1.res.zernike_coeff[1, 0:4] = 0.0
    f1, _ = genpsf(p, f1, psf_info, Nz=1, xsz=xsz)
    I1 = f1.res.I_model[0, xsz // 2, xsz // 2] / np.sum(
        f1.res.I_model
    )

    f1.res.zernike_coeff = np.zeros(
        f1.res.zernike_coeff.shape, dtype=np.float32
    )
    f1.res.zernike_coeff[0, 0] = 1
    f1, _ = genpsf(p, f1, psf_info, Nz=1, xsz=xsz)
    I0 = f1.res.I_model[0, xsz // 2, xsz // 2] / np.sum(
        f1.res.I_model
    )

    strehlratio = np.float32(I1 / I0)
    print("Strehl ratio: ", strehlratio)
    return strehlratio


def _strehl_multi(p, f1, psf_info, xsz):
    n_channel = f1.rois.cor.shape[0]
    I1, I0 = [], []

    for i in range(n_channel):
        f1.res["channel" + str(i)].zernike_coeff[1, 0:4] = 0.0

    f1, _ = genpsf(p, f1, psf_info, Nz=1, xsz=xsz)
    coeff = np.zeros(
        f1.res.channel0.zernike_coeff.shape, dtype=np.float32
    )
    coeff[0, 0] = 1

    for i in range(n_channel):
        I_model = f1.res["channel" + str(i)].I_model
        I_model = I_model / np.sum(I_model)
        I1.append(I_model[0, xsz // 2, xsz // 2])
        f1.res["channel" + str(i)].zernike_coeff = coeff

    f1, _ = genpsf(p, f1, psf_info, Nz=1, xsz=31)
    for i in range(n_channel):
        I_model = f1.res["channel" + str(i)].I_model
        I0.append(I_model[0, xsz // 2, xsz // 2] / np.sum(I_model))

    strehlratio = np.float32(np.stack(I1) / np.stack(I0))
    print("Strehl ratio: ", strehlratio)
    return strehlratio


def _strehl_4pi(f):
    n_channel = f.rois.cor.shape[0]
    mdepth = np.array(
        [
            f.res["channel" + str(i)].modulation_depth
            for i in range(n_channel)
        ]
    )
    print("modulation depth: ", np.round(mdepth, 2))
    return mdepth


def calfwhm(
    param: DictConfig, f, psf_info: dict
) -> Tuple:
    """Compute the full-width at half-maximum of the fitted PSF.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    f : DottedDict
        Fitting result.
    psf_info : dict
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
        psfsize = f.res.I_model_bead.shape
        f1.res.zernike_map = f.res.zernike_map.copy()
        f1.res.zernike_map[1, 0:4] = 0.0
        f1, _ = genpsf(p, f1, psf_info, Nz=psfsize[-3], xsz=psfsize[-1])
        I_model = f1.res.I_model

        fwhmx = np.zeros(I_model.shape[0])
        fwhmy = np.zeros(I_model.shape[0])
        fwhmz = np.zeros(I_model.shape[0])
        for i, psfi in enumerate(I_model):
            Ix, xh, Iy, yh, Iz, zh = getfwhm(psfi)
            fwhmx[i] = np.diff(xh) * p.pixel_size.x * 1e3
            fwhmy[i] = np.diff(yh) * p.pixel_size.y * 1e3
            fwhmz[i] = np.diff(zh) * p.pixel_size.z * 1e3

        shape = f.res.zernike_map.shape[-2:]
        fwhmx_map = np.reshape(fwhmx, shape)
        fwhmy_map = np.reshape(fwhmy, shape)
        fwhmz_map = np.reshape(fwhmz, shape)

        return fwhmx_map, fwhmy_map, fwhmz_map

    I_model = f.res.I_model
    Ix, xh, Iy, yh, Iz, zh = getfwhm(I_model)
    fwhmx = np.diff(xh) * p.pixel_size.x * 1e3
    fwhmy = np.diff(yh) * p.pixel_size.y * 1e3
    fwhmz = np.diff(zh) * p.pixel_size.z * 1e3

    return fwhmx, fwhmy, fwhmz


def _fwhm_multi(p, f):
    n_channel = f.rois.cor.shape[0]
    fwhmx, fwhmy, fwhmz = [], [], []

    for i in range(n_channel):
        I_model = f.res["channel" + str(i)].I_model
        Ix, xh, Iy, yh, Iz, zh = getfwhm(I_model)
        fwhmxi = np.diff(xh) * p.pixel_size.x * 1e3
        fwhmyi = np.diff(yh) * p.pixel_size.y * 1e3
        fwhmzi = np.diff(zh) * p.pixel_size.z * 1e3
        fwhmx.append(fwhmxi)
        fwhmz.append(fwhmzi)

    return (
        np.stack(fwhmx),
        np.stack(fwhmy) if fwhmy else None,
        np.stack(fwhmz),
    )


def getfwhm(I_model: np.ndarray) -> Tuple:
    """Compute the FWHM along x, y, and z of a 3-D PSF volume.

    Returns
    -------
    tuple
        ``(Ix, xh, Iy, yh, Iz, zh)`` where each ``*h`` is a 2-element
        array with the half-maximum crossing positions.
    """
    cor = np.unravel_index(np.argmax(I_model), I_model.shape)

    Ix = I_model[cor[0], cor[1]]
    xh = get1dfwhm(Ix, cor[2])
    Iy = I_model[cor[0], :, cor[2]]
    yh = get1dfwhm(Iy, cor[1])

    Iz = I_model[:, cor[1], cor[2]]
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
