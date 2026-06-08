from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..psfs.PSFInterface import PSFInterface
from ..loclib import localizationlib, LocalizationResult
from ..psf_variables import Positions
from typing import Union
from ...io.param import RunParameters
from omegaconf import DictConfig
from ..psfs.PSFZernikeBased import ZernikePSFResult
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class LocalizationOutput:
    """Result from Localizer.localize(), bundling the localization result
    with the rejection metrics computed during localization."""
    locres: Any
    reject_metric: Optional[List[np.ndarray]] = None
    minI: Optional[np.ndarray] = None


"""
Handles localization given a learned PSF model.

Stateless with respect to data and PSF. All data, PSF model, ROIs,
and forward images are passed explicitly to each method call.
"""

def localize(
    pixelsize_z: np.ndarray,
    psf_model_image: np.ndarray,
    rois: np.ndarray,
    param: Union[RunParameters, DictConfig],
    toc: Optional[float] = None,
) -> LocalizationResult:
    """Perform localization using the learned PSF.

    Pure localization -- no relearning.  For non-insitu PSF types the
    localizer also computes rejection metrics accessible via
    ``localizer.reject_metric`` and ``localizer.minI``.

    Parameters
    ----------
    learner : PSFLearner
        The learner instance with fitted PSF.
    res : list
        Learning result (optimized variables).
    param : DictConfig
        Experiment parameters.
    toc : float, optional
        End time from learning.

    Returns
    -------
    tuple of (Localizer, list, float)
        ``(localizer, locres, toc)``
    """
    dll = localizationlib(usecuda=param.usecuda)
    locres = dll.loc_ast(rois, psf_model_image, pixelsize_z, initz=None, start_time=toc)

    return locres

def localize_old(
    res: Any,
    rois: np.ndarray,
    forward_images: np.ndarray,
    data: Any,
    psf: PSFInterface,
    channeltype: str,
    usecuda: bool = True,
    initz: Optional[Any] = None,
    plot: bool = True,
    start_time: Optional[float] = None,
) -> LocalizationOutput:
    """Localize emitters using the fitted PSF model and compute rejection metrics.

    Parameters
    ----------
    res : ZernikePSFResult
        Fitted PSF result with intensities, psf_model_image_with_bead, drift_xy, etc.
    rois : np.ndarray
        ROI image data.
    forward_images : np.ndarray
        Forward images from PSF learning.
    data : PreprocessedImageData
        Data object with pixel sizes and centers.
    psf : PSFInterface
        Fitted PSF model (used for imgcenter, sub_psfs in multi/4pi).
    channeltype : str
        One of 'single', 'multi', '4pi'.
    usecuda : bool
        Whether to use CUDA for MLE fitting.
    initz : optional
        Initial z offset for localization.
    plot : bool
        Whether to show diagnostic plots.
    start_time : float, optional
        Start-time stamp for progress reporting.

    Returns
    -------
    LocalizationOutput
        Contains locres, reject_metric, and minI.
    """
    intensity = np.abs(np.squeeze(res.intensities, axis=(-1, -2)))
    if res.intensities.dtype == 'complex64':
        intensityR = intensity
    else:
        intensityR = np.real(np.squeeze(res.intensities, axis=(-1, -2)))
    psf_model_image = res.psf_model_image_with_bead
    modeled_forward_images = forward_images
    measured_roi_images = rois
    pz = data.pixelsize_z

    dll = localizationlib(usecuda=usecuda)
    if channeltype == 'single':
        locres = dll.loc_ast(measured_roi_images, psf_model_image, pz, initz=initz, start_time=start_time)
        mydiff = modeled_forward_images[:, 1:-1] - measured_roi_images[:, 1:-1]
        mse1 = np.mean(np.square(mydiff), axis=(-3, -2, -1)) / np.mean(measured_roi_images, axis=(-3, -2, -1))

    elif channeltype == 'multi':
        _, _, centers, _ = data.get_image_data()
        cor = np.stack(centers)[..., -2:]  # pyright: ignore[reportCallIssue]
        imgcenter = psf.imgcenter
        T = res.drift_xy
        locres = dll.loc_ast_dual(measured_roi_images, psf_model_image, pz, cor, imgcenter, T, initz=initz, plot=plot, start_time=start_time)
        mydiff = modeled_forward_images[:, :, 1:-1] - measured_roi_images[:, :, 1:-1]
        mse1 = np.mean(np.mean(np.square(mydiff), axis=(-3, -2, -1)) / np.mean(measured_roi_images, axis=(-3, -2, -1)), axis=0)

    elif channeltype == '4pi':
        _, _, centers, _ = data.get_image_data()
        interference_amplitude = res.psf_model_image
        cor = np.stack(centers)
        imgcenter = psf.imgcenter
        T = np.squeeze(res.drift_xy)
        zT = np.array([psf.sub_psfs[0].zT])
        locres = dll.loc_4pi(measured_roi_images, psf_model_image, interference_amplitude, pz, cor, imgcenter, T, zT, initz=initz, plot=plot, start_time=start_time)
        mydiff = modeled_forward_images[:, :, :, 1:-1] - measured_roi_images[:, :, :, 1:-1]
        mse1 = np.mean(np.mean(np.square(mydiff), axis=(-4, -3, -2, -1)) / np.mean(measured_roi_images, axis=(-4, -3, -2, -1)), axis=0)
    else:
        raise TypeError('supported channeltype is:', str(['single', 'multi', '4pi']))

    if channeltype == 'single':
        if len(intensity.shape) < 2:
            avgI = intensity
            minI = intensityR
        else:
            avgI = np.median(intensity, axis=1)
            minI = np.min(intensityR, axis=1)
    else:
        if len(intensity.shape) < 3:
            avgI = intensity[0]
            minI = intensityR[0]
        else:
            avgI = np.median(intensity[0], axis=1)
            minI = np.min(intensityR[0], axis=1)

    if measured_roi_images.shape[0] == 1:
        intRatio = np.array([1.0])
        mseRatio = np.array([1.0])
    else:
        intRatio = np.square(avgI - np.median(avgI)) / np.median(avgI) / avgI
        mseRatio = mse1 / np.median(mse1)
    msezRatio = locres.mse_z_ratio
    metric = [msezRatio, mseRatio, intRatio]
    label = ['relative MSE in z', 'relative MSE']
    if plot & (mseRatio.size > 1):
        fig = plt.figure(figsize=[8, 8])
        for i, val in enumerate(metric[:-1]):
            plt.plot(intRatio, val, '.')
        plt.xlabel('relative intensity')
        plt.ylabel('relative MSE')
        plt.xlim([0, 3])
        plt.ylim([0, 5])
        plt.legend(label)
        plt.grid(True)
        plt.show()

    return LocalizationOutput(locres=locres, reject_metric=metric, minI=minI)

def localize_smlm(
    res: Any,
    rois: np.ndarray,
    data: Any,
    psf: PSFInterface,
    channeltype: str,
    usecuda: bool = True,
    initz: Optional[Any] = None,
    plot: bool = True,
) -> Any:
    """Localize emitters in SMLM data using the fitted PSF model.

    Parameters
    ----------
    res : ZernikePSFResult
        Fitted PSF result.
    rois : np.ndarray
        ROI image data.
    data : PreprocessedImageData
        Data object with pixel sizes and centers.
    psf : PSFInterface
        Fitted PSF model.
    channeltype : str
        One of 'single', 'multi', '4pi'.
    usecuda : bool
        Whether to use CUDA for MLE fitting.
    initz : optional
        Initial z offset for localization.
    plot : bool
        Whether to show diagnostic plots.

    Returns
    -------
    LocalizationResult
    """
    psf_model_image = res.psf_model_image_with_bead

    measured_roi_images = rois
    pz = data.pixelsize_z

    dll = localizationlib(usecuda=usecuda)
    if channeltype == 'single':
        locres = dll.loc_ast(measured_roi_images, psf_model_image, pz, initz=initz)

    elif channeltype == 'multi':
        _, _, centers, _ = data.get_image_data()
        cor = np.stack(centers)
        imgcenter = psf.imgcenter
        T = res.drift_xy
        locres = dll.loc_ast_dual(measured_roi_images, psf_model_image, pz, cor, imgcenter, T, initz=initz, plot=plot)

    elif channeltype == '4pi':
        _, _, centers, _ = data.get_image_data()
        interference_amplitude = res.psf_model_image
        cor = np.stack(centers)
        imgcenter = psf.imgcenter
        T = np.squeeze(res.drift_xy)
        zT = np.array([data.channels[0].zT])
        locres = dll.loc_4pi(measured_roi_images, psf_model_image, interference_amplitude, pz, cor, imgcenter, T, zT, initz=initz, plot=plot)

    else:
        raise TypeError('supported psftype is:', str(['single', 'multi', '4pi']))

    return locres

def localize_FD(
    res: Any,
    rois: np.ndarray,
    forward_images: np.ndarray,
    data: Any,
    psf: PSFInterface,
    channeltype: str,
    usecuda: bool = True,
    initz: Optional[Any] = None,
    plot: bool = True,
) -> Positions:
    """Localize emitters frame-by-frame and compute z-bias drift over the acquisition.

    Parameters
    ----------
    res : ZernikePSFResult
        Fitted PSF result with drift_xy, etc.
    rois : np.ndarray
        ROI image data.
    forward_images : np.ndarray
        Forward images from PSF learning.
    data : PreprocessedImageData
        Data object with pixel sizes and centers.
    psf : PSFInterface
        Fitted PSF model.
    channeltype : str
        One of 'single', 'multi', '4pi'.
    usecuda : bool
        Whether to use CUDA for MLE fitting.
    initz : optional
        Initial z offset for localization.
    plot : bool
        Whether to show diagnostic plots.

    Returns
    -------
    Positions
    """
    modeled_forward_images = forward_images
    measured_roi_images = rois
    pz = data.pixelsize_z
    if len(measured_roi_images.shape) > 3:
        Nz = measured_roi_images.shape[-3]
    else:
        Nz = 1
    _, _, centers, _ = data.get_image_data()
    cor = np.stack(centers)
    dll = localizationlib(usecuda=usecuda)
    x = []
    y = []
    z = []
    for i in range(measured_roi_images.shape[-4]):
        if channeltype == 'single':
            loci = dll.loc_ast(measured_roi_images[i], modeled_forward_images[i], pz, initz=initz, start_time=0)
        elif channeltype == 'multi':

            imgcenter = psf.imgcenter
            T = res.drift_xy
            loci = dll.loc_ast_dual(measured_roi_images[:, i:i + 1], modeled_forward_images[:, i], pz, cor[:, i:i + 1], imgcenter, T, initz=initz, start_time=0)

        x.append(np.squeeze(loci.positions.x))
        y.append(np.squeeze(loci.positions.y))
        z.append(np.squeeze(loci.positions.z))

    xf = np.stack(x)
    yf = np.stack(y)
    zf = np.stack(z)

    zg = np.linspace(0, Nz - 1, Nz)
    if Nz > 1:
        zf = zf - np.median(zf - zg, axis=1, keepdims=True)
        zdiff = zf - zg
        xf = xf - np.median(xf, axis=1, keepdims=True)
        yf = yf - np.median(yf, axis=1, keepdims=True)
        if Nz > 4:
            zind = range(2, Nz - 2, 1)
        else:
            zind = range(0, Nz, 1)

        zdiff = zdiff - np.mean(zdiff[:, zind], axis=1, keepdims=True)
    else:
        zdiff = zf
    if plot & (Nz > 1):
        fig = plt.figure(figsize=[12, 6])
        ax = fig.add_subplot(1, 2, 1)
        plt.plot(zf.transpose(), color=(0.6, 0.6, 0.6))
        plt.plot(zg)
        ax = fig.add_subplot(1, 2, 2)
        plt.plot((zdiff).transpose(), 'k', alpha=0.1)
        plt.plot(np.median(zdiff, axis=0), color='r')
        plt.plot(zg - zg, color='k')
        ax.set_ylabel('z bias')
        ax.set_ylim([-0.1, 0.1] / np.array(pz))

    loc_FD = Positions(x=xf, y=yf, z=zf)
    return loc_FD
