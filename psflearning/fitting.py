"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu

PSF fitting orchestration: initialising PSF models, running the
L-BFGS-B optimiser for single-channel Zernike PSF fitting.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .reader import Reader
from omegaconf import DictConfig

from .learning import PSFLearner, L_BFGS_B, LocalizationResult, remove_outliers
from .learning.psf_variables import PSFInfo
from .learning.psfs.PSFZernikeBased import ZernikePSFResult
from .learning.psfs.PSFInterface import PSFInterface
from .learning.fitters.Localizer import localize
from .io.param import RunParameters


def initialize_psf(param: Union[RunParameters, DictConfig], psf_info: PSFInfo):
    """Create and return a PSF model object from *psf_info*.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    psf_info : PSFInfo
        As returned by :func:`psf_registry.get_psf_info`.

    Returns
    -------
    PSFInterface
        Initialised PSF model.
    """
    psf_class = psf_info.psf_class
    psf_model = psf_class(options=param.option)
    return psf_model


def create_learner(
    param: Union[RunParameters, DictConfig],
    psf_info: PSFInfo,
) -> PSFLearner:
    """Create a PSFLearner instance with configured optimizer and loss functions.

    The learner holds only learning-related state (optimizer, loss function,
    penalty parameters). Data and PSF model are passed to method calls.
    """
    lossfun = psf_info.loss_fun
    w = list(param.loss_weight.values())
    optimizer = L_BFGS_B(maxiter=param.iteration, batch_size=param.batch_size)
    return PSFLearner(optimizer, lossfun, loss_weight=w)


def learn_psf(
    param: Union[RunParameters, DictConfig],
    dataobj,
    psf_info: PSFInfo,
    time: Optional[float] = None,
) -> Tuple[PSFInterface, ZernikePSFResult, PSFLearner, np.ndarray, Optional[float]]:
    """Run the PSF fitting optimisation.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    dataobj : PreprocessedImageData
        Prepared data object with extracted ROIs.
    psf_info : PSFInfo
        As returned by :func:`psf_registry.get_psf_info`.
    time : float, optional
        Start-time stamp for progress reporting.

    Returns
    -------
    tuple of (PSFInterface, ZernikePSFResult, PSFLearner, np.ndarray, float)
        ``(psf_model, fit_result, learner, forward_images, toc)``
    """
    psf_model = initialize_psf(param, psf_info)
    Reader.load_initial_pupil(param, psf_model, dataobj)

    learner = create_learner(param, psf_info)

    _, _, centers, _ = dataobj.get_image_data()

    variables, time = psf_model.calc_initials(dataobj, time)
    fit_result, forward_images, toc = learner.learn_psf(dataobj, psf_model, variables, start_time=time)

    return psf_model, fit_result, learner, forward_images, toc


def relearn(
    data,
    psf: PSFInterface,
    learner: PSFLearner,
    forward_images: np.ndarray,
    fit_result: ZernikePSFResult,
    param: Union[RunParameters, DictConfig],
    toc: Optional[float] = None,
    threshold: Optional[list] = None,
) -> Tuple[ZernikePSFResult, LocalizationResult, np.ndarray, Optional[float]]:
    """Remove outliers, re-learn PSF, then re-localize.

    Parameters
    ----------
    data : PreprocessedImageData
        Data object with ROIs.
    psf : PSFInterface
        Fitted PSF model.
    learner : PSFLearner
        The learner instance (holds optimizer and loss function).
    forward_images : np.ndarray
        Forward images from the previous learning step.
    fit_result : ZernikePSFResult
        Current learning result.
    param : DictConfig
        Experiment parameters.
    toc : float, optional
        Current end time.
    threshold : list, optional
        Custom rejection thresholds. If None, uses param.rej_threshold.

    Returns
    -------
    tuple of (ZernikePSFResult, LocalizationResult, np.ndarray, float)
        ``(fit_result, locres, forward_images, toc)``
    """
    if threshold is None:
        threshold = list(param.rej_threshold.values())

    locres = localize(data.pixelsize_z, fit_result.psf_model_image_with_bead, data.measured_roi_images, param, toc=toc)

    filtered_vars = psf.remove_outliers(
        data, fit_result, locres, forward_images, threshold,
    )
    if filtered_vars is not None:
        fit_result, forward_images, toc = learner.learn_psf(
            data, psf, filtered_vars, start_time=toc,
        )
        locres = localize(data.pixelsize_z, fit_result.psf_model_image_with_bead, data.measured_roi_images, param, toc=toc)

    return fit_result, locres, forward_images, toc


def learn_psf_with_relearn(
    param: Union[RunParameters, DictConfig],
    data,
    psf_info: PSFInfo,
    time: Optional[float] = None,
) -> Tuple[PSFInterface, ZernikePSFResult, LocalizationResult, np.ndarray, Optional[float]]:
    """Learn PSF, localize, remove outliers and re-learn.

    Replicates the original learn_psf pipeline:
    1. Learn the PSF model.
    2. Localize emitters.
    3. For multi-file data: remove outliers based on rejection metrics,
       re-learn the PSF, and re-localize.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    dataobj : PreprocessedImageData
        Prepared data object with extracted ROIs.
    psf_info : PSFInfo
        As returned by :func:`psf_registry.get_psf_info`.
    time : float, optional
        Start-time stamp for progress reporting.

    Returns
    -------
    tuple of (PSFInterface, ZernikePSFResult, LocalizationResult, np.ndarray, float)
        ``(psf_model, fit_result, locres, forward_images, toc)``
    """
    psf, fit_result, learner, forward_images, toc = learn_psf(param, data, psf_info, time=time)

    _, _, _, file_idxs = data.get_image_data()

    if len(file_idxs) > 1:
        fit_result, locres, forward_images, toc = relearn(
            data, psf, learner, forward_images,
            fit_result, param, toc=toc,
        )
    else:
        locres = localize(data.pixelsize_z, fit_result.psf_model_image_with_bead, data.measured_roi_images, param, toc=toc)

    return psf, fit_result, locres, forward_images, toc
