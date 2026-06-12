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
from .psf_registry import PSFInfo
from .learning.psfs.PSFZernikeBased import ZernikePSFResult, ZernikePSFVariables
from .learning.psfs.PSFZernikeBase import PSFContext
from .learning.psfs.IPSFModel import IPSFModel
from .learning.fitters.Localizer import localize
from .io.param import RunParameters


def initialize_psf(param: RunParameters, psf_info: PSFInfo):
    """Create and return a PSF model object from *psf_info*.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    psf_info : PSFInfo
        As returned by :func:`psf_registry.get_psf_info`.

    Returns
    -------
    IPSFModel
        Initialised PSF model (stateless).
    """
    psf_class = psf_info.psf_class
    psf_model = psf_class()
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
    w = list(param.model.loss_weight.values())
    optimizer = L_BFGS_B(maxiter=param.runtime.max_iterations, batch_size=param.runtime.batch_size)
    return PSFLearner(optimizer, lossfun, loss_weight=w)


def learn_psf(
    param: Union[RunParameters, DictConfig],
    dataobj,
    psf_info: PSFInfo,
    time: Optional[float] = None,
) -> Tuple[IPSFModel, ZernikePSFResult, PSFLearner, ZernikePSFVariables, np.ndarray, Optional[float], PSFContext]:
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
    tuple of (IPSFModel, ZernikePSFResult, PSFLearner, ZernikePSFVariables, np.ndarray, float, PSFContext)
        ``(psf_model, fit_result, learner, variables, forward_images, toc, context)``
    """
    psf_model = initialize_psf(param, psf_info)
    initial_pupil = Reader.load_initial_pupil(param)

    learner = create_learner(param, psf_info)

    variables, context, time = psf_model.calc_initials(dataobj, param, initial_pupil=initial_pupil, start_time=time)
    fit_result, forward_images, toc = learner.learn_psf(dataobj, psf_model, variables, context, start_time=time)

    return psf_model, fit_result, learner, variables, forward_images, toc, context


def relearn(
    data,
    psf: IPSFModel,
    learner: PSFLearner,
    variables: ZernikePSFVariables,
    forward_images: np.ndarray,
    fit_result: ZernikePSFResult,
    param: Union[RunParameters, DictConfig],
    context: PSFContext,
    toc: Optional[float] = None,
    threshold: Optional[list] = None,
) -> Tuple[ZernikePSFResult, LocalizationResult, np.ndarray, Optional[float]]:
    """Remove outliers, re-learn PSF, then re-localize.

    Parameters
    ----------
    data : PreprocessedImageData
        Data object with ROIs.
    psf : IPSFModel
        Fitted PSF model.
    learner : PSFLearner
        The learner instance (holds optimizer and loss function).
    variables : ZernikePSFVariables
        Optimized variables from the previous learning step.
    forward_images : np.ndarray
        Forward images from the previous learning step.
    fit_result : ZernikePSFResult
        Current learning result.
    param : DictConfig
        Experiment parameters.
    context : PSFContext
        PSF context carrying all operational state.
    toc : float, optional
        Current end time.
    threshold : list, optional
        Custom rejection thresholds. If None, uses param.model.rej_threshold.

    Returns
    -------
    tuple of (ZernikePSFResult, LocalizationResult, np.ndarray, float)
        ``(fit_result, locres, forward_images, toc)``
    """
    if threshold is None:
        threshold = list(param.model.rej_threshold.values())

    locres = localize(data.pixelsize_z, fit_result.psf_model_image_with_bead, data.measured_roi_images, param, toc=toc)

    filtered_vars = remove_outliers(
        data, variables, fit_result, locres, forward_images, threshold,
    )
    if filtered_vars is not None:
        fit_result, forward_images, toc = learner.learn_psf(
            data, psf, filtered_vars, context, start_time=toc,
        )
        locres = localize(data.pixelsize_z, fit_result.psf_model_image_with_bead, data.measured_roi_images, param, toc=toc)

    return fit_result, locres, forward_images, toc


def learn_psf_with_relearn(
    param: Union[RunParameters, DictConfig],
    data,
    psf_info: PSFInfo,
    time: Optional[float] = None,
) -> Tuple[IPSFModel, ZernikePSFResult, LocalizationResult, np.ndarray, Optional[float], PSFContext]:
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
    tuple of (IPSFModel, ZernikePSFResult, LocalizationResult, np.ndarray, float, PSFContext)
        ``(psf_model, fit_result, locres, forward_images, toc, context)``
    """
    psf, fit_result, learner, variables, forward_images, toc, context = learn_psf(param, data, psf_info, time=time)

    _, _, _, file_idxs = data.get_image_data()

    if len(file_idxs) > 1:
        fit_result, locres, forward_images, toc = relearn(
            data, psf, learner, variables, forward_images,
            fit_result, param, context, toc=toc,
        )
    else:
        locres = localize(data.pixelsize_z, fit_result.psf_model_image_with_bead, data.measured_roi_images, param, toc=toc)

    return psf, fit_result, locres, forward_images, toc, context
