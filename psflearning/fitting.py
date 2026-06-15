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

from .learning import PSFLearner, L_BFGS_B, LocalizationResult, get_intensity_difference_ratio, get_minimum_intensity, get_MSE_difference_ratio, filter_by_mask
from .psf_registry import PSFInfo
from .learning.psfs.PSFZernikeBased import ZernikePSFResult, ZernikePSFVariables
from .learning.psfs.PSFZernikeBase import PSFContext
from .learning.psfs.IPSFModel import IPSFModel
from .learning.fitters.Localizer import localize
from .io.param import RejThresholdParams, RunParameters
from .progress import ProgressReporter


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
    reporter: ProgressReporter,
) -> Tuple[IPSFModel, ZernikePSFResult, PSFLearner, ZernikePSFVariables, np.ndarray, PSFContext]:
    """Run the PSF fitting optimisation.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    dataobj : ImageData
        Prepared data object with extracted ROIs.
    psf_info : PSFInfo
        As returned by :func:`psf_registry.get_psf_info`.
    reporter : ProgressReporter
        Progress reporter.

    Returns
    -------
    tuple of (IPSFModel, ZernikePSFResult, PSFLearner, ZernikePSFVariables, np.ndarray, PSFContext)
        ``(psf_model, fit_result, learner, variables, forward_images, context)``
    """
    psf_model = initialize_psf(param, psf_info)
    initial_pupil = Reader.load_initial_pupil(param)

    learner = create_learner(param, psf_info)

    variables, context = psf_model.calc_initials(dataobj, param, initial_pupil=initial_pupil)
    fit_result, forward_images = learner.learn_psf(dataobj, psf_model, variables, context, reporter=reporter)

    return psf_model, fit_result, learner, variables, forward_images, context




def learn_psf_with_relearn(
    param: Union[RunParameters, DictConfig],
    data,
    psf_info: PSFInfo,
    reporter: ProgressReporter,
) -> Tuple[IPSFModel, ZernikePSFResult, np.ndarray, PSFContext]:
    """Learn PSF, localize, remove outliers and re-learn.

    Replicates the original learn_psf pipeline:
    1. Learn the PSF model.
    3. For multi-file data: remove outliers based on rejection metrics,
       re-learn the PSF.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    dataobj : ImageData
        Prepared data object with extracted ROIs.
    psf_info : PSFInfo
        As returned by :func:`psf_registry.get_psf_info`.
    reporter : ProgressReporter
        Progress reporter.

    Returns
    -------
    tuple of (IPSFModel, ZernikePSFResult, np.ndarray, PSFContext)
        ``(psf_model, fit_result, forward_images, context)``
    """
    psf, fit_result, learner, variables, forward_images, context = learn_psf(param, data, psf_info, reporter=reporter)

    file_idxs = data.source_file_indices

    if len(file_idxs) > 1:
        # remove outliers
        threshold = param.model.rej_threshold

        mseRatio = get_MSE_difference_ratio(forward_images, data.measured_roi_images)
        mask = mseRatio > threshold.mse

        intensityRatio = get_intensity_difference_ratio(fit_result.intensities)
        mask = (intensityRatio > threshold.photon) & mask

        minI = get_minimum_intensity(fit_result.intensities)
        mask = (minI > 0) & mask

        result = filter_by_mask(
            data, variables, mask
        )
        if result is not None:
            data, filtered_vars = result
            fit_result, forward_images = learner.learn_psf(
                data, psf, filtered_vars, context, reporter=reporter,
            )


    return psf, fit_result, forward_images, context


def learn_psf_with_relearn_with_localization(
    param: Union[RunParameters, DictConfig],
    data,
    psf_info: PSFInfo,
    reporter: ProgressReporter,
) -> Tuple[IPSFModel, ZernikePSFResult, LocalizationResult, np.ndarray, PSFContext]:
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
    dataobj : ImageData
        Prepared data object with extracted ROIs.
    psf_info : PSFInfo
        As returned by :func:`psf_registry.get_psf_info`.
    reporter : ProgressReporter
        Progress reporter.

    Returns
    -------
    tuple of (IPSFModel, ZernikePSFResult, LocalizationResult, np.ndarray, PSFContext)
        ``(psf_model, fit_result, locres, forward_images, context)``
    """
    psf, fit_result, learner, variables, forward_images, context = learn_psf(param, data, psf_info, reporter=reporter)

    file_idxs = data.source_file_indices

    # relearn
    if len(file_idxs) > 1:
        threshold = param.model.rej_threshold

        locres = localize(data.pixelsize_z, fit_result.psf_model_image_with_bead, data.measured_roi_images, param, reporter=reporter)

        # remove outliers
        mseRatio = get_MSE_difference_ratio(forward_images, data.measured_roi_images)
        mask = mseRatio > threshold.mse

        intensityRatio = get_intensity_difference_ratio(fit_result.intensities)
        mask = (intensityRatio > threshold.photon) & mask

        minI = get_minimum_intensity(fit_result.intensities)
        mask = (minI > 0) & mask

        mask = (locres.mse_z_ratio > threshold.bias_z) & mask


        result = filter_by_mask(
            data, variables, mask
        )

        if result is not None:
            data, filtered_vars = result
            fit_result, forward_images = learner.learn_psf(
                data, psf, filtered_vars, context, reporter=reporter,
            )
            locres = localize(data.pixelsize_z, fit_result.psf_model_image_with_bead, data.measured_roi_images, param, reporter=reporter)


    else:
        locres = localize(data.pixelsize_z, fit_result.psf_model_image_with_bead, data.measured_roi_images, param, reporter=reporter)

    return psf, fit_result, locres, forward_images, context
