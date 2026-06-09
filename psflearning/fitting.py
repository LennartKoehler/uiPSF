"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu

PSF fitting orchestration: initialising PSF models, running the
L-BFGS-B optimiser, and optional iterative re-learning for SMLM data.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

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
        Initialised PSF model (single- or multi-channel).
    """
    psf_class = psf_info.psf_class
    psf_class_multi = psf_info.psf_class_multi
    is_vector = "vector" in param.PSFtype

    if psf_class_multi is None:
        psf_model = psf_class(options=param.option)
        if is_vector:
            psf_model.psftype = "vector"
    else:
        optimizer_single = L_BFGS_B(maxiter=50, batch_size=param.batch_size)
        psf_model = psf_class_multi(
            psf_class, optimizer_single, options=param.option,
            loss_weight=list(param.loss_weight.values()),
        )
        if is_vector:
            psf_model.PSFtype = "vector"

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
    lossfunmulti = psf_info.loss_fun_multi
    w = list(param.loss_weight.values())

    optimizer = L_BFGS_B(maxiter=param.iteration, batch_size=param.batch_size)

    if lossfunmulti:
        return PSFLearner(
            optimizer, lossfunmulti,
            loss_func_single=lossfun, loss_weight=w,
        )
    else:
        return PSFLearner(optimizer, lossfun, loss_weight=w)


def learn_psf(
    param: Union[RunParameters, DictConfig],
    dataobj,
    psf_info: PSFInfo,
    time: Optional[float] = None,
    do_voxel_refit: bool = True,
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
    do_voxel_refit : bool, optional
        Whether to perform voxel-based refit after initial learning. Default True.

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

    if do_voxel_refit and param.PSFtype == "voxel":
        fit_result, forward_images, toc = _refit_voxel(
            fit_result, centers, dataobj, psf_model, learner,
            forward_images, param.roi.roi_size, time,
        )

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

    filtered_vars = remove_outliers(
        data, fit_result, locres, forward_images, threshold,
    )
    if filtered_vars is not None:
        fit_result, forward_images, toc = learner.learn_psf(
            data, psf, filtered_vars, start_time=toc,
        )
        locres = localize(data.pixelsize_z, fit_result.psf_model_image_with_bead, data.measured_roi_images, param, toc=toc)

    return fit_result, locres, forward_images, toc


def _refit_voxel(
    fit_result, centers, dataobj, psf_model, learner,
    forward_images, roi_size, time,
):
    """If the PSF type is *voxel* and z-drift is large, re-cut ROIs and re-fit."""
    pos = fit_result._variables.positions.numpy()
    zpos = pos[:, 0:1]
    zpos = zpos - np.mean(zpos)

    if centers.shape[-1] != 3 or np.max(np.abs(zpos)) <= 2:
        return fit_result, forward_images, time

    cor = dataobj.roi_centers
    if dataobj.skew_const:
        sk = dataobj.skew_const
        centers1 = np.int32(np.round(np.hstack((
            cor[:, 0:1] - zpos,
            cor[:, 1:2] - sk[0] * zpos,
            cor[:, 2:] - sk[1] * zpos,
        ))))
    else:
        centers1 = np.int32(np.round(np.hstack((
            cor[:, 0:1] - zpos,
            cor[:, 1:2],
            cor[:, 2:],
        ))))

    _, _, _, file_idxs = dataobj.get_image_data()
    dataobj.cut_new_rois(centers1, file_idxs, roi_size=roi_size)
    offset = np.min((np.quantile(dataobj.measured_roi_images, 1e-3), 0))
    dataobj.measured_roi_images = dataobj.measured_roi_images - offset
    if dataobj.skew_const:
        dataobj.deskew_roi(roi_size)

    variables, start_time = psf_model.calc_initials(dataobj, time)
    fit_result, forward_images, toc = learner.learn_psf(dataobj, psf_model, variables, start_time=start_time)
    return fit_result, forward_images, toc


# def localize_FD(
#     param: Union[RunParameters, DictConfig],
#     fit_result,
#     data,
#     psf: PSFInterface,
#     forward_images: np.ndarray,
#     initz=None,
# ):
#     """Localise in the Fourier domain.
#
#     Parameters
#     ----------
#     param : DictConfig
#         Experiment parameters.
#     fit_result : ZernikePSFResult
#         Learning result from learn_psf.
#     data : PreprocessedImageData
#         Data object with ROIs and pixel sizes.
#     psf : PSFInterface
#         Fitted PSF model.
#     forward_images : np.ndarray
#         Forward images from the learning step.
#     initz : optional
#         Initial z offset for localization.
#
#     Returns
#     -------
#     fourier_domain_positions
#         Localisation result in the Fourier domain.
#     """
#     localizer = Localizer()
#     return localizer.localize_FD(
#         fit_result, data.rois, forward_images, data, psf,
#         channeltype=param.channeltype,
#         usecuda=param.usecuda, initz=initz, plot=param.plotall,
#     )


def iterlearn_psf(
    learn_psf_fn: Callable,
    save_result_fn: Callable,
    load_initial_pupil_fn: Callable,
    get_psf_info_fn: Callable,
    param: Union[RunParameters, DictConfig],
    dataobj,
    time: Optional[float] = None,
) -> str:
    """Iterative PSF learning for SMLM / insitu data.

    On each iteration the result of the previous iteration is used to
    initialise the pupil for the next one, and the photon threshold is
    gradually relaxed.

    .. note::
       This function mutates the following fields of *param*:
       ``savename``, ``option.model.init_pupil_file``,
       ``option.insitu.min_photon``, ``option.insitu.stage_pos``.

    Parameters
    ----------
    learn_psf_fn, save_result_fn, load_initial_pupil_fn, get_psf_info_fn : callable
        Injected dependencies.
    param : DictConfig
        Experiment parameters (mutated in-place during iteration).
    dataobj : PreprocessedImageData
        Prepared data object.
    time : float, optional
        Start-time stamp.

    Returns
    -------
    str
        Path to the final result HDF5 file.
    """
    min_photon = param.option.insitu.min_photon
    iterN = param.option.insitu.repeat
    savename = param.savename
    fourier_domain_positions = None
    psf_info = get_psf_info_fn(param)

    for nn in range(iterN):
        if nn > 0:
            dataobj.resetdata()

        psf_model, fit_result, learner, forward_images, toc = learn_psf_fn(
            param, dataobj, psf_info, time=time,
        )

        _, _, _, file_idxs = dataobj.get_image_data()

        if len(file_idxs) > 1:
            fit_result, _, forward_images, toc = relearn(
                dataobj, psf_model, learner, forward_images,
                fit_result, param, toc=toc,
            )

        param.savename = savename + str(nn)
        resfile = save_result_fn(
            param, psf_model, dataobj,
            fit_result, None, fourier_domain_positions,
            forward_images=forward_images,
        )
        param.option.model.init_pupil_file = resfile
        param.option.insitu.min_photon = max(min_photon - nn * 0.1, 0.2)

        _update_stage_pos(psf_model, fit_result, dataobj, param)

    return resfile


def _update_stage_pos(psf_model, fit_result, dataobj, param):
    """Update the stage position parameter from the current fit."""
    channeltype = param.channeltype
    psf_result = psf_model.res2dict(fit_result)

    if channeltype == "single":
        if "stagepos" in psf_result:
            param.option.insitu.stage_pos = float(psf_result["stagepos"])
    else:
        try:
            ch0 = psf_result.get("channel0")
            if ch0 is not None:
                param.option.insitu.stage_pos = float(ch0["stagepos"])
        except (KeyError, TypeError):
            pass


def learn_with_relearn(
    param: Union[RunParameters, DictConfig],
    data,
    psf_info: PSFInfo,
    time: Optional[float] = None,
) -> Tuple[PSFInterface, ZernikePSFResult, LocalizationResult, np.ndarray, Optional[float]]:
    """Learn PSF, localize, remove outliers and re-learn.

    Replicates the original learn_psf pipeline:
    1. Learn the PSF model (with optional voxel refit).
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
