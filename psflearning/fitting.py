"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu

PSF fitting orchestration: initialising PSF models, running the
L-BFGS-B optimiser, and optional iterative re-learning for SMLM data.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from .reader import Reader
from omegaconf import DictConfig

from .learning import PSFLearner, Localizer, L_BFGS_B


def initialize_psf(param: DictConfig, psf_info: dict):
    """Create and return a PSF model object from *psf_info*.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    psf_info : dict
        As returned by :func:`psf_registry.get_psf_info`.

    Returns
    -------
    PSFInterface
        Initialised PSF model (single- or multi-channel).
    """
    psf_class = psf_info["psf_class"]
    psf_class_multi = psf_info["psf_class_multi"]
    is_vector = "vector" in param.PSFtype

    if psf_class_multi is None:
        psfobj = psf_class(options=param.option)
        if is_vector:
            psfobj.psftype = "vector"
    else:
        optimizer_single = L_BFGS_B(maxiter=50)
        optimizer_single.batch_size = param.batch_size
        psfobj = psf_class_multi(
            psf_class, optimizer_single, options=param.option,
            loss_weight=list(param.loss_weight.values()),
        )
        if is_vector:
            psfobj.PSFtype = "vector"

    return psfobj


def create_learner(
    param: DictConfig,
    dataobj,
    psfobj,
    psf_info: dict,
) -> PSFLearner:
    """Create a PSFLearner instance with configured optimizer and loss functions."""
    lossfun = psf_info["loss_fun"]
    lossfunmulti = psf_info["loss_fun_multi"]
    w = list(param.loss_weight.values())

    optimizer = L_BFGS_B(maxiter=param.iteration)
    optimizer.batch_size = param.batch_size

    if lossfunmulti:
        return PSFLearner(
            dataobj, psfobj, optimizer, lossfunmulti,
            loss_func_single=lossfun, loss_weight=w,
        )
    else:
        return PSFLearner(dataobj, psfobj, optimizer, lossfun, loss_weight=w)


def learn_psf(
    param: DictConfig,
    dataobj,
    psf_info: dict,
    time: Optional[float] = None,
    do_voxel_refit: bool = True,
) -> Tuple:
    """Run the PSF fitting optimisation.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    dataobj : PreprocessedImageData
        Prepared data object with extracted ROIs.
    psf_info : dict
        As returned by :func:`psf_registry.get_psf_info`.
    time : float, optional
        Start-time stamp for progress reporting.
    do_voxel_refit : bool, optional
        Whether to perform voxel-based refit after initial learning. Default True.

    Returns
    -------
    tuple of (PSFInterface, PSFLearner, list, float)
        ``(psfobj, learner, learning_result, toc)`` where toc is the end time.
    """
    psfobj = initialize_psf(param, psf_info)
    Reader.load_initial_pupil(param, psfobj, dataobj)

    learner = create_learner(param, dataobj, psfobj, psf_info)

    _, _, centers, _ = dataobj.get_image_data()

    res, toc = learner.learn_psf(start_time=time)

    if do_voxel_refit and param.PSFtype == "voxel":
        res, toc = _refit_voxel(
            res, centers, dataobj, learner,
            param.roi.roi_size, time,
        )

    return psfobj, learner, res, toc


def create_localizer(learner: PSFLearner) -> Localizer:
    """Create a Localizer instance from a PSFLearner's state."""
    return Localizer(
        data=learner.data,
        psf=learner.psf,
        rois=learner.rois,
        forward_images=learner.forward_images,
    )


def localize(
    learner: PSFLearner,
    res: list,
    toc: float,
    param: DictConfig,
) -> Tuple[list, list, float]:
    """Perform localization using the learned PSF.

    Parameters
    ----------
    learner : PSFLearner
        The learner instance with fitted PSF.
    res : list
        Learning result (optimized variables).
    toc : float
        End time from learning.
    param : DictConfig
        Experiment parameters.

    Returns
    -------
    tuple of (res, locres, toc)
        Updated learning result, localization result, and end time.
    """
    localizer = create_localizer(learner)

    _, _, _, file_idxs = learner.data.get_image_data()
    channeltype = param.channeltype

    if len(file_idxs) == 1:
        locres = localizer.localize(
            res, channeltype,
            usecuda=param.usecuda,
            plot=param.plotall,
            start_time=toc,
        )
        return res, locres, toc

    return _localize_with_relearn(learner, localizer, res, toc, param)


def _localize_with_relearn(
    learner: PSFLearner,
    localizer: Localizer,
    res: list,
    toc: float,
    param: DictConfig,
) -> Tuple[list, list, float]:
    """Internal: localization with outlier rejection and re-learning."""
    channeltype = param.channeltype
    psf_type = param.PSFtype
    rej_threshold = list(param.rej_threshold.values())

    if "insitu" in psf_type:
        res1, toc = learner.relearn_smlm(
            res, localizer, channeltype, rej_threshold, start_time=toc,
        )
        localizer.rois = learner.rois
        localizer.forward_images = learner.forward_images
        locres = localizer.localize_smlm(res1, channeltype, plot=param.plotall)
    else:
        locres = localizer.localize(
            res, channeltype,
            usecuda=param.usecuda,
            plot=param.plotall,
            start_time=toc,
        )
        toc = locres[-2]

        res1, toc = learner.relearn(
            res, localizer, channeltype, rej_threshold, start_time=toc,
        )

        if res1[0].shape[-2] < res[0].shape[-2]:
            localizer.rois = learner.rois
            localizer.forward_images = learner.forward_images
            locres = localizer.localize(
                res1, channeltype,
                usecuda=param.usecuda,
                plot=param.plotall,
                start_time=toc,
            )

    return res1, locres, toc


def relearn(
    learner: PSFLearner,
    localizer: Localizer,
    res: list,
    toc: float,
    param: DictConfig,
    threshold: Optional[list] = None,
) -> Tuple[list, float]:
    """Re-learn PSF after rejecting outliers based on localization metrics.

    Parameters
    ----------
    learner : PSFLearner
        The learner instance.
    localizer : Localizer
        Localizer instance with computed rejection metrics.
    res : list
        Current learning result.
    toc : float
        Current end time.
    param : DictConfig
        Experiment parameters.
    threshold : list, optional
        Custom rejection thresholds. If None, uses param.rej_threshold.

    Returns
    -------
    tuple of (res, toc)
        Updated learning result and end time.
    """
    if threshold is None:
        threshold = list(param.rej_threshold.values())

    channeltype = param.channeltype
    psf_type = param.PSFtype

    if "insitu" in psf_type:
        return learner.relearn_smlm(
            res, localizer, channeltype, threshold, start_time=toc,
        )
    else:
        return learner.relearn(
            res, localizer, channeltype, threshold, start_time=toc,
        )


def _refit_voxel(
    res, centers, dataobj, learner, roi_size, time
):
    """If the PSF type is *voxel* and z-drift is large, re-cut ROIs and re-fit."""
    pos = res[-1][0]
    zpos = pos[:, 0:1]
    zpos = zpos - np.mean(zpos)

    if centers.shape[-1] != 3 or np.max(np.abs(zpos)) <= 2:
        return res, time

    cor = dataobj.centers
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
    offset = np.min((np.quantile(dataobj.rois, 1e-3), 0))
    dataobj.rois = dataobj.rois - offset
    if dataobj.skew_const:
        dataobj.deskew_roi(roi_size)

    learner.data = dataobj
    res, toc = learner.learn_psf(start_time=time)
    return res, toc


def localize_FD(
    param: DictConfig, learning_result, learner, initz=None
):
    """Localise in the Fourier domain.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    learning_result : list
        Learning result from learn_psf.
    learner : PSFLearner
        The learner instance.
    initz : optional
        Initial z offset for localization.

    Returns
    -------
    loc_FD
        Localisation result in the Fourier domain.
    """
    localizer = create_localizer(learner)
    return localizer.localize_FD(
        learning_result, param.channeltype,
        usecuda=param.usecuda, initz=initz, plot=param.plotall,
    )


def iterlearn_psf(
    learn_psf_fn: Callable,
    save_result_fn: Callable,
    load_initial_pupil_fn: Callable,
    get_psf_info_fn: Callable,
    param: DictConfig,
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
    loc_FD = None
    psf_info = get_psf_info_fn(param)

    for nn in range(iterN):
        if nn > 0:
            dataobj.resetdata()

        psfobj, learner, learning_result, toc = learn_psf_fn(
            param, dataobj, psf_info, time=time,
        )

        localizer = create_localizer(learner)
        _, _, _, file_idxs = learner.data.get_image_data()

        if len(file_idxs) > 1:
            _, learning_result, toc = localize(learner, learning_result, toc, param)

        param.savename = savename + str(nn)
        resfile = save_result_fn(
            param, psfobj, dataobj, learner,
            learning_result, None, loc_FD,
        )
        param.option.model.init_pupil_file = resfile
        param.option.insitu.min_photon = max(min_photon - nn * 0.1, 0.2)

        _update_stage_pos(psfobj, learning_result, dataobj, param)

    return resfile


def _update_stage_pos(psfobj, learning_result, dataobj, param):
    """Update the stage position parameter from the current fit."""
    channeltype = param.channeltype
    res = psfobj.res2dict(learning_result)

    if channeltype == "single":
        param.option.insitu.stage_pos = float(res["stagepos"])
    else:
        try:
            param.option.insitu.stage_pos = float(
                res["channel0"]["stagepos"]
            )
        except (KeyError, TypeError):
            pass


def learn_and_localize(
    param: DictConfig,
    dataobj,
    psf_info: dict,
    time: Optional[float] = None,
) -> Tuple:
    """Convenience function that runs full pipeline: learn PSF, then localize.

    Parameters
    ----------
    param : DictConfig
        Experiment parameters.
    dataobj : PreprocessedImageData
        Prepared data object with extracted ROIs.
    psf_info : dict
        As returned by :func:`psf_registry.get_psf_info`.
    time : float, optional
        Start-time stamp for progress reporting.

    Returns
    -------
    tuple of (PSFInterface, PSFLearner, Localizer, list, list)
        ``(psfobj, learner, localizer, learning_result, loc_result)``
    """
    psfobj, learner, res, toc = learn_psf(param, dataobj, psf_info, time=time)
    localizer = create_localizer(learner)

    _, _, _, file_idxs = learner.data.get_image_data()

    if len(file_idxs) == 1:
        locres = localizer.localize(
            res, param.channeltype,
            usecuda=param.usecuda,
            plot=param.plotall,
            start_time=toc,
        )
        return psfobj, learner, localizer, res, locres

    res, locres, toc = _localize_with_relearn(learner, localizer, res, toc, param)
    return psfobj, learner, localizer, res, locres