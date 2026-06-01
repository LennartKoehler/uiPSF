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

from .learning import Fitter, L_BFGS_B


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


def learn_psf(
    param: DictConfig,
    dataobj,
    psf_info: dict,
    time: Optional[float] = None,
    load_initial_pupil_fn: Callable = Reader.load_initial_pupil,
) -> tuple:
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
    load_initial_pupil_fn : callable, optional
        Function ``f(param, psfobj, dataobj) -> None`` that loads
        initial pupil state.  Defaults to
        :meth:`Reader.load_initial_pupil` when *None*.

    Returns
    -------
    tuple of (PSFInterface, Fitter, list, list)
        ``(psfobj, fitter, learning_result, loc_result)``
    """

    psfobj = initialize_psf(param, psf_info)
    load_initial_pupil_fn(param, psfobj, dataobj)

    lossfun = psf_info["loss_fun"]
    lossfunmulti = psf_info["loss_fun_multi"]
    w = list(param.loss_weight.values())

    optimizer = L_BFGS_B(maxiter=param.iteration)
    optimizer.batch_size = param.batch_size

    if lossfunmulti:
        fitter = Fitter(
            dataobj, psfobj, optimizer, lossfunmulti,
            loss_func_single=lossfun, loss_weight=w,
        )
    else:
        fitter = Fitter(dataobj, psfobj, optimizer, lossfun, loss_weight=w)

    _, _, centers, file_idxs = dataobj.get_image_data()
    centers = np.stack(centers)

    res, toc = fitter.learn_psf(start_time=time)

    res = _maybe_refit_voxel(
        res, centers, dataobj, fitter,
        param.roi.roi_size, param.PSFtype, time,
    )

    learning_result, loc_result = _localize(
        res, fitter, toc,
        file_idxs=file_idxs,
        channeltype=param.channeltype,
        psf_type=param.PSFtype,
        rej_threshold=list(param.rej_threshold.values()),
        usecuda=param.usecuda,
        showplot=param.plotall,
    )

    return psfobj, fitter, learning_result, loc_result


def _maybe_refit_voxel(
    res, centers, dataobj, fitter, roi_size, psf_type, time
):
    """If the PSF type is *voxel* and z-drift is large, re-cut ROIs and
    re-fit."""
    if psf_type != "voxel":
        return res

    pos = res[-1][0]
    zpos = pos[:, 0:1]
    zpos = zpos - np.mean(zpos)

    if centers.shape[-1] != 3 or np.max(np.abs(zpos)) <= 2:
        return res

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

    fitter.dataobj = dataobj
    res, _ = fitter.learn_psf(start_time=time)
    return res


def _localize(
    res, fitter, toc, *, file_idxs, channeltype, psf_type,
    rej_threshold, usecuda, showplot,
):
    """Perform localisation (and optional outlier-removal + re-learning)."""
    if len(file_idxs) == 1:
        locres = fitter.localize(
            res, channeltype, usecuda=usecuda, plot=showplot,
            start_time=toc,
        )
        return res, locres

    if "insitu" in psf_type:
        res1, toc = fitter.relearn_smlm(
            res, channeltype, rej_threshold, start_time=toc,
        )
        locres = fitter.localize_smlm(res1, channeltype, plot=showplot)
    else:
        locres = fitter.localize(
            res, channeltype, usecuda=usecuda, plot=showplot,
            start_time=toc,
        )
        toc = locres[-2]
        res1, toc = fitter.relearn(
            res, channeltype, rej_threshold, start_time=toc,
        )
        if res1[0].shape[-2] < res[0].shape[-2]:
            locres = fitter.localize(
                res1, channeltype, usecuda=usecuda, plot=showplot,
                start_time=toc,
            )
        else:
            locres = locres

    return res1, locres


def localize_FD(
    param: DictConfig, learning_result, fitter, initz=None
):
    """Localise in the Fourier domain.

    Returns
    -------
    loc_FD
        Localisation result in the Fourier domain.
    """
    return fitter.localize_FD(
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
    channeltype = param.channeltype
    savename = param.savename
    loc_FD = None
    psf_info = get_psf_info_fn(param)

    for nn in range(iterN):
        if nn > 0:
            dataobj.resetdata()

        psfobj, fitter, learning_result, loc_result = learn_psf_fn(
            param, dataobj, psf_info, time=time,
            load_initial_pupil_fn=load_initial_pupil_fn,
        )

        param.savename = savename + str(nn)
        resfile = save_result_fn(
            param, psfobj, dataobj, fitter,
            learning_result, loc_result, loc_FD,
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
