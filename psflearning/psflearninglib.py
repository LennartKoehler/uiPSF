"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu

Main orchestrator for the PSF-learning pipeline.

Delegates to focused sub-modules:

* :class:`Reader`      – image / parameter / result loading
* :class:`Writer`      – all output operations
* :class:`Plotter`     – all visualisations
* :mod:`psf_registry`  – PSF type → class / loss-function look-up
* :mod:`fitting`       – PSF fitting and iterative re-learning
* :mod:`analysis`      – PSF generation, Strehl ratio, FWHM

Typical workflows
-----------------
**1. Bead PSF learning:**

>>> lib = PSFLearningLib()
>>> images = lib.load_data(param)
>>> psf_info = lib.get_psf_info(param)
>>> dataobj = lib.prep_data(param, images)
>>> psf_model, fit_result, locres, toc, context = lib.learn_with_relearn(param, dataobj, psf_info)
>>> resfile = lib.save_result(param, context.pupil_field, dataobj, fit_result, locres)

**2. Evaluate / generate PSF from saved results:**

>>> from psflearning import io
>>> f, p = io.h5.load(resfile)
>>> lib.genpsf(param, f, psf_info)
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig

from psflearning.learning.data_representation.PreprocessedImageDataInterface import PreprocessedImageDataInterface
from psflearning.learning.psfs.IPSFModel import IPSFModel
from psflearning.learning.psfs.PSFZernikeBased import ZernikePSFResult
from psflearning.learning.psfs.PSFZernikeBase import PSFContext
from psflearning.learning.loclib import LocalizationResult


from .psf_registry import get_psf_info
from .fitting import (
    learn_psf,
    learn_psf_with_relearn,
)

from .io.param import RunParameters
from .learning.data_representation.PreprocessedImageDataSingleChannel import PreprocessedImageDataSingleChannel
from .learning.fitters.Localizer import localize


class PSFLearningLib:
    """High-level PSF-learning workflow orchestrator.

    Each public method delegates to the corresponding sub-module so that
    the sub-modules can also be used directly without instantiating this
    class.  ``param`` is always passed explicitly – the orchestrator is
    stateless with respect to experiment configuration.

    The :attr:`plotter` property exposes the :class:`Plotter` instance
    for direct access to visualisation methods.
    """


    @staticmethod
    def run(parameters: RunParameters, images: PreprocessedImageDataInterface) -> Tuple[
        RunParameters,
        IPSFModel,
        PreprocessedImageDataInterface,
        ZernikePSFResult,
        LocalizationResult,
        np.ndarray,
        PSFContext]:

        psf_info = get_psf_info(parameters)
        dataobj = PSFLearningLib._prep_data(parameters, images)

        if parameters.runtime.relearn:
            psf_model, learning_result, loc_result, forward_images, toc, context = learn_psf_with_relearn(parameters, dataobj, psf_info, time=0)
        else:
            psf_model, learning_result, _, _, forward_images, toc, context = learn_psf(parameters, dataobj, psf_info, time=0)
            loc_result = localize(dataobj.pixelsize_z, learning_result.psf_model_image_with_bead, dataobj.measured_roi_images, parameters, toc=toc)

        return parameters, psf_model, dataobj, learning_result, loc_result, forward_images, context




    @staticmethod
    def _prep_data( param: Union[RunParameters, DictConfig], images: np.ndarray):
        """Detect beads / localisations and build a data object.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters (ROI, pixel sizes, FOV, ...).
        images : numpy.ndarray
            Image array as returned by :meth:`read_images`.

        Returns
        -------
        PreprocessedImageData
            Data object with extracted ROIs ready for PSF fitting.
        """
        roi_size = param.selection.roi.roi_size
        fov = list(param.selection.FOV.values())
        skew_const = param.data.LLS.skew_const

        zstart = fov[-3]
        zend = images.shape[-3]+fov[-2]
        zstep = fov[-1]
        zind = range(zstart,zend,zstep)
        ims = np.swapaxes(images,0,-3)

        ims = ims[zind]
        images = np.swapaxes(ims,0,-3)

        dataobj = PreprocessedImageDataSingleChannel(images)

        fov_param = None if fov[2] == 0 else fov
        skew_param = (
            None
            if (skew_const[0] == 0.0 and skew_const[1] == 0.0)
            else skew_const
        )

        dataobj.process(
            roi_size=roi_size,
            gaus_sigma=param.selection.roi.gauss_sigma,
            min_border_dist=list(np.array(roi_size) // 2 + 1),
            min_center_dist=np.max(roi_size),
            FOV=fov_param,
            max_threshold=param.selection.roi.peak_height,
            max_kernel=param.selection.roi.max_kernel,
            pixelsize_x=param.data.pixel_size.x,
            pixelsize_y=param.data.pixel_size.y,
            pixelsize_z=param.data.pixel_size.z,
            bead_radius=param.selection.roi.bead_radius,
            plot=param.runtime.plotall,
            padPSF=True,
            isVolume=False,
            skew_const=skew_param,
            max_bead_number=param.selection.roi.max_bead_number,
        )
        return dataobj


