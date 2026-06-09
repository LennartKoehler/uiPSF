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
>>> psf_model, fit_result, locres, toc = lib.learn_with_relearn(param, dataobj, psf_info)
>>> resfile = lib.save_result(param, psf_model, dataobj, fit_result, locres)

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

from .reader import Reader
from .writer import Writer
from .plotter import Plotter
from .psf_registry import get_psf_info
from .fitting import (
    initialize_psf,
    learn_psf,
    learn_psf_with_relearn,
    relearn,
)
from .analysis import genpsf, calstrehlratio, calfwhm

from .learning import PSFLearner, LocalizationOutput
from .learning.fitters.Localizer import localize
from .learning.psfs.PSFZernikeBased import ZernikePSFResult
from .learning.psfs.PSFInterface import PSFInterface
from .learning.loclib import LocalizationResult
from .io.param import RunParameters
from .learning.psf_variables import LocResResult, PSFInfo, PSFResult, ROIsResult
from .learning.data_representation.PreprocessedImageDataSingleChannel import PreprocessedImageDataSingleChannel


class PSFLearningLib:
    """High-level PSF-learning workflow orchestrator.

    Each public method delegates to the corresponding sub-module so that
    the sub-modules can also be used directly without instantiating this
    class.  ``param`` is always passed explicitly – the orchestrator is
    stateless with respect to experiment configuration.

    The :attr:`plotter` property exposes the :class:`Plotter` instance
    for direct access to visualisation methods.
    """

    def __init__(self,
            reader: Reader,
            writer: Writer
                 ) -> None:
        self._reader = reader
        self._writer = writer

    def run(self, parameters: RunParameters) -> str:

        images = self._reader.read_images(parameters)
        psf_info = get_psf_info(parameters)
        dataobj = self._prep_data(parameters, images)

        if parameters.relearn:
            psf_model, learning_result, loc_result, forward_images, toc = learn_psf_with_relearn(parameters, dataobj, psf_info, time=0)

        else:
            psf_model, learning_result, _, forward_images, toc = learn_psf(parameters, dataobj, psf_info, time=0)

        resfile = self._writer.save_result(parameters, psf_model, dataobj, learning_result, loc_result, forward_images=forward_images)
        return resfile



    def _prep_data(self, param: Union[RunParameters, DictConfig], images: np.ndarray):
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
        roi_size = param.roi.roi_size
        fov = list(param.FOV.values())
        skew_const = param.LLS.skew_const

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
            gaus_sigma=param.roi.gauss_sigma,
            min_border_dist=list(np.array(roi_size) // 2 + 1),
            min_center_dist=np.max(roi_size),
            FOV=fov_param,
            max_threshold=param.roi.peak_height,
            max_kernel=param.roi.max_kernel,
            pixelsize_x=param.pixel_size.x,
            pixelsize_y=param.pixel_size.y,
            pixelsize_z=param.pixel_size.z,
            bead_radius=param.roi.bead_radius,
            plot=param.plotall,
            padPSF=True,
            isVolume=False,
            skew_const=skew_param,
            max_bead_number=param.roi.max_bead_number,
        )
        return dataobj


