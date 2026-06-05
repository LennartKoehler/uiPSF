"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu

Main orchestrator for the PSF-learning pipeline.

Delegates to focused sub-modules:

* :class:`Reader`      – all input operations
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
>>> psfobj, fitter, res, loc = lib.learn_psf(param, dataobj, psf_info)
>>> resfile = lib.save_result(param, psfobj, dataobj, fitter, res, loc)

**2. Insitu iterative learning:**

>>> images = lib.load_data(param)
>>> psf_info = lib.get_psf_info(param)
>>> dataobj = lib.prep_data(param, images)
>>> resfile = lib.iterlearn_psf(param, dataobj)

**3. Evaluate / generate PSF from saved results:**

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
    learn_with_relearn,
    localize,
    relearn,
    localize_FD,
    iterlearn_psf as _iterlearn_psf,
)
from .analysis import genpsf, calstrehlratio, calfwhm

from .learning import PSFLearner, Localizer
from .learning.psfs.PSFZernikeBased import ZernikePSFResult
from .learning.psfs.PSFInterface import PSFInterface
from .learning.loclib import LocalizationResult
from .io.param import RunParameters
from .learning.psf_variables import LocResResult, PSFInfo, PSFResult, ROIsResult


class PSFLearningLib:
    """High-level PSF-learning workflow orchestrator.

    Each public method delegates to the corresponding sub-module so that
    the sub-modules can also be used directly without instantiating this
    class.  ``param`` is always passed explicitly – the orchestrator is
    stateless with respect to experiment configuration.

    The :attr:`plotter` property exposes the :class:`Plotter` instance
    for direct access to visualisation methods.
    """

    def __init__(self) -> None:
        self._reader = Reader()
        self._writer = Writer()
        self._plotter = Plotter()

    @property
    def plotter(self) -> Plotter:
        """Access the :class:`Plotter` for visualisation methods."""
        return self._plotter

    # ── Registry ─────────────────────────────────────────────────────

    @staticmethod
    def get_psf_info(param: Union[RunParameters, DictConfig]) -> PSFInfo:
        """Resolve *param* to PSF class, multi-channel class, and loss
        functions.

        Returns
        -------
        PSFInfo
            Typed container with ``psf_class``, ``psf_class_multi``,
            ``loss_fun``, ``loss_fun_multi``.
        """
        return get_psf_info(param)

    # ── Reading (delegates to Reader) ────────────────────────────────

    def load_data(
        self, param: Union[RunParameters, DictConfig], frange: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Load raw image stacks from disk.

        See :meth:`Reader.read_images`.
        """
        return self._reader.read_images(param, frange=frange)

    def prep_data(self, param: Union[RunParameters, DictConfig], images: np.ndarray):
        """Detect beads / localisations and build a data object.

        See :meth:`Reader.prep_data`.
        """
        return self._reader.prep_data(param, images)

    # ── Fitting ──────────────────────────────────────────────────────

    @staticmethod
    def initialize_psf(param: Union[RunParameters, DictConfig], psf_info: PSFInfo):
        """Create a PSF model object.

        See :func:`fitting.initialize_psf`.
        """
        return initialize_psf(param, psf_info)

    @staticmethod
    def learn_psf(
        param: Union[RunParameters, DictConfig], dataobj, psf_info: PSFInfo, time: Optional[float] = None
    ) -> Tuple[PSFInterface, ZernikePSFResult, Optional[float]]:
        """Run PSF fitting.

        Returns
        -------
        tuple
            ``(psfobj, learner, learning_result, toc)``

        See :func:`fitting.learn_psf`.
        """
        return learn_psf(param, dataobj, psf_info, time=time)

    @staticmethod
    def learn_with_relearn(
        param: Union[RunParameters, DictConfig], dataobj, psf_info: PSFInfo, time: Optional[float] = None
    ) -> Tuple[PSFInterface, ZernikePSFResult, LocalizationResult, Optional[float]]:
        """Learn PSF, localize, remove outliers and re-learn.

        Returns
        -------
        tuple
            ``(psfobj, learner, localizer, learning_result, loc_result)``

        See :func:`fitting.learn_with_relearn`.
        """
        return learn_with_relearn(param, dataobj, psf_info, time=time)

    @staticmethod
    def localize_psf(
        learner, res, param: Union[RunParameters, DictConfig], toc: Optional[float] = None
    ) -> tuple:
        """Run localization using a learned PSF.

        Returns
        -------
        tuple
            ``(localizer, loc_result, toc)``

        See :func:`fitting.localize`.
        """
        return localize(learner, res, param, toc=toc)

    @staticmethod
    def relearn_psf(
        learner, res, param: Union[RunParameters, DictConfig], toc: Optional[float] = None,
        threshold: Optional[list] = None,
    ) -> tuple:
        """Re-learn PSF after rejecting outliers, then re-localize.

        Returns
        -------
        tuple
            ``(localizer, res, loc_result, toc)``

        See :func:`fitting.relearn`.
        """
        return relearn(learner, res, param, toc=toc, threshold=threshold)

    @staticmethod
    def localize_fd(
        param: Union[RunParameters, DictConfig], learning_result, fitter, initz=None
    ):
        """Localise in the Fourier domain.

        See :func:`fitting.localize_FD`.
        """
        return localize_FD(param, learning_result, fitter, initz=initz)

    def iterlearn_psf(
        self, param: Union[RunParameters, DictConfig], dataobj, time: Optional[float] = None
    ) -> str:
        """Iterative PSF learning for insitu data.

        On each iteration the result of the previous iteration is used to
        initialise the pupil for the next one, and the photon threshold
        is gradually relaxed.

        .. note::
           This method mutates the following fields of *param*:
           ``savename``, ``option.model.init_pupil_file``,
           ``option.insitu.min_photon``, ``option.insitu.stage_pos``.

        See :func:`fitting.iterlearn_psf`.
        """
        return _iterlearn_psf(
            learn_psf_fn=learn_psf,
            save_result_fn=self._writer.save_result,
            load_initial_pupil_fn=Reader.load_initial_pupil,
            get_psf_info_fn=get_psf_info,
            param=param,
            dataobj=dataobj,
            time=time,
        )

    # ── Writing (delegates to Writer) ────────────────────────────────

    def save_result(
        self,
        param: Union[RunParameters, DictConfig],
        psfobj,
        dataobj,
        learning_result: list,
        loc_result: list,
        loc_FD=None,
    ) -> str:
        """Save results to HDF5.

        See :meth:`Writer.save_result`.
        """
        return self._writer.save_result(
            param, psfobj, dataobj,
            learning_result, loc_result, loc_FD=loc_FD,
        )

    def write_h5(
        self,
        param: Union[RunParameters, DictConfig],
        filename: str,
        res: PSFResult,
        locres: LocResResult,
        rois: ROIsResult,
    ) -> None:
        """Write result dataclasses to an HDF5 file.

        See :meth:`Writer.write_h5`.
        """
        return self._writer.write_h5(
            param, filename, res, locres, rois
        )

    def generate_cspline(
        self, param: Union[RunParameters, DictConfig], res: PSFResult, psfobj, keyname: str = "I_model"
    ):
        """Generate cubic-spline coefficients.

        See :meth:`Writer.generate_cspline`.
        """
        return self._writer.generate_cspline(
            param, res, psfobj, keyname=keyname
        )

    # ── Analysis (pure computation) ──────────────────────────────────

    @staticmethod
    def genpsf(
        param: Union[RunParameters, DictConfig], f, psf_info: Optional[PSFInfo] = None,
        Nz: int = 21, xsz: int = 21, stagepos: float = 1.0,
    ) -> tuple:
        """Generate a PSF model from fitted parameters.

        See :func:`analysis.genpsf`.
        """
        if psf_info is None:
            psf_info = get_psf_info(param)
        return genpsf(param, f, psf_info, Nz=Nz, xsz=xsz, stagepos=stagepos)

    @staticmethod
    def calstrehlratio(
        param: Union[RunParameters, DictConfig], f, psf_info: Optional[PSFInfo] = None, xsz: int = 31
    ):
        """Compute the Strehl ratio.

        See :func:`analysis.calstrehlratio`.
        """
        if psf_info is None:
            psf_info = get_psf_info(param)
        return calstrehlratio(param, f, psf_info, xsz=xsz)

    @staticmethod
    def calfwhm(
        param: Union[RunParameters, DictConfig], f, psf_info: Optional[PSFInfo] = None
    ) -> tuple:
        """Compute the FWHM.

        See :func:`analysis.calfwhm`.
        """
        if psf_info is None:
            psf_info = get_psf_info(param)
        return calfwhm(param, f, psf_info)


# ── Backward-compatible alias ────────────────────────────────────────────


def __getattr__(name):
    """Provide ``psflearninglib`` as a deprecated alias for
    :class:`PSFLearningLib`."""
    if name == "psflearninglib":
        warnings.warn(
            "psflearninglib is deprecated; use PSFLearningLib instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return PSFLearningLib
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
