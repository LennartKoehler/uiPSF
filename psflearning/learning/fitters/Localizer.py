from __future__ import annotations

import numpy as np

from ..loclib import localizationlib, LocalizationResult
from ...progress import ProgressReporter
from ...io.param import RunParameters
from omegaconf import DictConfig
from typing import Union


def localize(
    pixelsize_z: float,
    psf_model_image: np.ndarray,
    rois: np.ndarray,
    param: Union[RunParameters, DictConfig],
    reporter: ProgressReporter,
) -> LocalizationResult:
    """Perform localization using the learned PSF model.

    Parameters
    ----------
    pixelsize_z : float
        Pixel size in z.
    psf_model_image : np.ndarray
        Learned PSF model image.
    rois : np.ndarray
        Measured ROI images.
    param : RunParameters or DictConfig
        Experiment parameters (uses ``param.runtime.use_cuda``).
    reporter : ProgressReporter
        Progress reporter.

    Returns
    -------
    LocalizationResult
    """
    dll = localizationlib(usecuda=param.runtime.use_cuda)
    return dll.loc_ast(rois, psf_model_image, pixelsize_z, initz=None, reporter=reporter)
