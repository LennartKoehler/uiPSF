from __future__ import annotations

import numpy as np

from ..loclib import localizationlib, LocalizationResult
from ...io.param import RunParameters
from omegaconf import DictConfig
from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class LocalizationOutput:
    """Result from Localizer.localize(), bundling the localization result
    with the rejection metrics computed during localization."""
    locres: Any
    reject_metric: Optional[List[np.ndarray]] = None
    minI: Optional[np.ndarray] = None


"""
Handles localization given a learned PSF model.

Stateless with respect to data and PSF. All data, PSF model, ROIs,
and forward images are passed explicitly to each method call.
"""

def localize(
    pixelsize_z: np.ndarray,
    psf_model_image: np.ndarray,
    rois: np.ndarray,
    param: Union[RunParameters, DictConfig],
    toc: Optional[float] = None,
) -> LocalizationResult:
    """Perform localization using the learned PSF.

    Pure localization -- no relearning.  For non-insitu PSF types the
    localizer also computes rejection metrics accessible via
    ``localizer.reject_metric`` and ``localizer.minI``.

    Parameters
    ----------
    learner : PSFLearner
        The learner instance with fitted PSF.
    res : list
        Learning result (optimized variables).
    param : DictConfig
        Experiment parameters.
    toc : float, optional
        End time from learning.

    Returns
    -------
    tuple of (Localizer, list, float)
        ``(localizer, locres, toc)``
    """
    dll = localizationlib(usecuda=param.usecuda)
    locres = dll.loc_ast(rois, psf_model_image, pixelsize_z, initz=None, start_time=toc)

    return locres
