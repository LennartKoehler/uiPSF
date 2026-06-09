"""
Registry mapping PSF type strings to their implementation classes and
corresponding loss functions.
"""

from __future__ import annotations

from omegaconf import DictConfig
from typing import Union

from .learning import (
    PSFZernikeBased,
    mse_real_zernike,
)
from .learning.psf_variables import PSFInfo
from .io.param import RunParameters


def get_psf_info(param: Union[RunParameters, DictConfig]) -> PSFInfo:
    """Resolve *param* into the concrete PSF class and loss function.

    Returns
    -------
    PSFInfo
        Typed container with ``psf_class`` and ``loss_fun``.
    """

    return PSFInfo(
        psf_class=PSFZernikeBased,
        loss_fun=mse_real_zernike,
    )
