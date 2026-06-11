"""
Registry mapping PSF type strings to their implementation classes and
corresponding loss functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

from omegaconf import DictConfig

from .learning import (
    PSFZernikeBased,
    mse_real_zernike,
)
from .io.param import RunParameters


@dataclass
class PSFInfo:
    """PSF type registry entry.

    Replaces the ``dict(psf_class=..., ...)`` returned
    by ``psf_registry.get_psf_info()``.
    """

    psf_class: type
    loss_fun: Callable



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
