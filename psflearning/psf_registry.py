"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu

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

    """Resolve *param* into the concrete PSF class, multi-channel class,
    loss function, and multi-channel loss function.

    Returns
    -------
    PSFInfo
        Typed container with ``psf_class``, ``psf_class_multi``,
        ``loss_fun``, ``loss_fun_multi``.
    """

    return PSFInfo(
        psf_class=PSFZernikeBased,
        psf_class_multi=None,
        loss_fun=mse_real_zernike,
        loss_fun_multi=None,
    )
