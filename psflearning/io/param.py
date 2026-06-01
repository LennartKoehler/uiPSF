from __future__ import annotations

from pathlib import Path
from typing import Union
import os
from omegaconf import DictConfig, OmegaConf

_MappingTypes = (dict, DictConfig)


def load(path: Union[str, Path]) -> DictConfig:
    """Load a YAML configuration file and return it as a DictConfig."""
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    return cfg


def combine(
    basefile: str,
    psftype: str | None = None,
    channeltype: str | None = None,
    sysfile: str | None = None,
) -> DictConfig:
    """Load a base config and layer in psftype, channeltype, and systemtype overrides."""
    thispath = os.path.dirname(os.path.abspath(__file__))
    pkgpath = os.path.dirname(os.path.dirname(thispath))
    fparam = load(pkgpath + "/config/" + basefile + ".yaml").Params
    if psftype is not None:
        psfparam = load(pkgpath + "/config/psftype/" + psftype + ".yaml").Params
        fparam = redefine(fparam, psfparam)
    if channeltype is not None:
        chparam = load(pkgpath + "/config/channeltype/" + channeltype + ".yaml").Params
        fparam = redefine(fparam, chparam)
    if sysfile is not None:
        sysparam = load(pkgpath + "/config/systemtype/" + sysfile + ".yaml").Params
        fparam = redefine(fparam, sysparam)
    if psftype == "zernike" and channeltype == "4pi":
        fparam.PSFtype = "zernike"
    if psftype is not None and "insitu" in psftype:
        fparam.roi.gauss_sigma[-1] = max([4, fparam.roi.gauss_sigma[-1]])
        fparam.roi.gauss_sigma[-2] = max([4, fparam.roi.gauss_sigma[-2]])
        fparam.roi.max_kernel[-1] = max([5, fparam.roi.max_kernel[-1]])
        fparam.roi.max_kernel[-2] = max([5, fparam.roi.max_kernel[-2]])
    if psftype is not None and "FD" in psftype:
        fparam.option.model.bin = 1

    return fparam


def redefine(baseparam: DictConfig, userparam: DictConfig) -> DictConfig:
    """Recursively merge *userparam* into *baseparam*, overwriting leaf values."""
    for k, v in userparam.items():
        if isinstance(v, _MappingTypes) and isinstance(baseparam.get(k), _MappingTypes):
            redefine(baseparam[k], v)
        else:
            baseparam[k] = v

    return baseparam
