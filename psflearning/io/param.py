from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, List, Optional, Union
import os
from omegaconf import DictConfig, OmegaConf

_MappingTypes = (dict, DictConfig)


def load(path: Union[str, Path]) -> DictConfig:
    """Load a YAML configuration file and return it as a DictConfig."""
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    return cfg


# ── Nested parameter dataclasses ────────────────────────────────────────


@dataclass
class RefractiveIndices:
    imm: float = 1.516
    med: float = 1.335
    cov: float = 1.516


@dataclass
class ImagingParams:
    emission_wavelength: float = 0.68
    NA: float = 1.43
    RI: RefractiveIndices = field(default_factory=RefractiveIndices)


@dataclass
class InsituOptionParams:
    stage_pos: float = 1.0
    min_photon: float = 0.4
    partition_data: bool = True
    partition_size: list = field(default_factory=lambda: [21, 100])
    zernike_index: list = field(default_factory=lambda: [5])
    zernike_coeff: list = field(default_factory=lambda: [0.5])
    z_range: float = 2.0
    zkorder_rank: str = "L"
    var_stagepos: bool = True
    repeat: int = 2
    backgroundROI: list = field(default_factory=list)


@dataclass
class FpiOptionParams:
    link_zernikecoeff: bool = True
    phase_dm: list = field(default_factory=lambda: [2, 0, -2])
    sampleheight: int = 2
    var_sampleheight: bool = False
    phase_delay_dir: str = "descend"


@dataclass
class MultiOptionParams:
    defocus_offset: float = 0.0
    defocus_delay: float = -0.0
    defocus: Optional[list] = None


@dataclass
class ModelParams:
    pupilsize: int = 64
    n_max: int = 8
    zernike_nl: list = field(default_factory=list)
    blur_sigma: float = 0.5
    var_blur: bool = True
    with_apoid: bool = True
    const_pupilmag: bool = False
    symmetric_mag: bool = False
    with_IMM: bool = False
    init_pupil_file: str = ""
    estimate_drift: bool = False
    var_photon: bool = False
    bin: int = 2
    division: int = 40
    search_radius: float = 0.0


@dataclass
class OptionParams:
    imaging: ImagingParams = field(default_factory=ImagingParams)
    insitu: InsituOptionParams = field(default_factory=InsituOptionParams)
    fpi: FpiOptionParams = field(default_factory=FpiOptionParams)
    multi: MultiOptionParams = field(default_factory=MultiOptionParams)
    model: ModelParams = field(default_factory=ModelParams)


@dataclass
class RoiParams:
    roi_size: list = field(default_factory=lambda: [21, 21])
    gauss_sigma: list = field(default_factory=lambda: [2, 2])
    max_kernel: list = field(default_factory=lambda: [3, 3])
    peak_height: float = 0.2
    max_bead_number: int = 40
    bead_radius: float = 0.0


@dataclass
class PixelSizeParams:
    x: float = 0.127
    y: float = 0.116
    z: float = 0.05


@dataclass
class FOVParams:
    y_center: int = 0
    x_center: int = 0
    radius: int = 0
    z_start: int = 0
    z_end: int = 0
    z_step: int = 1

    def values(self):
        return [self.y_center, self.x_center, self.radius,
                self.z_start, self.z_end, self.z_step]


@dataclass
class DualParams:
    mirrortype: str = "up-down"
    channel_arrange: str = "up-down"


@dataclass
class MultiChannelParams:
    channel_size: list = field(default_factory=list)


@dataclass
class FpiParams:
    modulation_period: float = 0.26


@dataclass
class LLSParams:
    skew_const: list = field(default_factory=lambda: [0, 0])


@dataclass
class InsituParams:
    frame_range: list = field(default_factory=lambda: [0, 3000])


@dataclass
class LossWeightParams:
    mse1: int = 1
    mse2: int = 1
    smooth: int = 0
    edge: float = 0.01
    psf_min: int = 1
    bg_min: int = 1
    photon_min: int = 1
    Inorm: int = 0
    gxy_min: int = 10

    def values(self):
        return [self.mse1, self.mse2, self.smooth, self.edge,
                self.psf_min, self.bg_min, self.photon_min,
                self.Inorm, self.gxy_min]


@dataclass
class RejThresholdParams:
    bias_z: float = 0.99
    mse: float = 0.8
    photon: float = 1.5

    def values(self):
        return [self.bias_z, self.mse, self.photon]


# ── Top-level RunParameters ────────────────────────────────────────────


@dataclass
class RunParameters:
    datapath: str = ""
    keyword: str = "Default."
    savename: str = ""
    subfolder: str = ""
    format: str = ".tif"
    stage_mov_dir: str = "normal"
    gain: float = 0.2
    ccd_offset: float = 398.6
    roi: RoiParams = field(default_factory=RoiParams)
    pixel_size: PixelSizeParams = field(default_factory=PixelSizeParams)
    FOV: FOVParams = field(default_factory=FOVParams)
    dual: DualParams = field(default_factory=DualParams)
    multi: MultiChannelParams = field(default_factory=MultiChannelParams)
    fpi: FpiParams = field(default_factory=FpiParams)
    LLS: LLSParams = field(default_factory=LLSParams)
    insitu: InsituParams = field(default_factory=InsituParams)
    option: OptionParams = field(default_factory=OptionParams)
    PSFtype: str = "insitu_zernike"
    channeltype: str = "single"
    datatype: str = "smlm"
    loss_weight: LossWeightParams = field(default_factory=LossWeightParams)
    rej_threshold: RejThresholdParams = field(default_factory=RejThresholdParams)
    usecuda: bool = True
    plotall: bool = False
    ref_channel: int = 0
    batch_size: int = 1600
    iteration: int = 200
    varname: str = ""
    filelist: list = field(default_factory=list)
    swapxy: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ── Construction from YAML ─────────────────────────────────────────────


def combine(
    basefile: str,
    psftype: Optional[str] = None,
    channeltype: Optional[str] = None,
    sysfile: Optional[str] = None,
) -> RunParameters:
    """Load a base config and layer in psftype, channeltype, and
    systemtype overrides, returning a typed :class:`RunParameters`."""
    thispath = os.path.dirname(os.path.abspath(__file__))
    pkgpath = os.path.dirname(os.path.dirname(thispath))
    fparam = load(pkgpath + "/config/" + basefile + ".yaml").Params
    if psftype is not None:
        psfparam = load(pkgpath + "/config/psftype/" + psftype + ".yaml").Params
        fparam = _redefine(fparam, psfparam)
    if channeltype is not None:
        chparam = load(pkgpath + "/config/channeltype/" + channeltype + ".yaml").Params
        fparam = _redefine(fparam, chparam)
    if sysfile is not None:
        sysparam = load(pkgpath + "/config/systemtype/" + sysfile + ".yaml").Params
        fparam = _redefine(fparam, sysparam)
    if psftype == "zernike" and channeltype == "4pi":
        fparam.PSFtype = "zernike"
    if psftype is not None and "insitu" in psftype:
        fparam.roi.gauss_sigma[-1] = max([4, fparam.roi.gauss_sigma[-1]])
        fparam.roi.gauss_sigma[-2] = max([4, fparam.roi.gauss_sigma[-2]])
        fparam.roi.max_kernel[-1] = max([5, fparam.roi.max_kernel[-1]])
        fparam.roi.max_kernel[-2] = max([5, fparam.roi.max_kernel[-2]])
    if psftype is not None and "FD" in psftype:
        fparam.option.model.bin = 1

    return _dictconfig_to_params(fparam)


def _dictconfig_to_params(cfg: DictConfig) -> RunParameters:
    """Convert a merged DictConfig into a :class:`RunParameters`."""
    return _from_dataclass_cfg(RunParameters, cfg)


def _from_dataclass_cfg(cls, cfg):
    """Recursively construct a dataclass *cls* from a DictConfig *cfg*.

    For each field in *cls*:
    - If the field type is itself a dataclass, recurse.
    - Otherwise, coerce the value from *cfg* to the declared type,
      falling back to the field default if the key is missing.
    """
    kwargs = {}
    cfg_map = cfg if isinstance(cfg, _MappingTypes) else {}

    for f in fields(cls):
        raw = cfg_map.get(f.name) if isinstance(cfg_map, _MappingTypes) else None
        ftype = f.type

        if isinstance(ftype, str):
            ftype = _resolve_type(cls, ftype)

        if raw is None:
            if f.default is not MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not MISSING:
                kwargs[f.name] = f.default_factory()
            else:
                kwargs[f.name] = None
            continue

        if _is_dataclass_type(ftype):
            nested_cfg = raw if isinstance(raw, _MappingTypes) else OmegaConf.create({})
            kwargs[f.name] = _from_dataclass_cfg(ftype, nested_cfg)
        else:
            kwargs[f.name] = _coerce(ftype, raw)

    return cls(**kwargs)


def _is_dataclass_type(tp) -> bool:
    try:
        return issubclass(tp, object) and hasattr(tp, "__dataclass_fields__")
    except TypeError:
        return False


def _resolve_type(cls, type_name: str):
    """Resolve a forward-reference type string to the actual class
    from the module where *cls* is defined."""
    import sys
    mod = sys.modules.get(cls.__module__, None)
    if mod is not None and hasattr(mod, type_name):
        return getattr(mod, type_name)
    return None


def _coerce(ftype, value):
    """Coerce *value* to *ftype*, handling common OmegaConf types."""
    if ftype in (int, float, str, bool):
        return ftype(value)
    if ftype is list:
        return list(value)
    if ftype is Optional:
        return value
    origin = getattr(ftype, "__origin__", None)
    if origin is list:
        return list(value)
    if origin is Union:
        args = [a for a in getattr(ftype, "__args__", ()) if a is not type(None)]
        if args:
            return _coerce(args[0], value)
    return value


def _redefine(baseparam: DictConfig, userparam: DictConfig) -> DictConfig:
    """Recursively merge *userparam* into *baseparam*, overwriting leaf values."""
    for k, v in userparam.items():
        if isinstance(v, _MappingTypes) and isinstance(baseparam.get(k), _MappingTypes):
            _redefine(baseparam[k], v)
        else:
            baseparam[k] = v
    return baseparam
