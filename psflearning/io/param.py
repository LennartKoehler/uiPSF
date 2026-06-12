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
    immersion: float = 1.516
    medium: float = 1.335
    coverslip: float = 1.516


@dataclass
class PSFModelParams:
    pupil_size: int = 64
    max_zernike_order: int = 8
    extra_blur_sigma: float = 0.5
    include_apodization: bool = True
    constant_pupil_magnitude: bool = False
    radially_symmetric_magnitude: bool = False
    include_index_mismatch: bool = False
    initial_pupil_file_path: str = ""
    estimate_drift: bool = False
    estimate_photon_variation: bool = False
    pixel_upsampling_factor: int = 2


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
class LLSParams:
    skew_translation_per_slice: list = field(default_factory=lambda: [0, 0])


@dataclass
class LossWeightParams:
    mse_weight: int = 1
    modified_mse_weight: int = 1
    smooth: int = 0
    edge: float = 0.01
    psf_positivity_weight: int = 1
    background_positivity_weight: int = 1
    photon_positivity_weight: int = 1
    intensity_normalization_weight: int = 0
    lateral_drift_weight: int = 10

    def values(self):
        return [self.mse_weight, self.modified_mse_weight, self.smooth, self.edge,
                self.psf_positivity_weight, self.background_positivity_weight,
                self.photon_positivity_weight,
                self.intensity_normalization_weight, self.lateral_drift_weight]


@dataclass
class RejThresholdParams:
    bias_z: float = 0.99
    mse: float = 0.8
    photon: float = 1.5

    def values(self):
        return [self.bias_z, self.mse, self.photon]


# ── Group dataclasses ──────────────────────────────────────────────────


@dataclass
class IOParams:
    data_path: str = ""
    keyword: str = "Default."
    output_path: str = ""
    subfolder: str = ""
    format: str = ".tif"
    variable_name: str = ""
    filelist: list = field(default_factory=list)


@dataclass
class DataParams:
    camera_gain: float = 0.2
    camera_offset: float = 398.6
    stage_movement_direction: str = "normal"
    swap_xy_dimensions: bool = False
    emission_wavelength: float = 0.68
    numerical_aperture: float = 1.43
    refractive_indices: RefractiveIndices = field(default_factory=RefractiveIndices)
    pixel_size: PixelSizeParams = field(default_factory=PixelSizeParams)
    lattice_light_sheet: LLSParams = field(default_factory=LLSParams)


@dataclass
class SelectionParams:
    roi: RoiParams = field(default_factory=RoiParams)
    FOV: FOVParams = field(default_factory=FOVParams)


@dataclass
class ModelConfig:
    psf_type: str = "zernike"
    psf: PSFModelParams = field(default_factory=PSFModelParams)
    loss_weight: LossWeightParams = field(default_factory=LossWeightParams)
    rej_threshold: RejThresholdParams = field(default_factory=RejThresholdParams)


@dataclass
class RuntimeParams:
    use_cuda: bool = True
    batch_size: int = 1600
    max_iterations: int = 200
    enable_relearning: bool = True
    plot_all_steps: bool = False


# ── Top-level RunParameters ────────────────────────────────────────────


@dataclass
class RunParameters:
    io: IOParams = field(default_factory=IOParams)
    data: DataParams = field(default_factory=DataParams)
    selection: SelectionParams = field(default_factory=SelectionParams)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeParams = field(default_factory=RuntimeParams)

    def to_dict(self) -> dict:
        return OmegaConf.to_container(OmegaConf.create(asdict(self)))


# ── Construction from YAML ─────────────────────────────────────────────

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_params(
    userfile: Optional[str] = None,
    psftype: Optional[str] = None,
    sysfile: Optional[str] = None,
) -> RunParameters:
    """Load parameters by layering configs on top of the base config.

    Merge order (each layer overwrites leaf values from the previous):

    1. **config_base.yaml** — always loaded first (complete defaults).
    2. *userfile* — sparse YAML with arbitrary overrides (only the fields
       that differ from the base need to be present).
    3. *psftype* — PSF-model-specific overrides.
    4. *sysfile* — microscope-system-specific overrides.

    Parameters
    ----------
    userfile : str, optional
        Name of a YAML file inside ``config/`` (without extension) whose
        ``Params`` key is merged on top of the base config.
    psftype : str, optional
        PSF type name (e.g. ``"zernike"``), resolved inside
        ``config/psftype/``.
    sysfile : str, optional
        System type name (e.g. ``"M2"``), resolved inside
        ``config/systemtype/``.

    Returns
    -------
    RunParameters
    """
    fparam = load(os.path.join(_PKG_DIR, "config", "config_base.yaml")).Params
    if userfile is not None:
        userparam = load(os.path.join(_PKG_DIR, "config", userfile + ".yaml")).Params
        fparam = _redefine(fparam, userparam)
    if psftype is not None:
        psfparam = load(os.path.join(_PKG_DIR, "config", "psftype", psftype + ".yaml")).Params
        fparam = _redefine(fparam, psfparam)
    if sysfile is not None:
        sysparam = load(os.path.join(_PKG_DIR, "config", "systemtype", sysfile + ".yaml")).Params
        fparam = _redefine(fparam, sysparam)
    if psftype is not None and "insitu" in psftype:
        fparam.selection.roi.gauss_sigma[-1] = max([4, fparam.selection.roi.gauss_sigma[-1]])
        fparam.selection.roi.gauss_sigma[-2] = max([4, fparam.selection.roi.gauss_sigma[-2]])
        fparam.selection.roi.max_kernel[-1] = max([5, fparam.selection.roi.max_kernel[-1]])
        fparam.selection.roi.max_kernel[-2] = max([5, fparam.selection.roi.max_kernel[-2]])
    if psftype is not None and "FD" in psftype:
        fparam.model.psf.pixel_upsampling_factor = 1

    return _dictconfig_to_params(fparam)


def save_params(
    param: RunParameters,
    path: Union[str, Path],
) -> None:
    """Write a :class:`RunParameters` instance to a YAML file.

    The file is written with a top-level ``Params:`` key so it can be
    re-loaded as a user config via :func:`load_params`.

    Parameters
    ----------
    param : RunParameters
    path : str or Path
        Destination file path (typically ending in ``.yaml``).
    """
    parent: Path = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.create({"Params": param.to_dict()})
    OmegaConf.save(cfg, path)



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
