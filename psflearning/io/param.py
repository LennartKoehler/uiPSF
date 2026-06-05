from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
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
    return RunParameters(
        datapath=cfg.get("datapath", ""),
        keyword=cfg.get("keyword", "Default."),
        savename=cfg.get("savename", ""),
        subfolder=cfg.get("subfolder", ""),
        format=cfg.get("format", ".tif"),
        stage_mov_dir=cfg.get("stage_mov_dir", "normal"),
        gain=cfg.get("gain", 0.2),
        ccd_offset=cfg.get("ccd_offset", 398.6),
        roi=RoiParams(
            roi_size=list(cfg.roi.get("roi_size", [21, 21])),
            gauss_sigma=list(cfg.roi.get("gauss_sigma", [2, 2])),
            max_kernel=list(cfg.roi.get("max_kernel", [3, 3])),
            peak_height=float(cfg.roi.get("peak_height", 0.2)),
            max_bead_number=int(cfg.roi.get("max_bead_number", 40)),
            bead_radius=float(cfg.roi.get("bead_radius", 0.0)),
        ),
        pixel_size=PixelSizeParams(
            x=float(cfg.pixel_size.get("x", 0.127)),
            y=float(cfg.pixel_size.get("y", 0.116)),
            z=float(cfg.pixel_size.get("z", 0.05)),
        ),
        FOV=FOVParams(
            y_center=int(cfg.FOV.get("y_center", 0)),
            x_center=int(cfg.FOV.get("x_center", 0)),
            radius=int(cfg.FOV.get("radius", 0)),
            z_start=int(cfg.FOV.get("z_start", 0)),
            z_end=int(cfg.FOV.get("z_end", 0)),
            z_step=int(cfg.FOV.get("z_step", 1)),
        ),
        dual=DualParams(
            mirrortype=str(cfg.dual.get("mirrortype", "up-down")),
            channel_arrange=str(cfg.dual.get("channel_arrange", "up-down")),
        ),
        multi=MultiChannelParams(
            channel_size=list(cfg.multi.get("channel_size", [])),
        ),
        fpi=FpiParams(
            modulation_period=float(cfg.fpi.get("modulation_period", 0.26)),
        ),
        LLS=LLSParams(
            skew_const=list(cfg.LLS.get("skew_const", [0, 0])),
        ),
        insitu=InsituParams(
            frame_range=list(cfg.insitu.get("frame_range", [0, 3000])),
        ),
        option=OptionParams(
            imaging=ImagingParams(
                emission_wavelength=float(cfg.option.imaging.get("emission_wavelength", 0.68)),
                NA=float(cfg.option.imaging.get("NA", 1.43)),
                RI=RefractiveIndices(
                    imm=float(cfg.option.imaging.RI.get("imm", 1.516)),
                    med=float(cfg.option.imaging.RI.get("med", 1.335)),
                    cov=float(cfg.option.imaging.RI.get("cov", 1.516)),
                ),
            ),
            insitu=InsituOptionParams(
                stage_pos=float(cfg.option.insitu.get("stage_pos", 1)),
                min_photon=float(cfg.option.insitu.get("min_photon", 0.4)),
                partition_data=bool(cfg.option.insitu.get("partition_data", True)),
                partition_size=list(cfg.option.insitu.get("partition_size", [21, 100])),
                zernike_index=list(cfg.option.insitu.get("zernike_index", [5])),
                zernike_coeff=list(cfg.option.insitu.get("zernike_coeff", [0.5])),
                z_range=float(cfg.option.insitu.get("z_range", 2.0)),
                zkorder_rank=str(cfg.option.insitu.get("zkorder_rank", "L")),
                var_stagepos=bool(cfg.option.insitu.get("var_stagepos", True)),
                repeat=int(cfg.option.insitu.get("repeat", 2)),
                backgroundROI=list(cfg.option.insitu.get("backgroundROI", [])),
            ),
            fpi=FpiOptionParams(
                link_zernikecoeff=bool(cfg.option.fpi.get("link_zernikecoeff", True)),
                phase_dm=list(cfg.option.fpi.get("phase_dm", [2, 0, -2])),
                sampleheight=int(cfg.option.fpi.get("sampleheight", 2)),
                var_sampleheight=bool(cfg.option.fpi.get("var_sampleheight", False)),
                phase_delay_dir=str(cfg.option.fpi.get("phase_delay_dir", "descend")),
            ),
            multi=MultiOptionParams(
                defocus_offset=float(cfg.option.multi.get("defocus_offset", 0)),
                defocus_delay=float(cfg.option.multi.get("defocus_delay", -0.0)),
            ),
            model=ModelParams(
                pupilsize=int(cfg.option.model.get("pupilsize", 64)),
                n_max=int(cfg.option.model.get("n_max", 8)),
                zernike_nl=list(cfg.option.model.get("zernike_nl", [])),
                blur_sigma=float(cfg.option.model.get("blur_sigma", 0.5)),
                var_blur=bool(cfg.option.model.get("var_blur", True)),
                with_apoid=bool(cfg.option.model.get("with_apoid", True)),
                const_pupilmag=bool(cfg.option.model.get("const_pupilmag", False)),
                symmetric_mag=bool(cfg.option.model.get("symmetric_mag", False)),
                with_IMM=bool(cfg.option.model.get("with_IMM", False)),
                init_pupil_file=str(cfg.option.model.get("init_pupil_file", "")),
                estimate_drift=bool(cfg.option.model.get("estimate_drift", False)),
                var_photon=bool(cfg.option.model.get("var_photon", False)),
                bin=int(cfg.option.model.get("bin", 2)),
                division=int(cfg.option.model.get("division", 40)),
            ),
        ),
        PSFtype=str(cfg.get("PSFtype", "insitu_zernike")),
        channeltype=str(cfg.get("channeltype", "single")),
        datatype=str(cfg.get("datatype", "smlm")),
        loss_weight=LossWeightParams(
            mse1=int(cfg.loss_weight.get("mse1", 1)),
            mse2=int(cfg.loss_weight.get("mse2", 1)),
            smooth=int(cfg.loss_weight.get("smooth", 0)),
            edge=float(cfg.loss_weight.get("edge", 0.01)),
            psf_min=int(cfg.loss_weight.get("psf_min", 1)),
            bg_min=int(cfg.loss_weight.get("bg_min", 1)),
            photon_min=int(cfg.loss_weight.get("photon_min", 1)),
            Inorm=int(cfg.loss_weight.get("Inorm", 0)),
            gxy_min=int(cfg.loss_weight.get("gxy_min", 10)),
        ),
        rej_threshold=RejThresholdParams(
            bias_z=float(cfg.rej_threshold.get("bias_z", 0.99)),
            mse=float(cfg.rej_threshold.get("mse", 0.8)),
            photon=float(cfg.rej_threshold.get("photon", 1.5)),
        ),
        usecuda=bool(cfg.get("usecuda", True)),
        plotall=bool(cfg.get("plotall", False)),
        ref_channel=int(cfg.get("ref_channel", 0)),
        batch_size=int(cfg.get("batch_size", 1600)),
        iteration=int(cfg.get("iteration", 200)),
        varname=str(cfg.get("varname", "")),
        filelist=list(cfg.get("filelist", [])),
        swapxy=bool(cfg.get("swapxy", False)),
    )


def _redefine(baseparam: DictConfig, userparam: DictConfig) -> DictConfig:
    """Recursively merge *userparam* into *baseparam*, overwriting leaf values."""
    for k, v in userparam.items():
        if isinstance(v, _MappingTypes) and isinstance(baseparam.get(k), _MappingTypes):
            _redefine(baseparam[k], v)
        else:
            baseparam[k] = v
    return baseparam
