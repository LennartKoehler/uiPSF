"""
Dataclasses for PSF model variables and result containers.

This module provides clear, typed structures replacing opaque dicts and
index-based lists used across different PSF model types and pipeline
stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, List, Optional, Union

import numpy as np
import tensorflow as tf


# ── Positions ───────────────────────────────────────────────────────────


@dataclass
class Positions:
    """Localized emitter positions.

    Replaces the ``dict(x=..., y=..., z=...)`` pattern used in
    ``loclib.py`` and ``Localizer.py``.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    zast: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, np.ndarray]:
        d: dict[str, np.ndarray] = {"x": self.x, "y": self.y, "z": self.z}
        if self.zast is not None:
            d["zast"] = self.zast
        return d


# ── PSF registry ───────────────────────────────────────────────────────


@dataclass
class PSFInfo:
    """PSF type registry entry.

    Replaces the ``dict(psf_class=..., psf_class_multi=..., ...)`` returned
    by ``psf_registry.get_psf_info()``.
    """

    psf_class: type
    psf_class_multi: Optional[type]
    loss_fun: Callable
    loss_fun_multi: Optional[Callable]


# ── Optimization weights ───────────────────────────────────────────────


@dataclass
class OptimizationWeights:
    """Scaling weights for optimization variables in ``PSFZernikeBased``.

    Replaces the ``self.weight`` dict with keys ``"intensity"``,
    ``"background"``, ``"drift"``, ``"zernikeMagnitude"``,
    ``"zernikePhase"``.
    """

    intensity: float
    background: float
    drift: float
    zernike_magnitude: float
    zernike_phase: float


# ── PSF fitting result ─────────────────────────────────────────────────


@dataclass
class PSFResult:
    """Structured PSF fitting result (single-channel).

    Replaces the ``res_dict`` returned by ``PSFZernikeBased.res2dict()``.
    """

    pos: np.ndarray
    bg: np.ndarray
    intensity: np.ndarray
    I_model_bead: np.ndarray
    I_model: np.ndarray
    pupil: np.ndarray
    zernike_coeff: np.ndarray
    sigma: np.ndarray
    drift_rate: np.ndarray
    I_model_reverse: np.ndarray
    offset: float
    zernike_polynomial: np.ndarray
    apodization: np.ndarray
    cor_all: np.ndarray
    cor: np.ndarray
    channels: Optional[dict[str, "PSFResult"]] = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "pos": self.pos,
            "bg": self.bg,
            "intensity": self.intensity,
            "I_model_bead": self.I_model_bead,
            "I_model": self.I_model,
            "pupil": self.pupil,
            "zernike_coeff": self.zernike_coeff,
            "sigma": self.sigma,
            "drift_rate": self.drift_rate,
            "I_model_reverse": self.I_model_reverse,
            "offset": self.offset,
            "zernike_polynomial": self.zernike_polynomial,
            "apodization": self.apodization,
            "cor_all": self.cor_all,
            "cor": self.cor,
        }
        if self.channels is not None:
            for name, ch in self.channels.items():
                d[name] = ch.to_dict()
        return d

    def __getitem__(self, key: str) -> Any:
        if key in ("pos", "bg", "intensity", "I_model_bead", "I_model",
                     "pupil", "zernike_coeff", "sigma", "drift_rate",
                     "I_model_reverse", "offset", "zernike_polynomial",
                     "apodization", "cor_all", "cor"):
            return getattr(self, key if key != "zernike_coeff" else "zernike_coeff")
        if self.channels is not None and key in self.channels:
            return self.channels[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        if key in ("pos", "bg", "intensity", "I_model_bead", "I_model",
                     "pupil", "zernike_coeff", "sigma", "drift_rate",
                     "I_model_reverse", "offset", "zernike_polynomial",
                     "apodization", "cor_all", "cor"):
            return True
        if self.channels is not None and key in self.channels:
            return True
        return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


# ── Localization result for HDF5 storage ────────────────────────────────


@dataclass
class LocResResult:
    """Localization result for HDF5 storage.

    Replaces the ``locres_dict`` assembled in ``Writer._build_locres_dict()``.
    """

    P: np.ndarray
    CRLB: np.ndarray
    LL: np.ndarray
    coeff: Union[np.ndarray, list]
    coeff_bead: np.ndarray
    loc: Union[Positions, dict]
    coeff_reverse: Union[np.ndarray, list]
    loc_FD: Optional[Union[Positions, dict]] = None

    def to_dict(self) -> dict[str, Any]:
        loc_val = self.loc.to_dict() if isinstance(self.loc, Positions) else self.loc
        d: dict[str, Any] = {
            "P": self.P,
            "CRLB": self.CRLB,
            "LL": self.LL,
            "coeff": self.coeff,
            "coeff_bead": self.coeff_bead,
            "loc": loc_val,
            "coeff_reverse": self.coeff_reverse,
        }
        if self.loc_FD is not None:
            d["loc_FD"] = (
                self.loc_FD.to_dict()
                if isinstance(self.loc_FD, Positions)
                else self.loc_FD
            )
        return d


# ── ROI data for HDF5 storage ──────────────────────────────────────────


@dataclass
class ROIsResult:
    """ROI data for HDF5 storage.

    Replaces the ``rois_dict`` assembled in ``Writer.save_result()``.
    """

    cor: np.ndarray
    fileID: np.ndarray
    psf_data: np.ndarray
    psf_fit: np.ndarray
    image_size: tuple

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Report result ──────────────────────────────────────────────────────


@dataclass
class ReportResult:
    """Saved report file paths.

    Replaces the ``saved`` dict returned by ``Plotter.generate_report()``.
    """

    psf_vs_data: List[str] = field(default_factory=list)
    localization: List[str] = field(default_factory=list)
    zernike: Optional[List[str]] = None
    pupil: Optional[List[str]] = None
    learned_params: List[str] = field(default_factory=list)
    coordinates: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, List[str]]:
        d: dict[str, List[str]] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                d[f.name] = val
        return d

    def items(self):
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                yield f.name, val


# ── Loss function variable containers ──────────────────────────────────
#
# These replace index-based ``variables[i]`` access in loss functions.
# Each class corresponds to a specific variable layout.  The
# ``from_list()`` classmethod unpacks the flat tensor list that the
# L-BFGS-B optimizer batching loop passes to the objective.


@dataclass
class ZernikeLossVariables:
    """Variables for ``mse_real`` (7 elements)."""

    positions: Any
    backgrounds: Any
    intensities: Any
    zernike_magnitude: Any
    zernike_phase: Any
    sigma: Any
    drift_xy: Any

    @classmethod
    def from_list(cls, variables: list) -> ZernikeLossVariables:
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            zernike_magnitude=variables[3],
            zernike_phase=variables[4],
            sigma=variables[5],
            drift_xy=variables[6],
        )


@dataclass
class Zernike4PiLossVariables:
    """Variables for ``mse_real_4pi`` / ``mse_zernike_4pi`` (11 elements)."""

    positions: Any
    backgrounds: Any
    intensities: Any
    intensity_phase: Any
    zernike_magnitude_1: Any
    zernike_phase_1: Any
    zernike_magnitude_2: Any
    zernike_phase_2: Any
    alpha: Any
    wavelength: Any
    drift_xy: Any

    @classmethod
    def from_list(cls, variables: list) -> Zernike4PiLossVariables:
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            intensity_phase=variables[3],
            zernike_magnitude_1=variables[4],
            zernike_phase_1=variables[5],
            zernike_magnitude_2=variables[6],
            zernike_phase_2=variables[7],
            alpha=variables[8],
            wavelength=variables[9],
            drift_xy=variables[10],
        )


@dataclass
class PupilLossVariables:
    """Variables for ``mse_real_pupil`` (7 elements)."""

    positions: Any
    backgrounds: Any
    intensities: Any
    pupil_real: Any
    pupil_imag: Any
    sigma: Any
    drift_xy: Any

    @classmethod
    def from_list(cls, variables: list) -> PupilLossVariables:
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            pupil_real=variables[3],
            pupil_imag=variables[4],
            sigma=variables[5],
            drift_xy=variables[6],
        )


@dataclass
class Pupil4PiLossVariables:
    """Variables for ``mse_pupil_4pi`` (10 elements)."""

    positions: Any
    backgrounds: Any
    intensities: Any
    pupil_real_1: Any
    pupil_imag_1: Any
    pupil_real_2: Any
    pupil_imag_2: Any
    alpha: Any
    wavelength: Any
    drift_xy: Any

    @classmethod
    def from_list(cls, variables: list) -> Pupil4PiLossVariables:
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            pupil_real_1=variables[3],
            pupil_imag_1=variables[4],
            pupil_real_2=variables[5],
            pupil_imag_2=variables[6],
            alpha=variables[7],
            wavelength=variables[8],
            drift_xy=variables[9],
        )


@dataclass
class Zernike4PiSMLMLossVariables:
    """Variables for ``mse_zernike_4pi_smlm`` (12 elements)."""

    positions: Any
    backgrounds: Any
    intensities: Any
    intensity_phase: Any
    stage_position: Any
    sample_height: Any
    zernike_magnitude_1: Any
    zernike_phase_1: Any
    zernike_magnitude_2: Any
    zernike_phase_2: Any
    alpha: Any
    drift_xy: Any

    @classmethod
    def from_list(cls, variables: list) -> Zernike4PiSMLMLossVariables:
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            intensity_phase=variables[3],
            stage_position=variables[4],
            sample_height=variables[5],
            zernike_magnitude_1=variables[6],
            zernike_phase_1=variables[7],
            zernike_magnitude_2=variables[8],
            zernike_phase_2=variables[9],
            alpha=variables[10],
            drift_xy=variables[11],
        )


@dataclass
class ZernikeFDLossVariables:
    """Variables for ``mse_real_zernike_FD`` / ``mse_real_zernike_IMM`` (6 elements)."""

    positions: Any
    backgrounds: Any
    intensities: Any
    zernike_map: Any
    sigma: Any
    drift_xy: Any

    @classmethod
    def from_list(cls, variables: list) -> ZernikeFDLossVariables:
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            zernike_map=variables[3],
            sigma=variables[4],
            drift_xy=variables[5],
        )


@dataclass
class ZernikeFDSMLMLossVariables:
    """Variables for ``mse_real_zernike_FD_smlm`` (7 elements)."""

    positions: Any
    backgrounds: Any
    intensities: Any
    zernike_map: Any
    sigma: Any
    stage_position: Any
    drift_xy: Any

    @classmethod
    def from_list(cls, variables: list) -> ZernikeFDSMLMLossVariables:
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            zernike_map=variables[3],
            sigma=variables[4],
            stage_position=variables[5],
            drift_xy=variables[6],
        )


@dataclass
class ZernikeSMLMLossVariables:
    """Variables for ``mse_real_zernike_smlm`` (8 elements)."""

    positions: Any
    backgrounds: Any
    intensities: Any
    zernike_magnitude: Any
    zernike_phase: Any
    stage_position: Any
    sigma: Any
    drift_xy: Any

    @classmethod
    def from_list(cls, variables: list) -> ZernikeSMLMLossVariables:
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            zernike_magnitude=variables[3],
            zernike_phase=variables[4],
            stage_position=variables[5],
            sigma=variables[6],
            drift_xy=variables[7],
        )


@dataclass
class PupilSMLMLossVariables:
    """Variables for ``mse_real_pupil_smlm`` (8 elements)."""

    positions: Any
    backgrounds: Any
    intensities: Any
    pupil_real: Any
    pupil_imag: Any
    stage_position: Any
    sigma: Any
    drift_xy: Any

    @classmethod
    def from_list(cls, variables: list) -> PupilSMLMLossVariables:
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            pupil_real=variables[3],
            pupil_imag=variables[4],
            stage_position=variables[5],
            sigma=variables[6],
            drift_xy=variables[7],
        )
