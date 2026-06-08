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

    fitted_positions: np.ndarray
    fitted_backgrounds: np.ndarray
    fitted_intensities: np.ndarray
    psf_model_image_with_bead: np.ndarray
    psf_model_image: np.ndarray
    pupil: np.ndarray
    zernike_coefficients: np.ndarray
    gaussian_blur_sigma: np.ndarray
    drift_rate: np.ndarray
    psf_model_image_reversed: np.ndarray
    model_image_offset: float
    zernike_polynomial_basis: np.ndarray
    apodization: np.ndarray
    all_roi_centers: np.ndarray
    selected_roi_centers: np.ndarray
    channels: Optional[dict[str, "PSFResult"]] = None

    _BACKWARD_COMPAT_KEYS = {
        "pos": "fitted_positions",
        "bg": "fitted_backgrounds",
        "intensity": "fitted_intensities",
        "I_model_bead": "psf_model_image_with_bead",
        "I_model": "psf_model_image",
        "zernike_coeff": "zernike_coefficients",
        "sigma": "gaussian_blur_sigma",
        "I_model_reverse": "psf_model_image_reversed",
        "offset": "model_image_offset",
        "zernike_polynomial": "zernike_polynomial_basis",
        "cor_all": "all_roi_centers",
        "cor": "selected_roi_centers",
    }

    def to_dict(self) -> dict[str, Any]:
        d = {
            "fitted_positions": self.fitted_positions,
            "fitted_backgrounds": self.fitted_backgrounds,
            "fitted_intensities": self.fitted_intensities,
            "psf_model_image_with_bead": self.psf_model_image_with_bead,
            "psf_model_image": self.psf_model_image,
            "pupil": self.pupil,
            "zernike_coefficients": self.zernike_coefficients,
            "gaussian_blur_sigma": self.gaussian_blur_sigma,
            "drift_rate": self.drift_rate,
            "psf_model_image_reversed": self.psf_model_image_reversed,
            "model_image_offset": self.model_image_offset,
            "zernike_polynomial_basis": self.zernike_polynomial_basis,
            "apodization": self.apodization,
            "all_roi_centers": self.all_roi_centers,
            "selected_roi_centers": self.selected_roi_centers,
        }
        if self.channels is not None:
            for name, ch in self.channels.items():
                d[name] = ch.to_dict()
        return d

    def _resolve_key(self, key: str) -> str:
        if key in self._BACKWARD_COMPAT_KEYS:
            return self._BACKWARD_COMPAT_KEYS[key]
        return key

    def __getitem__(self, key: str) -> Any:
        resolved = self._resolve_key(key)
        if resolved in (
            "fitted_positions", "fitted_backgrounds", "fitted_intensities",
            "psf_model_image_with_bead", "psf_model_image", "pupil",
            "zernike_coefficients", "gaussian_blur_sigma", "drift_rate",
            "psf_model_image_reversed", "model_image_offset",
            "zernike_polynomial_basis", "apodization", "all_roi_centers",
            "selected_roi_centers",
        ):
            return getattr(self, resolved)
        if self.channels is not None and key in self.channels:
            return self.channels[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        resolved = self._resolve_key(key)
        if resolved in (
            "fitted_positions", "fitted_backgrounds", "fitted_intensities",
            "psf_model_image_with_bead", "psf_model_image", "pupil",
            "zernike_coefficients", "gaussian_blur_sigma", "drift_rate",
            "psf_model_image_reversed", "model_image_offset",
            "zernike_polynomial_basis", "apodization", "all_roi_centers",
            "selected_roi_centers",
        ):
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

    mle_parameters: np.ndarray
    cramer_rao_bounds: np.ndarray
    log_likelihoods: np.ndarray
    spline_coefficients: Union[np.ndarray, list]
    spline_coefficients_per_bead: np.ndarray
    localized_positions: Union[Positions, dict]
    spline_coefficients_reversed: Union[np.ndarray, list]
    fourier_domain_positions: Optional[Union[Positions, dict]] = None

    def to_dict(self) -> dict[str, Any]:
        loc_val = self.localized_positions.to_dict() if isinstance(self.localized_positions, Positions) else self.localized_positions
        d: dict[str, Any] = {
            "mle_parameters": self.mle_parameters,
            "cramer_rao_bounds": self.cramer_rao_bounds,
            "log_likelihoods": self.log_likelihoods,
            "spline_coefficients": self.spline_coefficients,
            "spline_coefficients_per_bead": self.spline_coefficients_per_bead,
            "localized_positions": loc_val,
            "spline_coefficients_reversed": self.spline_coefficients_reversed,
        }
        if self.fourier_domain_positions is not None:
            d["fourier_domain_positions"] = (
                self.fourier_domain_positions.to_dict()
                if isinstance(self.fourier_domain_positions, Positions)
                else self.fourier_domain_positions
            )
        return d


# ── ROI data for HDF5 storage ──────────────────────────────────────────


@dataclass
class ROIsResult:
    """ROI data for HDF5 storage.

    Replaces the ``rois_dict`` assembled in ``Writer.save_result()``.
    """

    roi_centers: np.ndarray
    source_file_indices: np.ndarray
    measured_roi_images: np.ndarray
    modeled_roi_images: np.ndarray
    full_image_size: tuple

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
