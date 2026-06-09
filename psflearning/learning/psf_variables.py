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

    Replaces the ``dict(psf_class=..., ...)`` returned
    by ``psf_registry.get_psf_info()``.
    """

    psf_class: type
    loss_fun: Callable


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

    def _own_field_names(self) -> set[str]:
        return {f.name for f in fields(self) if f.name not in ("channels", "_BACKWARD_COMPAT_KEYS")}

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("channels", "_BACKWARD_COMPAT_KEYS")}
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
        if resolved in self._own_field_names():
            return getattr(self, resolved)
        if self.channels is not None and key in self.channels:
            return self.channels[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        resolved = self._resolve_key(key)
        if resolved in self._own_field_names():
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
    """Variables for ``mse_real`` and ``mse_real_zernike`` (7 elements)."""

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
