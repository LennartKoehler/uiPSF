"""
Dataclasses for PSF model variables.

This module provides clear, typed structures for the optimization variables
used across different PSF model types, replacing opaque list indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


@dataclass
class ZernikePSFVariables:
    """
    Variables for Zernike-based PSF models (scalar or vector).

    Attributes:
        positions: Emitter positions in ROI coordinates [n_beads, 2 or 3 or 4].
            Format: [z, y, x] for standard, or [z, y, x, ...] for IMM.
        backgrounds: Background level per emitter [n_beads, 1, 1, 1].
        intensities: Emitter intensity per bead [n_beads, ...].
            May vary per z-slice if var_photon is enabled.
        zernike_magnitude: Zernike polynomial coefficients for pupil magnitude
            [n_zernike, 1, 1]. Shared across all emitters.
        zernike_phase: Zernike polynomial coefficients for pupil phase
            [n_zernike, 1, 1]. Shared across all emitters.
        sigma: Gaussian blur parameters [2] = [sigma_y, sigma_x].
            Shared across all emitters.
        drift_xy: Lateral drift between z-slices [n_beads, 2].
    """

    positions: np.ndarray
    backgrounds: np.ndarray
    intensities: np.ndarray
    zernike_magnitude: np.ndarray
    zernike_phase: np.ndarray
    sigma: np.ndarray
    drift_xy: np.ndarray

    def to_list(self) -> list[np.ndarray]:
        """Convert to list format for optimizer compatibility."""
        return [
            self.positions,
            self.backgrounds,
            self.intensities,
            self.zernike_magnitude,
            self.zernike_phase,
            self.sigma,
            self.drift_xy,
        ]

    @classmethod
    def from_list(cls, variables: list[np.ndarray]) -> ZernikePSFVariables:
        """Create from list format."""
        if len(variables) != 7:
            raise ValueError(
                f"Expected 7 variables for ZernikePSFVariables, got {len(variables)}"
            )
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            zernike_magnitude=variables[3],
            zernike_phase=variables[4],
            sigma=variables[5],
            drift_xy=variables[6],
        )

    @property
    def n_beads(self) -> int:
        """Number of emitters/beads being fitted."""
        return self.positions.shape[0]


@dataclass
class FieldDependentZernikePSFVariables:
    """
    Variables for field-dependent Zernike PSF models.

    In field-dependent models, Zernike coefficients vary spatially across
    the FOV instead of being shared parameters.

    Attributes:
        positions: Emitter positions [n_beads, 2 or 3].
        backgrounds: Background level [n_beads, 1, 1, 1].
        intensities: Emitter intensity [n_beads, ...].
        zernike_map: Spatially varying Zernike coefficients [n_beads, n_zernike].
            Each bead has its own Zernike coefficients.
        sigma: Gaussian blur [2].
        drift_xy: Lateral drift [n_beads, 2].
    """

    positions: np.ndarray
    backgrounds: np.ndarray
    intensities: np.ndarray
    zernike_map: np.ndarray
    sigma: np.ndarray
    drift_xy: np.ndarray

    def to_list(self) -> list[np.ndarray]:
        """Convert to list format for optimizer compatibility."""
        return [
            self.positions,
            self.backgrounds,
            self.intensities,
            self.zernike_map,
            self.sigma,
            self.drift_xy,
        ]

    @classmethod
    def from_list(cls, variables: list[np.ndarray]) -> FieldDependentZernikePSFVariables:
        """Create from list format."""
        if len(variables) != 6:
            raise ValueError(
                f"Expected 6 variables for FieldDependentZernikePSFVariables, got {len(variables)}"
            )
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            zernike_map=variables[3],
            sigma=variables[4],
            drift_xy=variables[5],
        )



