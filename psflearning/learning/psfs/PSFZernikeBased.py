from __future__ import annotations

from dataclasses import dataclass, field, fields

from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter

from psflearning.learning.data_representation.PreprocessedImageDataSingleChannel import PreprocessedImageDataSingleChannel

from .PSFZernikeBase import PSFZernikeBase, PSFContext
from .IPSFModel import LearnableParameter, LearnablePSFParameters, ParameterScope, PupilField
from ..data_representation.PreprocessedImageDataInterface import PreprocessedImageDataInterface
from .PSFZernikeBase import OptimizationWeights


@dataclass
class ZernikePSFResult:
    """Structured result returned by :meth:`PSFZernikeBased.postprocess`.

    Provides dict-like access (``res["positions"]``, ``"positions" in res``,
    ``res.get("positions")``) and a ``to_dict()`` method so it can be used
    directly where a plain dict was expected before.
    """
    positions: np.ndarray
    backgrounds: np.ndarray
    intensities: np.ndarray
    psf_model_image_with_bead: np.ndarray
    psf_model_image: np.ndarray
    pupil: np.ndarray
    zernike_magnitude: np.ndarray
    zernike_phase: np.ndarray
    sigma: np.ndarray
    drift_xy: np.ndarray



    # ── Dict-like interface ─────────────────────────────────────────────

    def _own_field_names(self) -> set[str]:
        return {f.name for f in fields(self)}

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __getitem__(self, key: str) -> Any:
        if key in self._own_field_names():
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._own_field_names()

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default




@dataclass
class ZernikePSFVariables(LearnablePSFParameters):
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

    def __init__(
        self,
        positions: np.ndarray,
        backgrounds: np.ndarray,
        intensities: np.ndarray,
        zernike_magnitude: np.ndarray,
        zernike_phase: np.ndarray,
        sigma: np.ndarray,
        drift_xy: np.ndarray,
    ):
        self.positions: LearnableParameter = LearnableParameter(ParameterScope.NFIT, positions, 0)
        self.backgrounds: LearnableParameter = LearnableParameter(ParameterScope.NFIT, backgrounds, 0)
        self.intensities: LearnableParameter = LearnableParameter(ParameterScope.NFIT, intensities, 0)
        self.zernike_magnitude: LearnableParameter = LearnableParameter(ParameterScope.SHARED, zernike_magnitude, 0)
        self.zernike_phase: LearnableParameter = LearnableParameter(ParameterScope.SHARED, zernike_phase, 0)
        self.sigma: LearnableParameter = LearnableParameter(ParameterScope.SHARED, sigma, 0)
        self.drift_xy: LearnableParameter = LearnableParameter(ParameterScope.NFIT, drift_xy, 0)


    _NFIT_ATTRS = ('positions', 'backgrounds', 'intensities', 'drift_xy')
    _ATTR_ORDER = ('positions', 'backgrounds', 'intensities',
                   'zernike_magnitude', 'zernike_phase', 'sigma', 'drift_xy')

    @property
    def n_beads(self) -> int:
        """Number of emitters/beads being fitted."""
        return self.positions.numpy().shape[0]


    def toTensorList(self) -> list[tf.Variable]:
        """Return the underlying tf.Variable objects for optimizer apply_gradients.

        The optimizer mutates these in-place via .assign(), so changes
        are automatically reflected in this object's .value / .numpy().
        """
        return [getattr(self, name).variable for name in self._ATTR_ORDER]

    def toNumpy(self) -> dict:
        """Return a snapshot of all parameters as a dict of np.ndarrays."""
        return {name: getattr(self, name).numpy() for name in self._ATTR_ORDER}

    @classmethod
    def fromTensorList(cls, tensors: list[tf.Tensor]) -> ZernikePSFVariables:
        """Construct from a list of tensors/arrays (one per attribute in _ATTR_ORDER)."""
        if len(tensors) != len(cls._ATTR_ORDER):
            raise ValueError(
                f"Expected {len(cls._ATTR_ORDER)} tensors, got {len(tensors)}"
            )
        return cls(**dict(zip(cls._ATTR_ORDER, tensors)))

    def toLearnableParameterList(self) -> list[LearnableParameter]:
        """Return the LearnableParameter objects in canonical order."""
        return [getattr(self, name) for name in self._ATTR_ORDER]

    @classmethod
    def from_list(cls, variables: list) -> ZernikePSFVariables:
        return cls(
            positions=variables[0],
            backgrounds=variables[1],
            intensities=variables[2],
            zernike_magnitude=variables[3],
            zernike_phase=variables[4],
            sigma=variables[5],
            drift_xy=variables[6],
        )

    def filter_by_mask(self, mask: np.ndarray) -> ZernikePSFVariables:
        """Return a new instance with per-bead parameters filtered by *mask*.

        Shared parameters (zernike_magnitude, zernike_phase, sigma) are kept
        unchanged. Operates on optimization-space values directly, so no
        weight re-scaling is needed.
        """
        kwargs = {}
        for name in self._ATTR_ORDER:
            val = getattr(self, name).numpy()
            if name in self._NFIT_ATTRS:
                val = val[mask]
            kwargs[name] = val
        return ZernikePSFVariables(**kwargs)




class PSFZernikeBased(PSFZernikeBase):
    """
    PSF class that uses a 3D volume to describe the PSF.
    Should only be used with single-channel data.

    Stateless — all operational data is carried in :class:`PSFContext`.
    """

    def calc_initials(
        self,
        data: PreprocessedImageDataInterface,
        params,
        initial_pupil: Optional[Union[np.ndarray, list]] = None,
        start_time: float = 0.0,
    ) -> tuple[ZernikePSFVariables, PSFContext, Any]:
        """
        Provides initial values for the optimizable variables for the fitter class.

        Args:
            data: Preprocessed image data.
            params: Imaging and model parameters (OptionParams).
            initial_pupil: Optional initial pupil from a previous fit.
            start_time: Start-time stamp for progress reporting.

        Returns:
            tuple of (ZernikePSFVariables, PSFContext, start_time)
        """

        _, rois, _, _ = data.get_image_data()

        if params.model.with_IMM:
            init_positions = np.zeros((rois.shape[0], len(rois.shape)))
        else:
            init_positions = np.zeros((rois.shape[0], len(rois.shape) - 1))

        init_backgrounds = np.array(
            np.min(gaussian_filter(rois, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True)
        )
        init_intensities_local = np.sum(rois - init_backgrounds, axis=(-2, -1), keepdims=True)
        init_intensities = np.mean(init_intensities_local, axis=1, keepdims=True)

        bead_kernel = self.gen_bead_kernel(data, params)
        n_beads = rois.shape[0]
        n_z_slices = rois.shape[-3]

        psf_type = 'scalar'
        Nz = bead_kernel.shape[0]
        pupil_field = self.compute_pupil_field(data, params, psf_type, Nz=Nz)

        if params.model.const_pupilmag:
            max_magnitude_order = 0
        else:
            max_magnitude_order = 100

        init_backgrounds[init_backgrounds < 0.1] = 0.1
        bg_median = np.median(init_backgrounds)
        weight_intensity = np.lib.scimath.sqrt(np.median(init_intensities))

        optimization_weights = OptimizationWeights(
            intensity=weight_intensity * 100,
            background=bg_median,
            drift=1 / weight_intensity * 40,
            zernike_magnitude=0.5 / weight_intensity * 40,
            zernike_phase=0.5 / weight_intensity * 40,
        )

        sigma = np.ones((2,)) * params.model.blur_sigma * np.pi

        init_zernike_coeff_magnitude = np.zeros((pupil_field.zernike_polynomial_basis.shape[0], 1, 1))
        init_zernike_coeff_phase = np.zeros((pupil_field.zernike_polynomial_basis.shape[0], 1, 1))

        init_zernike_coeff_magnitude[0, 0, 0] = 1 / optimization_weights.zernike_magnitude

        init_backgrounds = (
            np.ones((n_beads, 1, 1, 1), dtype=np.float32)
            * np.median(init_backgrounds, axis=0, keepdims=True)
            / optimization_weights.background
        )
        init_drift_xy = np.zeros((n_beads, 2), dtype=np.float32)
        init_intensity_grid = np.ones((n_beads, n_z_slices, 1, 1), dtype=np.float32) * init_intensities


        if params.model.var_photon:
            init_intensity = init_intensity_grid / optimization_weights.intensity
        else:
            init_intensity = init_intensities / optimization_weights.intensity

        context = PSFContext(
            params=params,
            pupil_field=pupil_field,
            bead_kernel=bead_kernel,
            optimization_weights=optimization_weights,
            initial_pupil=initial_pupil,
            defocus=np.float32(0),
            psf_type=psf_type,
            max_magnitude_order=max_magnitude_order,
        )

        return ZernikePSFVariables(
            init_positions.astype(np.float32),
            init_backgrounds.astype(np.float32),
            init_intensity.astype(np.float32),
            init_zernike_coeff_magnitude.astype(np.float32),
            init_zernike_coeff_phase.astype(np.float32),
            sigma.astype(np.float32),
            init_drift_xy), context, start_time

    def calc_forward_images(self, variables, context: PSFContext, data=None) -> tf.Tensor:
        """
        Calculate forward images from the current guess of the variables.
        Shifting is done by Fourier transform and applying a phase ramp.

        Accepts either a ZernikePSFVariables object or a plain list of tensors
        (the latter is used by the L-BFGS-B optimizer batching loop, which needs
        raw tensors for gradient tracking).

        Args:
            variables: Learnable PSF variables (ZernikePSFVariables or list).
            context: PSF context carrying all operational state.
            data: Optional data object.
        """
        if isinstance(variables, ZernikePSFVariables):
            positions = variables.positions.value
            backgrounds = variables.backgrounds.value
            intensities = variables.intensities.value
            zernike_coeff_magnitude = variables.zernike_magnitude.value
            zernike_coeff_phase = variables.zernike_phase.value
            sigma = variables.sigma.value
            drift_xy = variables.drift_xy.value
        else:
            # Plain list: [positions, backgrounds, intensities, zernike_mag, zernike_phase, sigma, drift_xy]
            positions = variables[0]
            backgrounds = variables[1]
            intensities = variables[2]
            zernike_coeff_magnitude = variables[3]
            zernike_coeff_phase = variables[4]
            sigma = variables[5]
            drift_xy = variables[6]

        if context.initial_pupil is not None:
            pupil = context.initial_pupil
        else:
            pupil = self.compute_pupil_from_zernike(
                zernike_coeff_magnitude, zernike_coeff_phase,
                context.optimization_weights.zernike_magnitude, context.optimization_weights.zernike_phase,
                context,
            )

        positions = tf.complex(tf.reshape(positions, positions.shape + (1, 1, 1)), 0.0)

        pf = context.pupil_field
        phase_z, phase_xy = self._compute_phase(
            positions,
            pf.z_range,
            context.defocus,
            pf.frequency_x,
            pf.frequency_y,
            pf.frequency_z,
            pf.frequency_z_medium,
        )

        propagated_psf_intensity = self._render_psf(
            pupil, phase_z, sigma, context, phase_xy=phase_xy,
            use_bead_kernel=True, data=data,
        )

        if context.params.model.estimate_drift:
            drift_xy = drift_xy * context.optimization_weights.drift
            forward_images = self.applyDrift(propagated_psf_intensity, drift_xy, data, pf) * intensities * context.optimization_weights.intensity + backgrounds * context.optimization_weights.background
        else:
            forward_images = propagated_psf_intensity * intensities * context.optimization_weights.intensity + backgrounds * context.optimization_weights.background

        return forward_images

    def genpsfmodel(
        self,
        sigma: np.ndarray,
        context: PSFContext,
        Zcoeff_magnitude: tf.Tensor = None,
        Zcoeff_phase: tf.Tensor = None,
        pupil: Any = None,
        addbead: bool = False,
    ) -> tuple[tf.Tensor, Any]:
        """Generate a PSF model from Zernike coefficients or a given pupil function.

        Args:
            sigma: Blur sigma values.
            context: PSF context carrying all operational state.
            Zcoeff_magnitude: Optional Zernike magnitude coefficients.
            Zcoeff_phase: Optional Zernike phase coefficients.
            pupil: Optional pre-computed pupil (bypasses Zernike construction).
            addbead: Whether to include bead kernel in rendering.
        """
        pf = context.pupil_field
        if pupil is None:
            pupil_mag = tf.reduce_sum(pf.zernike_polynomial_basis * Zcoeff_magnitude, axis=0)
            pupil_mag = tf.math.maximum(pupil_mag, 0)
            pupil_phase = tf.reduce_sum(pf.zernike_polynomial_basis * Zcoeff_phase, axis=0)
            pupil = self.magnitude_phase_to_pupil(pupil_mag, pupil_phase, context)

        phiz = -1j * 2 * np.pi * pf.frequency_z * (pf.z_range + context.defocus)
        phase_z = tf.exp(phiz)

        psf_model_image = self._render_psf(pupil, phase_z, sigma, context, use_bead_kernel=addbead)

        return psf_model_image, pupil

    def postprocess(self, data: PreprocessedImageDataSingleChannel, variables: ZernikePSFVariables, context: PSFContext) -> ZernikePSFResult:
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.

        Accepts either a ZernikePSFVariables object or a plain list.

        Args:
            data: Preprocessed image data.
            variables: Optimized learnable variables.
            context: PSF context carrying all operational state.
        """
        pf = context.pupil_field
        positions = variables.positions.numpy()
        backgrounds = variables.backgrounds.numpy()
        intensities = variables.intensities.numpy()
        zernike_coeff_magnitude = variables.zernike_magnitude.numpy()
        zernike_coeff_phase = variables.zernike_phase.numpy()
        sigma = variables.sigma.numpy()
        drift_xy = variables.drift_xy.numpy()
        z_center = (pf.z_range.shape[-3] - 1) // 2

        zernike_coeff_magnitude = zernike_coeff_magnitude * context.optimization_weights.zernike_magnitude
        zernike_coeff_phase = zernike_coeff_phase * context.optimization_weights.zernike_phase

        bin_factor = context.params.model.bin
        positions[:, 1:] = positions[:, 1:] / bin_factor

        if context.initial_pupil is not None:
            pupil = context.initial_pupil
            psf_model_image, _ = self.genpsfmodel(sigma, context, pupil=pupil)
            psf_model_image_with_bead, _ = self.genpsfmodel(sigma, context, pupil=pupil, addbead=True)
        else:
            psf_model_image, pupil = self.genpsfmodel(
                sigma, context, Zcoeff_magnitude=zernike_coeff_magnitude, Zcoeff_phase=zernike_coeff_phase,
            )
            psf_model_image_with_bead, _ = self.genpsfmodel(
                sigma, context, Zcoeff_magnitude=zernike_coeff_magnitude, Zcoeff_phase=zernike_coeff_phase, addbead=True,
            )

        assert data is not None
        _, _, centers, _ = data.get_image_data()

        if positions.shape[1] > 3:
            global_positions = np.swapaxes(
                np.vstack((
                    positions[:, 0] + z_center,
                    positions[:, 1],
                    centers[:, -2] - positions[:, -2],
                    centers[:, -1] - positions[:, -1],
                )), 1, 0,
            )
        else:
            global_positions = np.swapaxes(
                np.vstack((
                    positions[:, 0] + z_center,
                    centers[:, -2] - positions[:, -2],
                    centers[:, -1] - positions[:, -1],
                )), 1, 0,
            )

        return ZernikePSFResult(
            positions=global_positions.astype(np.float32),
            backgrounds=backgrounds * context.optimization_weights.background,
            intensities=intensities * context.optimization_weights.intensity,
            psf_model_image_with_bead=psf_model_image_with_bead,
            psf_model_image=psf_model_image,
            pupil=np.complex64(pupil),
            zernike_magnitude=zernike_coeff_magnitude,
            zernike_phase=zernike_coeff_phase,
            sigma=sigma,
            drift_xy=drift_xy * context.optimization_weights.drift,
        )
