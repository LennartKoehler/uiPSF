from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter

from .PSFZernikeBase import PSFZernikeBase
from .PSFInterface import LearnableParameter, LearnablePSFParameters, ParameterScope
from ..data_representation.PreprocessedImageDataInterface import PreprocessedImageDataInterface
from ..loss_functions import mse_real_zernike
from ..psf_variables import OptimizationWeights, PSFResult


@dataclass
class ZernikePSFResult:
    """Structured result returned by :meth:`PSFZernikeBased.postprocess`.

    Replaces the former plain-list return value so that consumers can
    access fields by name instead of magic indices.
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
    psf_model_image_reversed: np.ndarray
    _variables: Optional[ZernikePSFVariables] = field(default=None, repr=False)

    def filter_by_mask(self, mask: np.ndarray) -> ZernikePSFVariables:
        assert self._variables is not None
        return self._variables.filter_by_mask(mask)

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


    _ATTR_ORDER = ('positions', 'backgrounds', 'intensities',
                   'zernike_magnitude', 'zernike_phase', 'sigma', 'drift_xy')

    _NFIT_ATTRS = ('positions', 'backgrounds', 'intensities', 'drift_xy')

    @property
    def n_beads(self) -> int:
        """Number of emitters/beads being fitted."""
        return self.positions.numpy().shape[0]

    def filter_by_mask(self, mask: np.ndarray) -> ZernikePSFVariables:
        """Return a new instance with per-bead parameters filtered by *mask*.

        Shared parameters (zernike_magnitude, zernike_phase, sigma) are kept
        unchanged.
        """
        kwargs = {}
        for name in self._ATTR_ORDER:
            val = getattr(self, name).numpy()
            if name in self._NFIT_ATTRS:
                val = val[mask]
            kwargs[name] = val
        return ZernikePSFVariables(**kwargs)

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





class PSFZernikeBased(PSFZernikeBase):
    """
    PSF class that uses a 3D volume to describe the PSF.
    Should only be used with single-channel data.
    """

    def __init__(self, options: Any = None) -> None:
        self.parameters = None
        self.data: Optional[PreprocessedImageDataInterface] = None
        self.Zphase = None
        self.zT = None
        self.bead_kernel = None
        self.options = options
        self.initial_pupil: Optional[Union[np.ndarray, list]] = None
        self.initial_psf_image: Optional[np.ndarray] = None
        self.initial_zernike_coefficients: Optional[np.ndarray] = None
        self.initial_interference_amplitude: Optional[list] = None
        self.z_offset: Optional[np.ndarray] = None
        self.defocus = np.float32(0)
        self.default_loss_func = mse_real_zernike
        self.psftype = 'scalar'

    def calc_initials(self, data: PreprocessedImageDataInterface, start_time: float = 0.0) -> tuple[ZernikePSFVariables, Any]:
        """
        Provides initial values for the optimizable variables for the fitter class.
        """
        self.data = data
        _, rois, _, _ = self.data.get_image_data()

        options = self.options
        if options.model.with_IMM:
            init_positions = np.zeros((rois.shape[0], len(rois.shape)))
        else:
            init_positions = np.zeros((rois.shape[0], len(rois.shape) - 1))

        init_backgrounds = np.array(
            np.min(gaussian_filter(rois, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True)
        )
        init_intensities_local = np.sum(rois - init_backgrounds, axis=(-2, -1), keepdims=True)
        init_intensities = np.mean(init_intensities_local, axis=1, keepdims=True)

        self.gen_bead_kernel()
        n_beads = rois.shape[0]
        n_z_slices = rois.shape[-3]

        if self.psftype == 'vector':
            self.calpupilfield('vector')
        else:
            self.calpupilfield('scalar')

        if options.model.const_pupilmag:
            self.n_max_mag = 0
        else:
            self.n_max_mag = 100

        init_backgrounds[init_backgrounds < 0.1] = 0.1
        bg_median = np.median(init_backgrounds)
        weight_intensity = np.lib.scimath.sqrt(np.median(init_intensities))

        self.weight = OptimizationWeights(
            intensity=weight_intensity * 100,
            background=bg_median,
            drift=1 / weight_intensity * 40,
            zernike_magnitude=0.5 / weight_intensity * 40,
            zernike_phase=0.5 / weight_intensity * 40,
        )

        sigma = np.ones((2,)) * self.options.model.blur_sigma * np.pi
        self.init_sigma = sigma

        init_zernike_coeff_magnitude = np.zeros((self.zernike_polynomial_basis.shape[0], 1, 1))
        init_zernike_coeff_phase = np.zeros((self.zernike_polynomial_basis.shape[0], 1, 1))

        init_zernike_coeff_magnitude[0, 0, 0] = 1 / self.weight.zernike_magnitude

        init_backgrounds = (
            np.ones((n_beads, 1, 1, 1), dtype=np.float32)
            * np.median(init_backgrounds, axis=0, keepdims=True)
            / self.weight.background
        )
        init_drift_xy = np.zeros((n_beads, 2), dtype=np.float32)
        init_intensity_grid = np.ones((n_beads, n_z_slices, 1, 1), dtype=np.float32) * init_intensities


        if options.model.var_photon:
            init_intensity = init_intensity_grid / self.weight.intensity
        else:
            init_intensity = init_intensities / self.weight.intensity

        return ZernikePSFVariables(
            init_positions.astype(np.float32),
            init_backgrounds.astype(np.float32),
            init_intensity.astype(np.float32),
            init_zernike_coeff_magnitude.astype(np.float32),
            init_zernike_coeff_phase.astype(np.float32),
            sigma.astype(np.float32),
            init_drift_xy),start_time

    def calc_forward_images(self, variables) -> tf.Tensor:
        """
        Calculate forward images from the current guess of the variables.
        Shifting is done by Fourier transform and applying a phase ramp.

        Accepts either a ZernikePSFVariables object or a plain list of tensors
        (the latter is used by the L-BFGS-B optimizer batching loop, which needs
        raw tensors for gradient tracking).
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

        if self.initial_pupil is not None:
            pupil = self.initial_pupil
        else:
            pupil = self.compute_pupil_from_zernike(
                zernike_coeff_magnitude, zernike_coeff_phase,
                self.weight.zernike_magnitude, self.weight.zernike_phase,
            )

        positions = tf.complex(tf.reshape(positions, positions.shape + (1, 1, 1)), 0.0)

        phase_z, phase_xy = self._compute_phase(
            positions,
            self.Zrange,
            self.defocus,
            self.kx,
            self.ky,
            self.kz,
            self.kz_med,
        )

        propagated_psf_intensity = self.propagate_pupil(pupil, phase_z, phase_xy)

        bin_factor = self.options.model.bin
        if not self.options.model.var_blur:
            sigma = self.init_sigma

        blurred_psf_intensity = self.apply_blur_3d(propagated_psf_intensity, sigma, use_bead_kernel=True)
        binned_psf_intensity = self.bin_image_3d(blurred_psf_intensity, bin_factor)
        trimmed_psf_intensity = binned_psf_intensity[..., 0]
        trimmed_psf_intensity = self.trim_z_padding(trimmed_psf_intensity)

        if self.options.model.estimate_drift:
            drift_xy = drift_xy * self.weight.drift
            drift_corrected_psf = self.applyDrift(trimmed_psf_intensity, drift_xy)
            forward_images = drift_corrected_psf * intensities * self.weight.intensity + backgrounds * self.weight.background
        else:
            forward_images = trimmed_psf_intensity * intensities * self.weight.intensity + backgrounds * self.weight.background

        return forward_images

    def genpsfmodel(
        self,
        sigma: np.ndarray,
        Zcoeff_magnitude: tf.Tensor = None,
        Zcoeff_phase: tf.Tensor = None,
        pupil: Any = None,
        addbead: bool = False,
    ) -> tuple[tf.Tensor, Any]:
        """Generate a PSF model from Zernike coefficients or a given pupil function."""
        if pupil is None:
            pupil_mag = tf.reduce_sum(self.zernike_polynomial_basis * Zcoeff_magnitude, axis=0)
            pupil_mag = tf.math.maximum(pupil_mag, 0)
            pupil_phase = tf.reduce_sum(self.zernike_polynomial_basis * Zcoeff_phase, axis=0)
            pupil = self.magnitude_phase_to_pupil(pupil_mag, pupil_phase)

        phiz = -1j * 2 * np.pi * self.kz * (self.Zrange + self.defocus)
        phase_z = tf.exp(phiz)
        propagated_psf_intensity = self.propagate_pupil(pupil, phase_z)

        bin_factor = self.options.model.bin
        blurred_psf_intensity = self.apply_blur_3d(propagated_psf_intensity, sigma, use_bead_kernel=addbead)

        if len(blurred_psf_intensity.shape) == 5:
            psf_model_image = self.bin_image_3d(blurred_psf_intensity, bin_factor)
        else:
            kernel = np.ones((bin_factor, bin_factor, 1, 1), dtype=np.float32)
            psf_model_image = tf.nn.convolution(
                blurred_psf_intensity, kernel,
                strides=(1, bin_factor, bin_factor, 1),
                padding='SAME', data_format='NHWC',
            )
        psf_model_image = psf_model_image[..., 0]

        return psf_model_image, pupil

    def postprocess(self, variables: ZernikePSFVariables) -> ZernikePSFResult:
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.

        Accepts either a ZernikePSFVariables object or a plain list.
        """
        positions = variables.positions.numpy()
        backgrounds = variables.backgrounds.numpy()
        intensities = variables.intensities.numpy()
        zernike_coeff_magnitude = variables.zernike_magnitude.numpy()
        zernike_coeff_phase = variables.zernike_phase.numpy()
        sigma = variables.sigma.numpy()
        drift_xy = variables.drift_xy.numpy()
        z_center = (self.Zrange.shape[-3] - 1) // 2

        zernike_coeff_magnitude = zernike_coeff_magnitude * self.weight.zernike_magnitude
        zernike_coeff_phase = zernike_coeff_phase * self.weight.zernike_phase

        bin_factor = self.options.model.bin
        positions[:, 1:] = positions[:, 1:] / bin_factor

        if self.initial_pupil is not None:
            pupil = self.initial_pupil
            psf_model_image, _ = self.genpsfmodel(sigma, pupil=pupil)
            psf_model_image_with_bead, _ = self.genpsfmodel(sigma, pupil=pupil, addbead=True)
        else:
            psf_model_image, pupil = self.genpsfmodel(
                sigma, Zcoeff_magnitude=zernike_coeff_magnitude, Zcoeff_phase=zernike_coeff_phase,
            )
            psf_model_image_with_bead, _ = self.genpsfmodel(
                sigma, Zcoeff_magnitude=zernike_coeff_magnitude, Zcoeff_phase=zernike_coeff_phase, addbead=True,
            )

        assert self.data is not None
        _, _, centers, _ = self.data.get_image_data()

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
            backgrounds=backgrounds * self.weight.background,
            intensities=intensities * self.weight.intensity,
            psf_model_image_with_bead=psf_model_image_with_bead,
            psf_model_image=psf_model_image,
            pupil=np.complex64(pupil),
            zernike_magnitude=zernike_coeff_magnitude,
            zernike_phase=zernike_coeff_phase,
            sigma=sigma,
            drift_xy=drift_xy * self.weight.drift,
            psf_model_image_reversed=np.flip(psf_model_image, axis=-3),
            _variables=variables,
        )

    def res2dict(self, res: ZernikePSFResult) -> PSFResult:
        assert self.data is not None
        return PSFResult(
            fitted_positions=res.positions,
            fitted_backgrounds=np.squeeze(res.backgrounds),
            fitted_intensities=np.squeeze(res.intensities),
            psf_model_image_with_bead=res.psf_model_image_with_bead,
            psf_model_image=res.psf_model_image,
            pupil=res.pupil,
            zernike_coefficients=np.array([np.squeeze(res.zernike_magnitude), np.squeeze(res.zernike_phase)]),
            gaussian_blur_sigma=np.squeeze(res.sigma) / np.pi,
            drift_rate=res.drift_xy,
            psf_model_image_reversed=res.psf_model_image_reversed,
            model_image_offset=np.min(res.psf_model_image),
            zernike_polynomial_basis=self.zernike_polynomial_basis,
            apodization=self.apodization,
            all_roi_centers=self.data.roi_centers_all,
            selected_roi_centers=self.data.roi_centers,
        )

