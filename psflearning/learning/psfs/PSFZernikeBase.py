from __future__ import annotations

from abc import ABCMeta
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm

from .IPSFModel import IPSFModel, PupilField

from .. import utilities as im
from psflearning.io.param import OptionParams

@dataclass
class OptimizationWeights:
    """Scaling weights for optimization variables in ``PSFZernikeBased``.

    Replaces the ``self.optimization_weights`` dict with keys ``"intensity"``,
    ``"background"``, ``"drift"``, ``"zernikeMagnitude"``,
    ``"zernikePhase"``.
    """

    intensity: float
    background: float
    drift: float
    zernike_magnitude: float
    zernike_phase: float


@dataclass
class PSFContext:
    """All runtime state needed for Zernike-based PSF forward modeling
    and postprocessing.

    Created by :meth:`PSFZernikeBased.calc_initials` and passed to
    :meth:`calc_forward_images`, :meth:`postprocess`, and
    :meth:`genpsfmodel`.  Carries configuration, precomputed optical
    fields, and optimization weights — nothing lives on the PSF instance.
    """

    params: OptionParams
    pupil_field: PupilField
    bead_kernel: Optional[Any] = None
    optimization_weights: Optional[OptimizationWeights] = None
    initial_pupil: Optional[Union[np.ndarray, list]] = None
    defocus: Any = field(default_factory=lambda: np.float32(0))
    psf_type: str = 'scalar'
    max_magnitude_order: int = 100



class PSFZernikeBase(IPSFModel, metaclass=ABCMeta):
    """
    Intermediate base class for Zernike-based PSF models.
    Provides shared helper methods for pupil computation, propagation, blurring, and binning.

    Subclasses must implement:
        - calc_initials
        - calc_forward_images
        - postprocess

    This class is stateless — all operational data is passed via ``context: PSFContext``.
    """

    __metaclass__ = ABCMeta


    def magnitude_phase_to_pupil(
        self,
        pupil_mag: tf.Tensor,
        pupil_phase: tf.Tensor,
        context: PSFContext,
    ) -> tf.Tensor:
        """
        Construct a complex pupil function from magnitude and phase.

        Args:
            pupil_mag: Magnitude of the pupil function.
            pupil_phase: Phase of the pupil function in radians.
            context: PSF context providing pupil_mask and apodization.

        Returns:
            Complex pupil function with aperture and apodization applied.
        """
        return tf.complex(
            pupil_mag * tfm.cos(pupil_phase),
            pupil_mag * tfm.sin(pupil_phase),
        ) * context.pupil_field.pupil_mask * context.pupil_field.apodization

    def _compute_phase(
        self,
        pos: tf.Tensor,
        z_range,
        defocus,
        frequency_x,
        frequency_y,
        frequency_z,
        frequency_z_medium):

        phiz = -1j * 2 * np.pi * frequency_z * (pos[:, 0] + z_range + defocus)
        if int(pos.shape[1]) > 3:
            phixy = 1j * 2 * np.pi * frequency_y * pos[:, 2] + 1j * 2 * np.pi * frequency_x * pos[:, 3]
            phiz = 1j * 2 * np.pi * (frequency_z_medium * pos[:, 1] - frequency_z * (pos[:, 0] + z_range))
        else:
            phixy = 1j * 2 * np.pi * frequency_y * pos[:, 1] + 1j * 2 * np.pi * frequency_x * pos[:, 2]

        phase_z = tf.exp(phiz)
        phase_xy = tf.exp(phixy)
        return phase_z, phase_xy

    def compute_pupil_from_zernike(
        self,
        Zcoeff_mag: tf.Tensor,
        Zcoeff_phase: tf.Tensor,
        weight_mag: float,
        weight_phase: float,
        context: PSFContext,
        noll_index: np.ndarray | None = None,
    ) -> tf.Tensor:
        """
        Compute pupil function from Zernike coefficients.

        Args:
            Zcoeff_mag: Zernike magnitude coefficients (shape: [N, ...] or [n_k, 1, 1]).
            Zcoeff_phase: Zernike phase coefficients (shape: [N, ...] or [n_k, 1, 1]).
            weight_mag: Weight multiplier for magnitude.
            weight_phase: Weight multiplier for phase.
            context: PSF context providing params, zernike_polynomial_basis, spherical_noll_indices.
            noll_index: Optional Noll indices to select specific Zernike modes.

        Returns:
            Complex pupil function.
        """
        pf = context.pupil_field
        c1 = pf.spherical_noll_indices
        n_max = context.max_magnitude_order
        Nk = np.min(((n_max + 1) * (n_max + 2) // 2, pf.zernike_polynomial_basis.shape[0]))
        mask = c1 < Nk
        c1 = c1[mask]

        if noll_index is not None:
            pupil_mag = tf.reduce_sum(
                pf.zernike_polynomial_basis[c1] * tf.gather(Zcoeff_mag, indices=c1) * weight_mag, axis=0
            )
        elif context.params.model.symmetric_mag:
            pupil_mag = tf.reduce_sum(
                pf.zernike_polynomial_basis[c1] * tf.gather(Zcoeff_mag, indices=c1) * weight_mag, axis=0
            )
        else:
            pupil_mag = tf.reduce_sum(
                pf.zernike_polynomial_basis[0:Nk] * Zcoeff_mag[0:Nk] * weight_mag, axis=0
            )
        pupil_mag = tfm.maximum(pupil_mag, 0)
        pupil_phase = tf.reduce_sum(pf.zernike_polynomial_basis[3:] * Zcoeff_phase[3:] * weight_phase, axis=0)

        return self.magnitude_phase_to_pupil(pupil_mag, pupil_phase, context)

    def propagate_pupil(
        self,
        pupil: tf.Tensor,
        phase_z: tf.Tensor,
        context: PSFContext,
        phase_xy: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """
        Propagate pupil function using CZT, handling scalar vs vector modes.

        Args:
            pupil: Complex pupil function.
            phase_z: Z-propagation phase (complex exponential).
            context: PSF context providing psf_type, dipole_field, czt_parameters, normalization_factor.
            phase_xy: Optional XY phase factor for field-dependent shifts.

        Returns:
            Complex PSF field (after CZT propagation).
        """
        pf = context.pupil_field
        if context.psf_type == "vector":
            I_res = tf.constant(0.0, dtype=tf.complex64)
            for h in pf.dipole_field:
                PupilFunction = pupil * phase_z * h
                if phase_xy is not None:
                    PupilFunction = PupilFunction * phase_xy
                propagated_psf_amplitude = im.cztfunc1(PupilFunction, pf.czt_parameters)
                I_res += propagated_psf_amplitude * tfm.conj(propagated_psf_amplitude) * pf.normalization_factor
            return I_res
        else:
            PupilFunction = pupil * phase_z
            if phase_xy is not None:
                PupilFunction = PupilFunction * phase_xy
            propagated_psf_amplitude = im.cztfunc1(PupilFunction, pf.czt_parameters)
            return propagated_psf_amplitude * tfm.conj(propagated_psf_amplitude) * pf.normalization_factor

    def compute_gaussian_filter(self, sigma: tf.Tensor | np.ndarray, context: PSFContext) -> tf.Tensor:
        """
        Compute a 2D Gaussian blur filter in frequency space.

        Args:
            sigma: Blur sigma values (shape: [2]) with [sigma_y, sigma_x].
            context: PSF context providing frequency_squared_x/y.

        Returns:
            Complex frequency-domain filter (normalized).
        """
        pf = context.pupil_field
        filter2 = tf.exp(
            -2 * sigma[1] * sigma[1] * pf.frequency_squared_x
            - 2 * sigma[0] * sigma[0] * pf.frequency_squared_y
        )
        return tf.complex(filter2 / tf.reduce_max(filter2), 0.0)

    def apply_blur_3d(
        self,
        I_res: tf.Tensor,
        sigma: tf.Tensor | np.ndarray,
        context: PSFContext,
        use_bead_kernel: bool = True,
    ) -> tf.Tensor:
        """
        Apply 3D blur (Gaussian + optional bead kernel) to an image volume.

        Args:
            I_res: Input intensity field (complex).
            sigma: Blur sigma values (shape: [2]).
            context: PSF context providing bead_kernel and frequency grids.
            use_bead_kernel: Whether to convolve with bead kernel.

        Returns:
            Blurred real image with trailing dimension expanded.
        """
        filter2 = self.compute_gaussian_filter(sigma, context)
        if use_bead_kernel:
            blurred = im.ifft3d(im.fft3d(I_res) * context.bead_kernel * filter2)
        else:
            blurred = im.ifft3d(im.fft3d(I_res) * filter2)
        return tf.expand_dims(tfm.real(blurred), axis=-1)

    def bin_image_3d(
        self,
        image: tf.Tensor,
        bin_factor: int,
    ) -> tf.Tensor:
        """
        Bin a 3D image volume spatially (NHWC format).

        Args:
            image: Input tensor with shape (..., H, W, C).
            bin_factor: Spatial binning factor.

        Returns:
            Binned image tensor.
        """
        ndim = len(image.shape)
        kernel_hwc = [1] * ndim
        kernel_hwc[-4] = bin_factor
        kernel_hwc[-3] = bin_factor
        kernel = np.ones(kernel_hwc, dtype=np.float32)
        return tf.nn.convolution(
            image,
            kernel,
            strides=[1, bin_factor, bin_factor, 1] if ndim == 4 else [1, 1, bin_factor, bin_factor, 1],
            padding="SAME",
            data_format="NDHWC" if ndim == 5 else "NHWC",
        )

    def _render_psf(
        self,
        pupil: tf.Tensor,
        phase_z: tf.Tensor,
        sigma: tf.Tensor | np.ndarray,
        context: PSFContext,
        phase_xy: tf.Tensor | None = None,
        use_bead_kernel: bool = True,
        data=None,
    ) -> tf.Tensor:
        """Propagate a pupil through phase, blur, bin, and optionally trim z-padding.

        This is the shared rendering pipeline used by both :meth:`calc_forward_images`
        and :meth:`genpsfmodel`.

        Args:
            pupil: Complex pupil function.
            phase_z: Z-propagation phase (complex exponential).
            sigma: Blur sigma values (shape: [2]).
            context: PSF context carrying all operational state.
            phase_xy: Optional XY phase factor for per-bead shifts.
            use_bead_kernel: Whether to convolve with the bead kernel.
            data: If provided, z-padding is trimmed using this data object.

        Returns:
            Rendered PSF intensity (last dimension squeezed, z-padding trimmed if *data* given).
        """
        propagated = self.propagate_pupil(pupil, phase_z, context, phase_xy)
        blurred = self.apply_blur_3d(propagated, sigma, context, use_bead_kernel=use_bead_kernel)
        binned = self.bin_image_3d(blurred, context.params.model.bin)
        psf = binned[..., 0]
        if data is not None:
            psf = self.trim_z_padding(psf, data, context)
        return psf

    def trim_z_padding(
        self,
        psf: tf.Tensor,
        data,
        context: PSFContext,
    ) -> tf.Tensor:
        """
        Trim z-padding from a PSF volume using bead kernel size.

        Args:
            psf: Input PSF tensor with shape (N, Nz, H, W) or similar.
            data: PreprocessedImageData providing measured ROI shape.
            context: PSF context providing z_range and bead_kernel.

        Returns:
            Trimmed PSF tensor.
        """
        pf = context.pupil_field
        Nz = pf.z_range.shape[0]
        st = (context.bead_kernel.shape[0] - data.measured_roi_images[0].shape[-3]) // 2
        return psf[..., st : Nz - st, :, :]


