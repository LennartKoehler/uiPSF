from __future__ import annotations

from abc import ABCMeta

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm
from dataclasses import dataclass, field

from .PSFInterface import PSFInterface

from .. import utilities as im

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



class PSFZernikeBase(PSFInterface, metaclass=ABCMeta):
    """
    Intermediate base class for Zernike-based PSF models.
    Provides shared helper methods for pupil computation, propagation, blurring, and binning.

    Subclasses must implement:
        - calc_initials
        - calc_forward_images
        - postprocess

    Expected instance attributes (set by subclasses or calpupilfield):
        - config: configuration object (RunParameters)
        - zernike_polynomial_basis: np.ndarray of Zernike polynomials (shape: [n_coeffs, xsz, xsz])
        - z_range: np.ndarray of z positions (shape: [Nz, 1, 1])
        - frequency_x, frequency_y: k-space coordinates (complex)
        - frequency_z, frequency_z_medium: z-k-space coordinates (complex)
        - pupil_mask: binary pupil mask
        - apodization: apodization function
        - czt_parameters: CZT parameters
        - normalization_factor: normalization factor
        - dipole_field: vector field for vector PSF (shape: [2, 3, xsz, xsz])
        - frequency_squared_x, frequency_squared_y: frequency space arrays for Gaussian blur
        - bead_kernel: complex bead kernel for convolution
        - spherical_noll_indices: Noll indices for spherical Zernike modes
        - max_magnitude_order: maximum Zernike order for magnitude
    """

    __metaclass__ = ABCMeta


    def magnitude_phase_to_pupil(
        self,
        pupil_mag: tf.Tensor,
        pupil_phase: tf.Tensor,
    ) -> tf.Tensor:
        """
        Construct a complex pupil function from magnitude and phase.

        Args:
            pupil_mag: Magnitude of the pupil function.
            pupil_phase: Phase of the pupil function in radians.

        Returns:
            Complex pupil function with aperture and apodization applied.
        """
        return tf.complex(
            pupil_mag * tfm.cos(pupil_phase),
            pupil_mag * tfm.sin(pupil_phase),
        ) * self.pupil_mask * self.apodization

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
        noll_index: np.ndarray | None = None,
    ) -> tf.Tensor:
        """
        Compute pupil function from Zernike coefficients.

        Args:
            Zcoeff_mag: Zernike magnitude coefficients (shape: [N, ...] or [n_k, 1, 1]).
            Zcoeff_phase: Zernike phase coefficients (shape: [N, ...] or [n_k, 1, 1]).
            weight_mag: Weight multiplier for magnitude.
            weight_phase: Weight multiplier for phase.
            noll_index: Optional Noll indices to select specific Zernike modes.

        Returns:
            Complex pupil function.
        """
        c1 = self.spherical_noll_indices
        n_max = getattr(self, 'n_max_mag', 15)
        Nk = np.min(((n_max + 1) * (n_max + 2) // 2, self.zernike_polynomial_basis.shape[0]))
        mask = c1 < Nk
        c1 = c1[mask]

        if noll_index is not None:
            pupil_mag = tf.reduce_sum(
                self.zernike_polynomial_basis[c1] * tf.gather(Zcoeff_mag, indices=c1) * weight_mag, axis=0
            )
        elif self.params.model.symmetric_mag:
            pupil_mag = tf.reduce_sum(
                self.zernike_polynomial_basis[c1] * tf.gather(Zcoeff_mag, indices=c1) * weight_mag, axis=0
            )
        else:
            pupil_mag = tf.reduce_sum(
                self.zernike_polynomial_basis[0:Nk] * Zcoeff_mag[0:Nk] * weight_mag, axis=0
            )
        pupil_mag = tfm.maximum(pupil_mag, 0)
        pupil_phase = tf.reduce_sum(self.zernike_polynomial_basis[3:] * Zcoeff_phase[3:] * weight_phase, axis=0)

        return self.magnitude_phase_to_pupil(pupil_mag, pupil_phase)

    def propagate_pupil(
        self,
        pupil: tf.Tensor,
        phase_z: tf.Tensor,
        phase_xy: tf.Tensor | None = None,
    ) -> tf.Tensor:
        """
        Propagate pupil function using CZT, handling scalar vs vector modes.

        Args:
            pupil: Complex pupil function.
            phase_z: Z-propagation phase (complex exponential).
            phase_xy: Optional XY phase factor for field-dependent shifts.

        Returns:
            Complex PSF field (after CZT propagation).
        """
        if self.psf_type == "vector":
            I_res = tf.constant(0.0, dtype=tf.complex64)
            for h in self.dipole_field:
                PupilFunction = pupil * phase_z * h
                if phase_xy is not None:
                    PupilFunction = PupilFunction * phase_xy
                propagated_psf_amplitude = im.cztfunc1(PupilFunction, self.czt_parameters)
                I_res += propagated_psf_amplitude * tfm.conj(propagated_psf_amplitude) * self.normalization_factor
            return I_res
        else:
            PupilFunction = pupil * phase_z
            if phase_xy is not None:
                PupilFunction = PupilFunction * phase_xy
            propagated_psf_amplitude = im.cztfunc1(PupilFunction, self.czt_parameters)
            return propagated_psf_amplitude * tfm.conj(propagated_psf_amplitude) * self.normalization_factor

    def compute_gaussian_filter(self, sigma: tf.Tensor | np.ndarray) -> tf.Tensor:
        """
        Compute a 2D Gaussian blur filter in frequency space.

        Args:
            sigma: Blur sigma values (shape: [2]) with [sigma_y, sigma_x].

        Returns:
            Complex frequency-domain filter (normalized).
        """
        filter2 = tf.exp(
            -2 * sigma[1] * sigma[1] * self.frequency_squared_x
            - 2 * sigma[0] * sigma[0] * self.frequency_squared_y
        )
        return tf.complex(filter2 / tf.reduce_max(filter2), 0.0)

    def apply_blur_3d(
        self,
        I_res: tf.Tensor,
        sigma: tf.Tensor | np.ndarray,
        use_bead_kernel: bool = True,
    ) -> tf.Tensor:
        """
        Apply 3D blur (Gaussian + optional bead kernel) to an image volume.

        Args:
            I_res: Input intensity field (complex).
            sigma: Blur sigma values (shape: [2]).
            use_bead_kernel: Whether to convolve with bead kernel.

        Returns:
            Blurred real image with trailing dimension expanded.
        """
        filter2 = self.compute_gaussian_filter(sigma)
        if use_bead_kernel:
            blurred = im.ifft3d(im.fft3d(I_res) * self.bead_kernel * filter2)
        else:
            blurred = im.ifft3d(im.fft3d(I_res) * filter2)
        return tf.expand_dims(tfm.real(blurred), axis=-1)

    def apply_blur_2d(
        self,
        I_res: tf.Tensor,
        sigma: tf.Tensor | np.ndarray,
    ) -> tf.Tensor:
        """
        Apply 2D blur (Gaussian only) to an image volume.

        Args:
            I_res: Input intensity field (complex).
            sigma: Blur sigma values (shape: [2]).

        Returns:
            Blurred real image with trailing dimension expanded.
        """
        filter2 = self.compute_gaussian_filter(sigma)
        blurred = im.ifft2d(im.fft2d(I_res) * filter2)
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

    def bin_image_2d(
        self,
        image: tf.Tensor,
        bin_factor: int,
    ) -> tf.Tensor:
        """
        Bin a 2D image (or 3D volume with z as first dim) spatially (NDHWC format).

        Args:
            image: Input tensor with shape (N, H, W, 1) or (N, D, H, W, 1).
            bin_factor: Spatial binning factor.

        Returns:
            Binned image tensor.
        """
        ndim = len(image.shape)
        if ndim == 5:
            kernel = np.ones((1, bin_factor, bin_factor, 1, 1), dtype=np.float32)
            return tf.nn.convolution(
                image,
                kernel,
                strides=(1, 1, bin_factor, bin_factor, 1),
                padding="SAME",
                data_format="NDHWC",
            )
        else:
            kernel = np.ones((bin_factor, bin_factor, 1, 1), dtype=np.float32)
            return tf.nn.convolution(
                image,
                kernel,
                strides=(1, bin_factor, bin_factor, 1),
                padding="SAME",
                data_format="NHWC",
            )

    def _render_psf(
        self,
        pupil: tf.Tensor,
        phase_z: tf.Tensor,
        sigma: tf.Tensor | np.ndarray,
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
            phase_xy: Optional XY phase factor for per-bead shifts.
            use_bead_kernel: Whether to convolve with the bead kernel.
            data: If provided, z-padding is trimmed using this data object.

        Returns:
            Rendered PSF intensity (last dimension squeezed, z-padding trimmed if *data* given).
        """
        propagated = self.propagate_pupil(pupil, phase_z, phase_xy)
        blurred = self.apply_blur_3d(propagated, sigma, use_bead_kernel=use_bead_kernel)
        binned = self.bin_image_3d(blurred, self.params.model.bin)
        psf = binned[..., 0]
        if data is not None:
            psf = self.trim_z_padding(psf, data)
        return psf

    def trim_z_padding(
        self,
        psf: tf.Tensor,
        data,
    ) -> tf.Tensor:
        """
        Trim z-padding from a PSF volume using bead kernel size.

        Args:
            psf: Input PSF tensor with shape (N, Nz, H, W) or similar.
            data: PreprocessedImageData providing measured ROI shape.

        Returns:
            Trimmed PSF tensor.
        """
        Nz = self.z_range.shape[0]
        st = (self.bead_kernel.shape[0] - data.measured_roi_images[0].shape[-3]) // 2
        return psf[..., st : Nz - st, :, :]

    def interpolate_zernike_map(
        self,
        zmap: tf.Tensor,
        centers: tf.Tensor,
        image_size: tuple[int, int],
    ) -> tf.Tensor:
        """
        Interpolate a per-bead Zernike map onto pixel coordinates.

        Args:
            zmap: Zernike coefficient map (shape: [N, n_zernike]).
            centers: Bead center coordinates (shape: [N, 2] or [N, 3]).
            image_size: Target image size (H, W).

        Returns:
            Interpolated Zernike coefficients (shape: [N, n_zernike, H, W]).
        """
        from tensorflow_probability import stats as tfs

        N, n_z = zmap.shape
        H, W = image_size
        x_coords = tf.cast(tf.linspace(0.0, tf.cast(W - 1, tf.float32), W), tf.float32)
        y_coords = tf.cast(tf.linspace(0.0, tf.cast(H - 1, tf.float32), H), tf.float32)
        grid_x, grid_y = tf.meshgrid(x_coords, y_coords, indexing="xy")
        grid_y = tf.expand_dims(tf.transpose(grid_y), axis=-1)
        grid_x = tf.expand_dims(tf.transpose(grid_x), axis=-1)

        centers_T = tf.transpose(centers[..., -2:])
        dist = tf.sqrt(
            (grid_x - tf.reshape(centers_T[0], [1, 1, N])) ** 2
            + (grid_y - tf.reshape(centers_T[1], [1, 1, N])) ** 2
        )
        radius = tf.cast(
            0.5 * np.sqrt(H * H + W * W), tf.float32
        ) * self.params.model.search_radius
        kernel = tfm.maximum(1.0 - dist / radius, 0.0)
        kernel = kernel / (tf.reduce_sum(kernel, axis=-1, keepdims=True) + 1e-12)
        zmap_exp = tf.transpose(zmap)
        interpolated = tf.einsum("zxy,nz->nxyz", kernel, zmap_exp)
        interpolated = tf.transpose(interpolated, [0, 3, 1, 2])

        return interpolated

