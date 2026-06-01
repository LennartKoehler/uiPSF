from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from ..core.fourier import fft3d, ifft3d, cztfunc
from ..core.forward import (
    construct_zernike_pupil,
    construct_4pi_pupils,
    interpolate_zmap,
    compute_psf_intensity,
    apply_blur_and_bead,
    apply_binning,
    trim_z,
    apply_drift,
)
from ..optics.pupil_field import PupilField, compute_pupil_field
from ..optics.bead_kernel import generate_bead_kernel
from ..optics.four_pi import psf2IAB


class PSFModel(nn.Module):
    # Registered buffer type annotations (set at runtime)
    zernike_basis: torch.Tensor
    pupil_aperture: torch.Tensor
    apodization: torch.Tensor
    axial_freq_immersion: torch.Tensor
    lateral_freq_x: torch.Tensor
    lateral_freq_y: torch.Tensor
    defocus_positions: torch.Tensor
    kspace_x_squared: torch.Tensor
    kspace_y_squared: torch.Tensor
    axial_freq_medium: torch.Tensor
    drift_z_positions: torch.Tensor
    drift_freq_x: torch.Tensor
    drift_freq_y: torch.Tensor
    drift_freq_z: torch.Tensor
    normalization_factor: torch.Tensor
    czt_chirp_A: torch.Tensor
    czt_chirp_B_fft: torch.Tensor
    czt_chirp_C: torch.Tensor
    dipole_field: torch.Tensor
    symmetric_zernike_indices: torch.Tensor
    bead_kernel: torch.Tensor

    def __init__(self):
        super().__init__()
        self.data: Any = None
        self.options: Any = None
        self.pf: Any = None
        self.weight: Any = None
        self.variable_info: Any = None
        self.batch_indices: list[int] = [0, 0]

    def calc_initials(self, data, start_time=None):
        raise NotImplementedError

    def calc_forward_images(self, variables):
        raise NotImplementedError

    def postprocess(self, variables):
        raise NotImplementedError

    def res2dict(self, res):
        raise NotImplementedError

    def _init_pupil_field(self, fieldtype="scalar", Nz=None):
        self.pf = compute_pupil_field(
            self.options, self.data, fieldtype=fieldtype, Nz=Nz, device="cpu"
        )
        self._register_pupil_field_buffers()

    def _register_pupil_field_buffers(self):
        pf = self.pf
        self.register_buffer('zernike_basis', torch.from_numpy(pf.Zk))
        self.register_buffer('pupil_aperture', torch.from_numpy(pf.aperture))
        self.register_buffer('apodization', torch.from_numpy(pf.apoid))
        self.register_buffer('axial_freq_immersion', torch.from_numpy(pf.kz))
        self.register_buffer('lateral_freq_x', torch.from_numpy(pf.kx))
        self.register_buffer('lateral_freq_y', torch.from_numpy(pf.ky))
        self.register_buffer('defocus_positions', torch.from_numpy(pf.Zrange))
        self.register_buffer('kspace_x_squared', torch.from_numpy(pf.kspace_x))
        self.register_buffer('kspace_y_squared', torch.from_numpy(pf.kspace_y))
        self.register_buffer('axial_freq_medium', torch.from_numpy(pf.kz_med))
        self.register_buffer('drift_z_positions', torch.from_numpy(pf.zv))
        self.register_buffer('drift_freq_x', torch.from_numpy(pf.kxv))
        self.register_buffer('drift_freq_y', torch.from_numpy(pf.kyv))
        self.register_buffer('drift_freq_z', torch.from_numpy(pf.kzv))
        self.register_buffer('normalization_factor', torch.tensor(pf.normf, dtype=torch.complex64))

        A, Bh, C = pf.paramxy
        self.register_buffer('czt_chirp_A', torch.from_numpy(A).to(torch.complex64))
        self.register_buffer('czt_chirp_B_fft', torch.from_numpy(Bh).to(torch.complex64))
        self.register_buffer('czt_chirp_C', torch.from_numpy(C).to(torch.complex64))

        if pf.dipole_field is not None:
            self.register_buffer('dipole_field', torch.from_numpy(pf.dipole_field))

        self.register_buffer('symmetric_zernike_indices', torch.from_numpy(pf.spherical_terms))

    @property
    def czt_params(self):
        return (self.czt_chirp_A, self.czt_chirp_B_fft, self.czt_chirp_C)

    def _init_bead_kernel(self, is_volume=False):
        bin_val = self.options.model.bin
        bead_kernel_numpy = generate_bead_kernel(
            pixelsize_z=self.data.pixelsize_z,
            pixelsize_x=self.data.pixelsize_x,
            pixelsize_y=self.data.pixelsize_y,
            bead_radius=self.data.bead_radius,
            roi_shape=self.data.rois.shape,
            is_volume=is_volume,
            bin_val=bin_val,
        )
        bead_kernel_tensor = torch.complex(
            torch.from_numpy(bead_kernel_numpy), torch.tensor(0.0)
        )
        self.register_buffer('bead_kernel', bead_kernel_tensor)
        self.bead_kernel_numpy = bead_kernel_numpy
        return bead_kernel_numpy, bead_kernel_tensor


class ZernikePSF(PSFModel):
    """Single-objective PSF model with Zernike polynomial pupil function.

    The PSF is computed via Fourier optics: the complex pupil function
    P(kx,ky) = A(kx,ky) * exp(i*phi(kx,ky)) * aperture * apodization
    is propagated to the image plane using a chirped z-transform, where
    both the pupil magnitude A and phase phi are expanded as weighted
    sums of Zernike polynomials with fit coefficients.

    Weight array indices:
        [0] = intensity      — scales photon count
        [1] = background     — scales offset level
        [2] = drift_rate     — scales lateral drift per z-slice
        [3] = zernike_phase  — scales phase aberration coefficients
        [4] = zernike_mag    — scales pupil magnitude coefficients
    """

    W_INTENSITY = 0
    W_BACKGROUND = 1
    W_DRIFT = 2
    W_ZERNIKE_PHASE = 3
    W_ZERNIKE_MAG = 4

    def __init__(self, options=None):
        super().__init__()
        self.options = options
        self.initial_pupil = None
        self.defocus = np.float32(0)
        self.psftype = "scalar"

    def calc_initials(self, data, start_time=None):
        """Initialize all fitting variables for the Zernike PSF model.

        Returns
        -------
        tuple[list[np.ndarray], Any]
            [positions, backgrounds, intensities, Zcoeff, sigma, drift] and start_time.
        """
        self.data = data
        _, rois, _, _ = self.data.get_image_data()
        options = self.options

        if options.model.with_IMM:
            initial_positions = np.zeros((rois.shape[0], len(rois.shape)))
        else:
            initial_positions = np.zeros((rois.shape[0], len(rois.shape) - 1))

        initial_backgrounds = np.array(
            np.min(gaussian_filter(rois, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True)
        )
        initial_intensities = np.sum(rois - initial_backgrounds, axis=(-2, -1), keepdims=True)
        initial_intensities = np.mean(initial_intensities, axis=1, keepdims=True)

        self._init_bead_kernel()
        num_beads = rois.shape[0]
        num_z_slices = rois.shape[-3]

        self._init_pupil_field("vector" if self.psftype == "vector" else "scalar")

        if options.model.const_pupilmag:
            self.max_zernike_mag_degree = 0
        else:
            self.max_zernike_mag_degree = 100

        initial_backgrounds[initial_backgrounds < 0.1] = 0.1
        background_median = np.median(initial_backgrounds)
        median_intensity = np.lib.scimath.sqrt(np.median(initial_intensities))
        weight = [
            median_intensity * 100,
            background_median,
            1.0 / median_intensity * 40,
            0.5 / median_intensity * 40,
            0.5 / median_intensity * 40,
        ]
        self.weight = np.array(weight, dtype=np.float32)

        sigma = np.ones((2,)) * self.options.model.blur_sigma * np.pi
        self.initial_blur_sigma = sigma

        initial_zernike_coeff = np.zeros((2, self.pf.Zk.shape[0], 1, 1))
        initial_zernike_coeff[:, 0, 0, 0] = [1, 0] / self.weight[self.W_ZERNIKE_MAG]

        initial_backgrounds = (
            np.ones((num_beads, 1, 1, 1), dtype=np.float32)
            * np.median(initial_backgrounds, axis=0, keepdims=True)
            / self.weight[self.W_BACKGROUND]
        )
        initial_drift_rates = np.zeros((num_beads, 2), dtype=np.float32)

        self.variable_info = [
            dict(type="Nfit", id=0),
            dict(type="Nfit", id=0),
            dict(type="Nfit", id=0),
            dict(type="shared"),
            dict(type="shared"),
            dict(type="Nfit", id=0),
        ]

        if options.model.var_photon:
            per_slice_intensity = np.ones((num_beads, num_z_slices, 1, 1), dtype=np.float32) * initial_intensities
            initial_intensities_weighted = per_slice_intensity / self.weight[self.W_INTENSITY]
        else:
            initial_intensities_weighted = initial_intensities / self.weight[self.W_INTENSITY]

        return [
            initial_positions.astype(np.float32),
            initial_backgrounds.astype(np.float32),
            initial_intensities_weighted.astype(np.float32),
            initial_zernike_coeff.astype(np.float32),
            sigma.astype(np.float32),
            initial_drift_rates,
        ], start_time

    def calc_forward_images(self, variables):
        """Compute the forward PSF model images for all beads.

        Parameters
        ----------
        variables : list
            [positions, backgrounds, intensities, Zcoeff, sigma, drift]

        Returns
        -------
        torch.Tensor
            Forward model images matching the data ROI shapes.
        """
        positions, backgrounds, intensities, Zcoeff, sigma, drift = variables

        if self.initial_pupil is not None:
            pupil = torch.from_numpy(self.initial_pupil.astype(np.complex64))
        else:
            pupil = construct_zernike_pupil(
                self.zernike_basis,
                torch.from_numpy(Zcoeff),
                torch.from_numpy(Zcoeff),
                self.pupil_aperture,
                self.apodization,
                self.symmetric_zernike_indices,
                self.max_zernike_mag_degree,
                self.options.model.symmetric_mag,
                self.weight[self.W_ZERNIKE_MAG],
                self.weight[self.W_ZERNIKE_PHASE],
            )

        positions_complex = torch.complex(
            torch.from_numpy(positions.reshape(positions.shape + (1, 1, 1))), torch.tensor(0.0)
        )

        if positions.shape[1] > 3:
            defocus_phase = 1j * 2 * np.pi * (
                self.axial_freq_medium * positions_complex[:, 1]
                - self.axial_freq_immersion * (positions_complex[:, 0] + self.defocus_positions)
            )
            lateral_shift_phase = (
                1j * 2 * np.pi * self.lateral_freq_y * positions_complex[:, 2]
                + 1j * 2 * np.pi * self.lateral_freq_x * positions_complex[:, 3]
            )
        else:
            defocus_phase = -1j * 2 * np.pi * self.axial_freq_immersion * (
                positions_complex[:, 0] + self.defocus_positions + self.defocus
            )
            lateral_shift_phase = (
                1j * 2 * np.pi * self.lateral_freq_y * positions_complex[:, 1]
                + 1j * 2 * np.pi * self.lateral_freq_x * positions_complex[:, 2]
            )

        psf_intensity = compute_psf_intensity(
            pupil, defocus_phase, lateral_shift_phase, self.czt_params, self.normalization_factor,
            dipole_field=self.dipole_field if self.psftype == "vector" else None,
            fieldtype=self.psftype,
        )

        if not self.options.model.var_blur:
            sigma = self.initial_blur_sigma
        sigma_tensor = torch.from_numpy(sigma.astype(np.float32))
        blurred_psf = apply_blur_and_bead(
            psf_intensity, sigma_tensor, self.kspace_x_squared, self.kspace_y_squared, self.bead_kernel,
        )

        bin_val = self.options.model.bin
        if bin_val > 1:
            psf_fit = apply_binning(blurred_psf, bin_val)
        else:
            psf_fit = torch.real(blurred_psf)

        psf_fit = trim_z(psf_fit, self.bead_kernel_numpy.shape, self.data.rois.shape[-3])

        if self.options.model.estimate_drift:
            drift_scaled = drift * self.weight[self.W_DRIFT]
            psf_drifted = apply_drift(
                psf_fit,
                torch.from_numpy(drift_scaled.astype(np.float32)),
                self.drift_z_positions, self.drift_freq_x, self.drift_freq_y, self.drift_freq_z,
                self.data.skew_const if hasattr(self.data, "skew_const") and self.data.skew_const else None,
            )
            psf_for_output = psf_drifted
        else:
            psf_for_output = psf_fit

        forward_images = (
            psf_for_output
            * torch.from_numpy(intensities.astype(np.float32))
            * self.weight[self.W_INTENSITY]
            + torch.from_numpy(backgrounds.astype(np.float32)) * self.weight[self.W_BACKGROUND]
        )

        return forward_images

    def genpsfmodel(self, sigma, Zcoeff=None, pupil=None, addbead=False):
        if pupil is None:
            zcoeff_mag = Zcoeff[0] * self.weight[self.W_ZERNIKE_MAG] if Zcoeff is not None else None
            zcoeff_phase = Zcoeff[1] * self.weight[self.W_ZERNIKE_PHASE] if Zcoeff is not None else None
            pupil_mag = torch.sum(self.zernike_basis * torch.from_numpy(zcoeff_mag), dim=0) if zcoeff_mag is not None else None
            pupil_mag = torch.clamp(pupil_mag, min=0) if pupil_mag is not None else None
            pupil_phase = torch.sum(self.zernike_basis * torch.from_numpy(zcoeff_phase), dim=0) if zcoeff_phase is not None else None
            if pupil_mag is not None and pupil_phase is not None:
                pupil = torch.complex(
                    pupil_mag * torch.cos(pupil_phase),
                    pupil_mag * torch.sin(pupil_phase),
                ) * self.pupil_aperture * self.apodization

        defocus_phase = -1j * 2 * np.pi * self.axial_freq_immersion * (self.defocus_positions + self.defocus)

        if self.psftype == "vector":
            psf_intensity = torch.zeros(
                self.defocus_positions.shape[0], pupil.shape[-2], pupil.shape[-1], dtype=torch.complex64
            )
            for dipole_component in self.dipole_field:
                pupil_func = pupil * torch.exp(defocus_phase) * dipole_component
                field = cztfunc(pupil_func, self.czt_params)
                psf_intensity = psf_intensity + field * torch.conj(field) * self.normalization_factor
        else:
            pupil_func = pupil * torch.exp(defocus_phase)
            field = cztfunc(pupil_func, self.czt_params)
            psf_intensity = field * torch.conj(field) * self.normalization_factor

        sigma_tensor = torch.from_numpy(sigma.astype(np.float32))
        gaussian_filter_freq = torch.exp(
            -2 * sigma_tensor[1] ** 2 * self.kspace_x_squared - 2 * sigma_tensor[0] ** 2 * self.kspace_y_squared
        )
        gaussian_filter_freq = torch.complex(gaussian_filter_freq / torch.max(gaussian_filter_freq), torch.tensor(0.0))

        if addbead:
            blurred_psf = torch.real(ifft3d(fft3d(psf_intensity) * gaussian_filter_freq * self.bead_kernel))
        else:
            blurred_psf = torch.real(ifft3d(fft3d(psf_intensity) * gaussian_filter_freq))

        bin_val = self.options.model.bin
        if bin_val > 1:
            blurred_psf = blurred_psf.unsqueeze(-1)
            bin_kernel = torch.ones(bin_val, bin_val, 1, 1, dtype=torch.float32)
            I_model = F.conv2d(
                blurred_psf.permute(2, 0, 1, 3),
                bin_kernel.permute(2, 0, 1, 3),
                stride=(1, bin_val),
                padding="same",
            )
            I_model = I_model.permute(1, 2, 0, 3)[..., 0]
        else:
            I_model = blurred_psf

        return I_model, pupil

    def postprocess(self, variables):
        positions, backgrounds, intensities, Zcoeff, sigma, drift_rates = variables
        z_center = (self.defocus_positions.shape[-3] - 1) // 2
        Zcoeff[0] = Zcoeff[0] * self.weight[self.W_ZERNIKE_MAG]
        Zcoeff[1] = Zcoeff[1] * self.weight[self.W_ZERNIKE_PHASE]

        bin_val = self.options.model.bin
        positions[:, 1:] = positions[:, 1:] / bin_val

        if self.initial_pupil is not None:
            pupil = self.initial_pupil
            I_model, _ = self.genpsfmodel(sigma, pupil=pupil)
            I_model_bead, _ = self.genpsfmodel(sigma, pupil=pupil, addbead=True)
        else:
            I_model, pupil = self.genpsfmodel(sigma, Zcoeff=Zcoeff)
            I_model_bead, _ = self.genpsfmodel(sigma, Zcoeff=Zcoeff, addbead=True)

        _, _, centers, _ = self.data.get_image_data()

        if positions.shape[1] > 3:
            global_positions = np.swapaxes(
                np.vstack(
                    (
                        positions[:, 0] + z_center,
                        positions[:, 1],
                        centers[:, -2] - positions[:, -2],
                        centers[:, -1] - positions[:, -1],
                    )
                ),
                1,
                0,
            )
        else:
            global_positions = np.swapaxes(
                np.vstack(
                    (
                        positions[:, 0] + z_center,
                        centers[:, -2] - positions[:, -2],
                        centers[:, -1] - positions[:, -1],
                    )
                ),
                1,
                0,
            )

        return [
            global_positions.astype(np.float32),
            backgrounds * self.weight[self.W_BACKGROUND],
            intensities * self.weight[self.W_INTENSITY],
            I_model_bead,
            I_model,
            np.complex64(pupil.numpy()) if isinstance(pupil, torch.Tensor) else np.complex64(pupil),
            Zcoeff,
            sigma,
            drift_rates * self.weight[self.W_DRIFT],
            np.flip(I_model.numpy() if isinstance(I_model, torch.Tensor) else I_model, axis=-3),
            variables,
        ]

    def res2dict(self, res):
        I_model = res[4]
        I_model_np = I_model.numpy() if isinstance(I_model, torch.Tensor) else I_model
        pupil = res[5]
        pupil_np = pupil.numpy() if isinstance(pupil, torch.Tensor) else pupil
        res_dict = dict(
            pos=res[0],
            bg=np.squeeze(res[1]),
            intensity=np.squeeze(res[2]),
            I_model_bead=res[3],
            I_model=I_model_np,
            pupil=pupil_np,
            zernike_coeff=np.squeeze(res[6]),
            sigma=np.squeeze(res[7]) / np.pi,
            drift_rate=res[8],
            I_model_reverse=res[9],
            offset=np.min(I_model_np),
            zernike_polynomial=self.pf.Zk,
            apodization=self.pf.apoid,
            cor_all=self.data.centers_all,
            cor=self.data.centers,
        )
        return res_dict


class ZernikePSF4Pi(PSFModel):
    """4Pi microscope PSF model with Zernike polynomial pupil function.

    In a 4Pi microscope, two opposing objectives illuminate the sample
    from both sides, creating an interference pattern in the pupil plane.
    The detected intensity is a mixture of the coherent interference
    (modulated) and incoherent (widefield) components:

        I = alpha * I_modulated + (1-alpha) * I_widefield

    Weight array indices:
        [0] = intensity          — scales photon count
        [1] = background         — scales offset level
        [2] = drift_rate         — scales lateral drift per z-slice
        [3] = zernike_phase      — scales phase aberration coefficients
        [4] = zernike_mag        — scales pupil magnitude coefficients
        [5] = modulation_depth   — scales the alpha parameter
    """

    W_INTENSITY = 0
    W_BACKGROUND = 1
    W_DRIFT = 2
    W_ZERNIKE_PHASE = 3
    W_ZERNIKE_MAG = 4
    W_MODULATION_DEPTH = 5

    def __init__(self, options=None):
        super().__init__()
        self.options = options
        self.deconv_phase_offset = None

    def calc_initials(self, data, start_time=None):
        """Initialize all fitting variables for the 4Pi Zernike PSF model.

        Returns
        -------
        tuple[list[np.ndarray], Any]
            [positions, backgrounds, intensities, intensity_phase,
             Zcoeff_mag, Zcoeff_phase, sigma, alpha, pos_shift, phase_dm, drift]
            and start_time.
        """
        self.data = data
        _, rois, _, _ = self.data.get_image_data()

        I_data, A_data, _, initial_interference_phase = psf2IAB(rois)
        initial_interference_phase = np.zeros((I_data.shape[0], 1, 1, 1))
        initial_positions = np.zeros([I_data.shape[0], len(I_data.shape) - 1]).astype(np.float32)
        initial_backgrounds = np.min(gaussian_filter(I_data, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True)
        initial_intensities = np.sum(I_data - initial_backgrounds, axis=(-2, -1), keepdims=True)
        initial_intensities = np.mean(initial_intensities, axis=1, keepdims=True)

        self._init_bead_kernel()
        num_beads = rois.shape[0]
        num_z_slices = self.bead_kernel_numpy.shape[0]

        self._init_pupil_field("scalar")

        if self.options.model.const_pupilmag:
            self.max_zernike_mag_degree = 0
        else:
            self.max_zernike_mag_degree = 100

        sigma = np.ones((2,)) * self.options.model.blur_sigma * np.pi

        self.axial_phase_array = (
            np.linspace(-num_z_slices / 2 + 0.5, num_z_slices / 2 - 0.5, num_z_slices, dtype=np.float32).reshape(
                num_z_slices, 1, 1
            )
            * 2
            * np.pi
        )
        self.modulation_period_z = self.data.zT

        initial_backgrounds[initial_backgrounds < 0.1] = 0.1
        background_median = np.median(initial_backgrounds)
        median_intensity = np.lib.scimath.sqrt(np.median(initial_intensities))
        weight = [
            median_intensity * 100,
            background_median,
            1.0 / median_intensity * 40,
            1.0 / median_intensity * 40,
            1.0 / median_intensity * 40,
            1.0 / median_intensity * 40,
        ]
        self.weight = np.array(weight, dtype=np.float32)

        initial_zernike_mag = np.zeros((2, self.pf.Zk.shape[0], 1, 1))
        initial_zernike_mag[:, 0, 0, 0] = [1, 1] / self.weight[self.W_ZERNIKE_MAG]
        initial_zernike_phase = np.zeros((2, self.pf.Zk.shape[0], 1, 1))

        phase_dm = self.options.fpi.phase_dm
        deconv_phase_steps = np.reshape(np.array(phase_dm), (len(phase_dm), 1, 1, 1, 1)).astype(np.float32)

        initial_backgrounds = (
            np.ones((num_beads, 1, 1, 1), dtype=np.float32)
            * np.median(initial_backgrounds, axis=0, keepdims=True)
            / self.weight[self.W_BACKGROUND]
        )
        initial_drift_rates = np.zeros((num_beads, 2), dtype=np.float32)
        alpha = np.array([0.8]) / self.weight[self.W_MODULATION_DEPTH]
        initial_obj_misalignment = np.zeros(initial_positions.shape)

        self.variable_info = [
            dict(type="Nfit", id=0),
            dict(type="Nfit", id=0),
            dict(type="Nfit", id=0),
            dict(type="Nfit", id=0),
            dict(type="shared"),
            dict(type="shared"),
            dict(type="shared"),
            dict(type="shared"),
            dict(type="Nfit", id=0),
            dict(type="shared"),
            dict(type="Nfit", id=0),
        ]

        if self.options.model.var_photon:
            per_slice_intensity = np.ones((num_beads, rois.shape[-3], 1, 1), dtype=np.float32) * initial_intensities
            initial_intensities_weighted = per_slice_intensity / self.weight[self.W_INTENSITY]
        else:
            initial_intensities_weighted = initial_intensities / self.weight[self.W_INTENSITY]

        return [
            initial_positions.astype(np.float32),
            initial_backgrounds.astype(np.float32),
            initial_intensities_weighted.astype(np.float32),
            initial_interference_phase.astype(np.float32),
            initial_zernike_mag.astype(np.float32),
            initial_zernike_phase.astype(np.float32),
            sigma.astype(np.float32),
            alpha.astype(np.float32),
            initial_obj_misalignment.astype(np.float32),
            deconv_phase_steps.astype(np.float32),
            initial_drift_rates,
        ], start_time

    def calc_forward_images(self, variables):
        """Compute the forward 4Pi PSF model images.

        Parameters
        ----------
        variables : list
            [positions, backgrounds, intensities, intensity_phase,
             Zcoeff_mag, Zcoeff_phase, sigma, alpha, pos_shift, phase_dm, drift]

        Returns
        -------
        torch.Tensor
            Forward model 4Pi images.
        """
        positions, backgrounds, intensity_abs, intensity_phase, Zcoeff_mag, Zcoeff_phase, sigma, alpha, pos_shift, phase_dm, drift = variables

        interference_phase_complex = torch.complex(
            torch.cos(torch.from_numpy(intensity_phase.astype(np.float32))),
            torch.sin(torch.from_numpy(intensity_phase.astype(np.float32))),
        )
        deconv_offset = self.deconv_phase_offset if self.deconv_phase_offset is not None else 0
        deconv_phase_complex = torch.complex(
            torch.cos(torch.from_numpy(phase_dm.astype(np.float32)) + deconv_offset),
            torch.sin(torch.from_numpy(phase_dm.astype(np.float32)) + deconv_offset),
        )

        positions_complex = torch.complex(
            torch.from_numpy(positions.reshape(positions.shape + (1, 1, 1)).astype(np.float32)),
            torch.tensor(0.0),
        )
        pos_shift_complex = torch.complex(
            torch.from_numpy(pos_shift.reshape(pos_shift.shape + (1, 1, 1)).astype(np.float32)),
            torch.tensor(0.0),
        )

        pupil_upper, pupil_lower = construct_4pi_pupils(
            self.zernike_basis, self.pupil_aperture, self.apodization,
            torch.from_numpy(Zcoeff_mag.astype(np.float32)),
            torch.from_numpy(Zcoeff_phase.astype(np.float32)),
            self.symmetric_zernike_indices, self.max_zernike_mag_degree,
            self.options.model.symmetric_mag,
            self.weight[self.W_ZERNIKE_MAG],
            self.weight[self.W_ZERNIKE_PHASE],
        )

        defocus_phase = -1j * 2 * np.pi * self.axial_freq_immersion * (positions_complex[:, 0] + self.defocus_positions)
        lateral_shift_phase = (
            1j * 2 * np.pi * self.lateral_freq_y * positions_complex[:, 1]
            + 1j * 2 * np.pi * self.lateral_freq_x * positions_complex[:, 2]
        )

        pupil_func_coherent = (
            pupil_upper * torch.exp(-defocus_phase) * interference_phase_complex
            + pupil_lower * torch.exp(defocus_phase) * deconv_phase_complex
        ) * torch.exp(lateral_shift_phase)
        field_modulated = cztfunc(pupil_func_coherent, self.czt_params)
        I_modulated = field_modulated * torch.conj(field_modulated) * self.normalization_factor / 2.0

        pupil_func_upper = pupil_upper * torch.exp(-defocus_phase) * torch.exp(lateral_shift_phase)
        field_upper = cztfunc(pupil_func_upper, self.czt_params)
        I_upper = field_upper * torch.conj(field_upper) * self.normalization_factor / 2.0

        pupil_func_lower = pupil_lower * torch.exp(defocus_phase) * torch.exp(lateral_shift_phase)
        field_lower = cztfunc(pupil_func_lower, self.czt_params)
        I_lower = field_lower * torch.conj(field_lower) * self.normalization_factor / 2.0

        I_widefield = I_upper + I_lower

        alpha_complex = torch.complex(
            torch.from_numpy(alpha.astype(np.float32)) * self.weight[self.W_MODULATION_DEPTH],
            torch.tensor(0.0),
        )
        psf_intensity = alpha_complex * I_modulated + (1 - alpha_complex) * I_widefield

        sigma_tensor = torch.from_numpy(sigma.astype(np.float32))
        blurred_psf = apply_blur_and_bead(
            psf_intensity, sigma_tensor, self.kspace_x_squared, self.kspace_y_squared, self.bead_kernel,
        )

        psf_fit = torch.real(blurred_psf)
        psf_fit = trim_z(psf_fit, self.bead_kernel_numpy.shape, self.data.rois.shape[-3])

        if self.options.model.estimate_drift:
            drift_scaled = drift * self.weight[self.W_DRIFT]
            psf_drifted = apply_drift(
                psf_fit,
                torch.from_numpy(drift_scaled.astype(np.float32)),
                self.drift_z_positions, self.drift_freq_x, self.drift_freq_y, self.drift_freq_z,
                self.data.skew_const if hasattr(self.data, "skew_const") and self.data.skew_const else None,
            )
            psf_drifted = psf_drifted * torch.from_numpy(intensity_abs.astype(np.float32)) * self.weight[self.W_INTENSITY] + torch.from_numpy(backgrounds.astype(np.float32)) * self.weight[self.W_BACKGROUND]
            forward_images = psf_drifted.permute(1, 0, 2, 3, 4)
        else:
            psf_fit = psf_fit * torch.from_numpy(intensity_abs.astype(np.float32)) * self.weight[self.W_INTENSITY] + torch.from_numpy(backgrounds.astype(np.float32)) * self.weight[self.W_BACKGROUND]
            forward_images = psf_fit.permute(1, 0, 2, 3, 4)

        return forward_images

    def genpsfmodel(self, sigma, Zcoeffmag, Zcoeffphase, alpha):
        deconv_offset = self.deconv_phase_offset if self.deconv_phase_offset is not None else 0
        phase0 = np.reshape(np.array([-2 / 3, 0, 2 / 3]) * np.pi + deconv_offset, (3, 1, 1, 1)).astype(np.float32)
        deconv_phase_complex = torch.complex(torch.cos(torch.from_numpy(phase0)), torch.sin(torch.from_numpy(phase0)))

        pupil_mag_upper = torch.sum(self.zernike_basis * torch.from_numpy(Zcoeffmag[0]), dim=0)
        pupil_mag_upper = torch.clamp(pupil_mag_upper, min=0)
        pupil_phase_upper = torch.sum(self.zernike_basis[1:] * torch.from_numpy(Zcoeffphase[0][1:]), dim=0)
        pupil_upper = torch.complex(
            pupil_mag_upper * torch.cos(pupil_phase_upper),
            pupil_mag_upper * torch.sin(pupil_phase_upper),
        ) * self.pupil_aperture * self.apodization

        pupil_mag_lower = torch.sum(self.zernike_basis * torch.from_numpy(Zcoeffmag[1]), dim=0)
        pupil_mag_lower = torch.clamp(pupil_mag_lower, min=0)
        pupil_phase_lower = torch.sum(self.zernike_basis * torch.from_numpy(Zcoeffphase[1]), dim=0)
        pupil_lower = torch.complex(
            pupil_mag_lower * torch.cos(pupil_phase_lower),
            pupil_mag_lower * torch.sin(pupil_phase_lower),
        ) * self.pupil_aperture * self.apodization

        defocus_phase = -1j * 2 * np.pi * self.axial_freq_immersion * self.defocus_positions

        pupil_func_coherent = pupil_upper * torch.exp(-defocus_phase) + pupil_lower * torch.exp(defocus_phase) * deconv_phase_complex
        field_modulated = cztfunc(pupil_func_coherent, self.czt_params)
        I_modulated = field_modulated * torch.conj(field_modulated) * self.normalization_factor / 2.0

        pupil_func_upper = pupil_upper * torch.exp(-defocus_phase)
        field_upper = cztfunc(pupil_func_upper, self.czt_params)
        I_upper = field_upper * torch.conj(field_upper) * self.normalization_factor / 2.0

        pupil_func_lower = pupil_lower * torch.exp(defocus_phase)
        field_lower = cztfunc(pupil_func_lower, self.czt_params)
        I_lower = field_lower * torch.conj(field_lower) * self.normalization_factor / 2.0

        I_widefield = I_upper + I_lower
        psf_intensity = alpha * I_modulated + (1 - alpha) * I_widefield

        sigma_tensor = torch.from_numpy(sigma.astype(np.float32))
        gaussian_filter_freq = torch.exp(
            -2 * sigma_tensor[1] ** 2 * self.kspace_x_squared
            - 2 * sigma_tensor[0] ** 2 * self.kspace_y_squared
        )
        gaussian_filter_freq = torch.complex(gaussian_filter_freq / torch.max(gaussian_filter_freq), torch.tensor(0.0))

        axial_phase_normalized = -torch.from_numpy(self.axial_phase_array) / self.modulation_period_z
        axial_phase_complex = torch.complex(torch.cos(axial_phase_normalized), torch.sin(axial_phase_normalized))

        psf_model = torch.real(ifft3d(fft3d(psf_intensity) * gaussian_filter_freq))

        I_model, A_model, _, _ = psf2IAB(np.expand_dims(psf_model.numpy(), axis=0))
        A_model = A_model[0] * axial_phase_complex.numpy()

        return psf_model.numpy()[1], I_model[0], A_model, pupil_upper, pupil_lower

    def postprocess(self, variables):
        pos, bg, intensity_abs, intensity_phase, Zcoeffmag, Zcoeffphase, sigma, alpha, pos_d, phasec, drift_rates = variables

        interference_phase_complex = torch.complex(
            torch.cos(torch.from_numpy(intensity_phase.astype(np.float32))),
            torch.sin(torch.from_numpy(intensity_phase.astype(np.float32))),
        )
        intensities = intensity_abs * self.weight[self.W_INTENSITY] * interference_phase_complex.numpy()
        alpha_val = alpha * self.weight[self.W_MODULATION_DEPTH]

        Zcoeffmag = Zcoeffmag * self.weight[self.W_ZERNIKE_MAG]
        Zcoeffphase = Zcoeffphase * self.weight[self.W_ZERNIKE_PHASE]
        psf_model, I_model, A_model, pupil_upper, pupil_lower = self.genpsfmodel(
            sigma, Zcoeffmag, Zcoeffphase, alpha_val
        )
        drift_rates = drift_rates * self.weight[self.W_DRIFT]

        z_center = (I_model.shape[-3] - 1) // 2
        _, _, centers, _ = self.data.get_image_data()
        global_positions = np.swapaxes(
            np.vstack(
                (
                    pos[:, 0] + z_center,
                    centers[:, -2] - pos[:, -2],
                    centers[:, -1] - pos[:, -1],
                )
            ),
            1,
            0,
        )

        return [
            global_positions.astype(np.float32),
            bg * self.weight[self.W_BACKGROUND],
            intensities,
            I_model,
            A_model,
            np.complex64(pupil_upper.numpy()) if isinstance(pupil_upper, torch.Tensor) else pupil_upper,
            np.complex64(pupil_lower.numpy()) if isinstance(pupil_lower, torch.Tensor) else pupil_lower,
            sigma,
            np.real(alpha_val),
            pos_d,
            phasec,
            drift_rates,
            Zcoeffmag,
            Zcoeffphase,
            np.flip(I_model, axis=-3),
            np.flip(A_model, axis=-3),
            variables,
        ]

    def res2dict(self, res):
        res_dict = dict(
            pos=res[0],
            bg=np.squeeze(res[1]),
            intensity=np.squeeze(res[2]),
            I_model=res[3],
            A_model=res[4],
            pupil1=res[5],
            pupil2=res[6],
            sigma=np.squeeze(res[7]) / np.pi,
            modulation_depth=res[8],
            obj_misalign=res[9],
            phase_dm=np.squeeze(res[10]),
            drift_rate=res[11],
            zernike_coeff_mag=np.squeeze(res[12]),
            zernike_coeff_phase=np.squeeze(res[13]),
            I_model_reverse=res[14],
            A_model_reverse=res[15],
            offset=np.min(res[3] - 2 * np.abs(res[4])),
            axial_phase=np.array(self.axial_phase_array),
            zernike_polynomial=self.pf.Zk,
            apodization=self.pf.apoid,
            cor_all=self.data.centers_all,
            cor=self.data.centers,
        )
        return res_dict


class ZernikePSFFD(PSFModel):
    """Field-dependent Zernike PSF model (spatially varying aberrations).

    Unlike ZernikePSF which fits a single set of Zernike coefficients shared
    by all beads, this model represents the Zernike coefficients as a spatial
    map (Zmap) over the field of view. Each bead's coefficients are obtained
    by bilinear interpolation from Zmap at the bead's position.

    Weight array indices:
        [0] = intensity      — scales photon count
        [1] = background     — scales offset level
        [2] = drift_rate     — scales lateral drift per z-slice
        [3] = zernike_coeff  — scales all Zernike coefficients (mag + phase)
    """

    W_INTENSITY = 0
    W_BACKGROUND = 1
    W_DRIFT = 2
    W_ZERNIKE = 3

    def __init__(self, options=None):
        super().__init__()
        self.options = options
        self.psftype = "scalar"

    def calc_initials(self, data, start_time=None):
        """Initialize all fitting variables for the field-dependent Zernike PSF model.

        Returns
        -------
        tuple[list[np.ndarray], Any]
            [positions, backgrounds, intensities, Zmap, sigma, drift] and start_time.
        """
        self.data = data
        _, rois, _, _ = self.data.get_image_data()
        options = self.options

        initial_positions = np.zeros((rois.shape[0], len(rois.shape) - 1))
        initial_backgrounds = np.array(
            np.min(gaussian_filter(rois, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True)
        )
        initial_intensities = np.sum(rois - initial_backgrounds, axis=(-2, -1), keepdims=True)
        initial_intensities = np.mean(initial_intensities, axis=1, keepdims=True)

        self._init_bead_kernel()
        num_beads = rois.shape[0]
        num_z_slices = self.bead_kernel_numpy.shape[0]

        image_size = self.data.image_size
        division_factor = options.model.division
        grid_y, grid_x = np.meshgrid(
            np.linspace(0, image_size[-2], image_size[-2] // division_factor),
            np.linspace(0, image_size[-1], image_size[-1] // division_factor),
            indexing="ij",
        )

        self._init_pupil_field("vector" if self.psftype == "vector" else "scalar")

        if options.model.const_pupilmag:
            self.max_zernike_mag_degree = 0
        else:
            self.max_zernike_mag_degree = 100

        sigma = np.ones((2,)) * self.options.model.blur_sigma * np.pi

        initial_backgrounds[initial_backgrounds < 0.1] = 0.1
        background_median = np.median(initial_backgrounds)
        median_intensity = np.lib.scimath.sqrt(np.median(initial_intensities))
        weight = [
            median_intensity * 100,
            background_median,
            1.0 / median_intensity * 40,
            20.0 / median_intensity * 40,
        ]
        self.weight = np.array(weight, dtype=np.float32)

        zernike_map = np.zeros((2, self.pf.Zk.shape[0]) + grid_x.shape, dtype=np.float32)
        zernike_map[0, 0] = 1.0 / self.weight[self.W_ZERNIKE]

        initial_backgrounds = (
            np.ones((num_beads, 1, 1, 1), dtype=np.float32)
            * np.median(initial_backgrounds, axis=0, keepdims=True)
            / self.weight[self.W_BACKGROUND]
        )
        initial_drift_rates = np.zeros((num_beads, 2), dtype=np.float32)

        self.variable_info = [
            dict(type="Nfit", id=0),
            dict(type="Nfit", id=0),
            dict(type="Nfit", id=0),
            dict(type="shared"),
            dict(type="shared"),
            dict(type="Nfit", id=0),
        ]

        if options.model.var_photon:
            per_slice_intensity = np.ones((num_beads, num_z_slices, 1, 1), dtype=np.float32) * initial_intensities
            initial_intensities_weighted = per_slice_intensity / self.weight[self.W_INTENSITY]
        else:
            initial_intensities_weighted = initial_intensities / self.weight[self.W_INTENSITY]

        self._grid_x = grid_x
        self._grid_y = grid_y

        return [
            initial_positions.astype(np.float32),
            initial_backgrounds.astype(np.float32),
            initial_intensities_weighted.astype(np.float32),
            zernike_map,
            sigma.astype(np.float32),
            initial_drift_rates,
        ], start_time

    def calc_forward_images(self, variables):
        """Compute the forward field-dependent PSF model images.

        Parameters
        ----------
        variables : list
            [positions, backgrounds, intensities, Zmap, sigma, drift]

        Returns
        -------
        torch.Tensor
            Forward model images.
        """
        positions, backgrounds, intensities, zernike_map, sigma, drift = variables

        centers = np.float32(self.data.centers)
        image_size = self.data.image_size
        Zcoeff_mag, Zcoeff_phase = interpolate_zmap(
            torch.from_numpy(zernike_map.astype(np.float32)), centers, image_size, self.batch_indices,
            self.weight[self.W_ZERNIKE],
        )

        pupil = construct_zernike_pupil(
            self.zernike_basis,
            Zcoeff_mag,
            Zcoeff_phase,
            self.pupil_aperture,
            self.apodization,
            self.symmetric_zernike_indices,
            self.max_zernike_mag_degree,
            self.options.model.symmetric_mag,
            self.weight[self.W_ZERNIKE_MAG],
            self.weight[self.W_ZERNIKE_PHASE],
        )

        positions_complex = torch.complex(
            torch.from_numpy(positions.reshape(positions.shape + (1, 1, 1)).astype(np.float32)),
            torch.tensor(0.0),
        )

        defocus_phase = -1j * 2 * np.pi * self.axial_freq_immersion * (positions_complex[:, 0] + self.defocus_positions)
        lateral_shift_phase = (
            1j * 2 * np.pi * self.lateral_freq_y * positions_complex[:, 1]
            + 1j * 2 * np.pi * self.lateral_freq_x * positions_complex[:, 2]
        )

        psf_intensity = compute_psf_intensity(
            pupil, defocus_phase, lateral_shift_phase, self.czt_params, self.normalization_factor,
            dipole_field=self.dipole_field if self.psftype == "vector" else None,
            fieldtype=self.psftype,
        )

        sigma_tensor = torch.from_numpy(sigma.astype(np.float32))
        blurred_psf = apply_blur_and_bead(
            psf_intensity, sigma_tensor, self.kspace_x_squared, self.kspace_y_squared, self.bead_kernel,
        )
        psf_fit = torch.real(blurred_psf) * torch.from_numpy(intensities.astype(np.float32)) * self.weight[self.W_INTENSITY]
        psf_fit = trim_z(psf_fit, self.bead_kernel_numpy.shape, self.data.rois.shape[-3])

        if self.options.model.estimate_drift:
            drift_scaled = drift * self.weight[self.W_DRIFT]
            psf_drifted = apply_drift(
                psf_fit,
                torch.from_numpy(drift_scaled.astype(np.float32)),
                self.drift_z_positions, self.drift_freq_x, self.drift_freq_y, self.drift_freq_z,
                self.data.skew_const if hasattr(self.data, "skew_const") and self.data.skew_const else None,
            )
            forward_images = psf_drifted + torch.from_numpy(backgrounds.astype(np.float32)) * self.weight[self.W_BACKGROUND]
        else:
            forward_images = psf_fit + torch.from_numpy(backgrounds.astype(np.float32)) * self.weight[self.W_BACKGROUND]

        return forward_images

    def postprocess(self, variables):
        positions, backgrounds, intensities, zernike_map, sigma, drift_rates = variables
        z_center = (self.defocus_positions.shape[-3] - 1) // 2

        zernike_map = zernike_map * self.weight[self.W_ZERNIKE]
        zernike_map[0, 0] = zernike_map[0, 0, 0, 0]
        centers = np.float32(self.data.centers)

        I_model, Zcoeff, pupil = self.genpsfmodel(sigma, zernike_map, centers[:, -2:])

        I_model_avg, _, _ = self.genpsfmodel(sigma, pupil=pupil, addbead=True)

        _, _, bead_centers, _ = self.data.get_image_data()
        global_positions = np.swapaxes(
            np.vstack(
                (
                    positions[:, 0] + z_center,
                    bead_centers[:, -2] - positions[:, -2],
                    bead_centers[:, -1] - positions[:, -1],
                )
            ),
            1,
            0,
        )

        return [
            global_positions.astype(np.float32),
            backgrounds * self.weight[self.W_BACKGROUND],
            intensities * self.weight[self.W_INTENSITY],
            I_model_avg,
            I_model,
            np.complex64(pupil.numpy()) if isinstance(pupil, torch.Tensor) else pupil,
            zernike_map,
            Zcoeff,
            sigma,
            drift_rates * self.weight[self.W_DRIFT],
            variables,
        ]

    def genpsfmodel(self, sigma, zernike_map=None, centers=None, pupil=None, addbead=False):
        Zcoeff = None

        if pupil is None:
            image_size = self.data.image_size
            zcoeff_mag_list = []
            zcoeff_phase_list = []

            zernike_map_tensor = torch.from_numpy(zernike_map.astype(np.float32))

            centers_tensor = torch.from_numpy(np.float32(centers))
            ny = image_size[-2]
            nx = image_size[-1]
            normalized_y = centers_tensor[:, 0] / (ny - 1) * 2 - 1
            normalized_x = centers_tensor[:, 1] / (nx - 1) * 2 - 1
            grid = torch.stack([normalized_x, normalized_y], dim=-1).unsqueeze(0).unsqueeze(0)

            for i in range(zernike_map.shape[-3]):
                z_mag = F.grid_sample(
                    zernike_map_tensor[0, i].unsqueeze(0).unsqueeze(0),
                    grid,
                    mode="bilinear",
                    align_corners=True,
                )
                z_phase = F.grid_sample(
                    zernike_map_tensor[1, i].unsqueeze(0).unsqueeze(0),
                    grid,
                    mode="bilinear",
                    align_corners=True,
                )
                zcoeff_mag_list.append(z_mag.squeeze(0).squeeze(0).squeeze(0))
                zcoeff_phase_list.append(z_phase.squeeze(0).squeeze(0).squeeze(0))

            Zcoeff1 = torch.stack(zcoeff_mag_list, dim=1).reshape(-1, zernike_map.shape[-3], 1, 1)
            Zcoeff2 = torch.stack(zcoeff_phase_list, dim=1).reshape(-1, zernike_map.shape[-3], 1, 1)
            Zcoeff = torch.stack([Zcoeff1, Zcoeff2])

            pupil_mag = torch.sum(self.zernike_basis.unsqueeze(0) * Zcoeff1, dim=-3, keepdim=True)
            pupil_mag = torch.clamp(pupil_mag, min=0)
            pupil_phase = torch.sum(self.zernike_basis.unsqueeze(0) * Zcoeff2, dim=-3, keepdim=True)
            pupil = torch.complex(
                pupil_mag * torch.cos(pupil_phase),
                pupil_mag * torch.sin(pupil_phase),
            ) * self.pupil_aperture * self.apodization

        sigma_tensor = torch.from_numpy(sigma.astype(np.float32))
        gaussian_filter_freq = torch.exp(
            -2 * sigma_tensor[1] ** 2 * self.kspace_x_squared - 2 * sigma_tensor[0] ** 2 * self.kspace_y_squared
        )
        gaussian_filter_freq = torch.complex(gaussian_filter_freq / torch.max(gaussian_filter_freq), torch.tensor(0.0))

        defocus_phase = -1j * 2 * np.pi * self.axial_freq_immersion * self.defocus_positions

        if self.psftype == "vector":
            psf_intensity = torch.zeros(
                self.defocus_positions.shape[0], pupil.shape[-2], pupil.shape[-1], dtype=torch.complex64
            )
            for dipole_component in self.dipole_field:
                pupil_func = pupil * torch.exp(defocus_phase) * dipole_component
                field = cztfunc(pupil_func, self.czt_params)
                psf_intensity = psf_intensity + field * torch.conj(field) * self.normalization_factor
        else:
            pupil_func = pupil * torch.exp(defocus_phase)
            field = cztfunc(pupil_func, self.czt_params)
            psf_intensity = field * torch.conj(field) * self.normalization_factor

        if addbead:
            I_model = torch.real(ifft3d(fft3d(psf_intensity) * gaussian_filter_freq * self.bead_kernel))
        else:
            I_model = torch.real(ifft3d(fft3d(psf_intensity) * gaussian_filter_freq))

        return I_model.numpy() if isinstance(I_model, torch.Tensor) else I_model, Zcoeff, pupil

    def res2dict(self, res):
        I_model = res[4]
        I_model_np = I_model.numpy() if isinstance(I_model, torch.Tensor) else I_model
        pupil = res[5]
        pupil_np = pupil.numpy() if isinstance(pupil, torch.Tensor) else pupil
        res_dict = dict(
            pos=res[0],
            bg=np.squeeze(res[1]),
            intensity=np.squeeze(res[2]),
            I_model_bead=res[3],
            I_model_all=I_model_np,
            pupil=np.squeeze(pupil_np),
            zernike_map=np.squeeze(res[6]),
            zernike_coeff=np.squeeze(res[7]),
            sigma=res[8] / np.pi,
            drift_rate=res[9],
            offset=np.min(I_model_np),
            zernike_polynomial=self.pf.Zk,
            apodization=self.pf.apoid,
            cor_all=self.data.centers_all,
            cor=self.data.centers,
        )
        return res_dict
