from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..core.fourier import fft3d, ifft3d, fft2d, ifft2d, cztfunc


def construct_zernike_pupil(
    zernike_basis: torch.Tensor,
    zcoeff_mag: torch.Tensor,
    zcoeff_phase: torch.Tensor,
    pupil_aperture: torch.Tensor,
    apodization: torch.Tensor,
    symmetric_zernike_indices: np.ndarray,
    max_zernike_mag_degree: int,
    symmetric_mag: bool,
    weight_magnitude: float,
    weight_phase: float,
) -> torch.Tensor:
    """Construct a complex pupil function from Zernike polynomial coefficients.

    The pupil function P(kx, ky) describes the amplitude and phase of light
    in the back focal plane of the objective. It is modeled as:

        P(kx, ky) = A(kx, ky) * exp(i * phi(kx, ky)) * aperture * apodization

    where the magnitude A and phase phi are each expressed as weighted sums
    of Zernike polynomials:

        A(kx, ky)   = clamp( sum_j  Z_j * c_mag_j  * weight_magnitude,  min=0 )
        phi(kx, ky) = sum_j  Z_j * c_phase_j * weight_phase

    Parameters
    ----------
    zernike_basis : torch.Tensor
        Zernike polynomial basis, shape (Nk, xsz, xsz).
    zcoeff_mag : torch.Tensor
        Zernike coefficients for pupil magnitude.
        Shape (2, Nk, 1, 1) for single-bead or (Nbead, Nk, 1, 1) for spatially varying.
    zcoeff_phase : torch.Tensor
        Zernike coefficients for pupil phase (same shape convention as zcoeff_mag).
    pupil_aperture : torch.Tensor
        Binary circular aperture mask (1 inside NA, 0 outside).
    apodization : torch.Tensor
        Apodization factor accounting for aplanatic and Fresnel transmission effects.
    symmetric_zernike_indices : np.ndarray
        Noll indices of rotationally symmetric Zernike modes (m=0), used when symmetric_mag=True.
    max_zernike_mag_degree : int
        Maximum radial degree for magnitude Zernike terms.
    symmetric_mag : bool
        If True, restrict magnitude to rotationally symmetric (m=0) Zernike terms only.
    weight_magnitude : float
        Normalization weight for magnitude coefficients.
    weight_phase : float
        Normalization weight for phase coefficients.

    Returns
    -------
    torch.Tensor
        Complex pupil function, shape (xsz, xsz) or (Nbead, 1, xsz, xsz).
    """
    symmetric_indices = symmetric_zernike_indices
    Nk = min((max_zernike_mag_degree + 1) * (max_zernike_mag_degree + 2) // 2, zernike_basis.shape[0])

    if zcoeff_mag.dim() == 4:
        if symmetric_mag:
            mask = symmetric_indices < Nk
            symmetric_idx = symmetric_indices[mask]
            pupil_mag = torch.sum(
                zernike_basis[symmetric_idx] * zcoeff_mag[0][symmetric_idx] * weight_magnitude, dim=0
            )
        else:
            pupil_mag = torch.sum(zernike_basis[:Nk] * zcoeff_mag[0, :Nk] * weight_magnitude, dim=0)
        pupil_phase = torch.sum(zernike_basis[3:] * zcoeff_phase[1, 3:] * weight_phase, dim=0)
    elif zcoeff_mag.dim() == 3:
        if symmetric_mag:
            mask = (symmetric_indices < Nk) & (symmetric_indices > 0)
            symmetric_idx = symmetric_indices[mask]
            if len(symmetric_idx) > 0:
                pupil_mag = torch.sum(
                    zernike_basis[symmetric_idx].unsqueeze(0) * zcoeff_mag[:, symmetric_idx] * weight_magnitude, dim=1, keepdim=True
                )
            else:
                pupil_mag = torch.zeros_like(zcoeff_mag[:, :1])
        else:
            pupil_mag = torch.sum(
                zernike_basis[1:Nk].unsqueeze(0) * zcoeff_mag[:, 1:Nk] * weight_magnitude, dim=1, keepdim=True
            )
        pupil_mag = pupil_mag + zernike_basis[0] * torch.mean(zcoeff_mag[:, 0], dim=0, keepdim=True)
        pupil_phase = torch.sum(
            zernike_basis[3:].unsqueeze(0) * zcoeff_phase[:, 3:] * weight_phase, dim=1, keepdim=True
        )
    else:
        raise ValueError(f"Unexpected zcoeff dimensions: {zcoeff_mag.dim()}")

    pupil_mag = torch.clamp(pupil_mag, min=0)
    pupil = torch.complex(
        pupil_mag * torch.cos(pupil_phase),
        pupil_mag * torch.sin(pupil_phase),
    )
    pupil = pupil * pupil_aperture * apodization
    return pupil


def construct_4pi_pupils(
    zernike_basis: torch.Tensor,
    pupil_aperture: torch.Tensor,
    apodization: torch.Tensor,
    zcoeff_mag: torch.Tensor,
    zcoeff_phase: torch.Tensor,
    symmetric_zernike_indices: torch.Tensor,
    max_zernike_mag_degree: int,
    symmetric_mag: bool,
    weight_magnitude: float,
    weight_phase: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the two pupil functions for upper and lower objectives in a 4Pi microscope.

    Each objective has its own pupil function:
        P_i(kx,ky) = A_i(kx,ky) * exp(i*phi_i(kx,ky)) * aperture * apodization

    Parameters
    ----------
    zernike_basis : torch.Tensor
        Zernike polynomial basis, shape (Nk, xsz, xsz).
    pupil_aperture : torch.Tensor
        Binary circular aperture mask.
    apodization : torch.Tensor
        Apodization factor.
    zcoeff_mag : torch.Tensor
        Magnitude Zernike coefficients, shape (2, Nk, 1, 1).
    zcoeff_phase : torch.Tensor
        Phase Zernike coefficients, shape (2, Nk, 1, 1).
    symmetric_zernike_indices : torch.Tensor
        Noll indices of rotationally symmetric Zernike modes.
    max_zernike_mag_degree : int
        Maximum radial degree for magnitude Zernike terms.
    symmetric_mag : bool
        If True, restrict magnitude to symmetric terms only.
    weight_magnitude : float
        Normalization weight for magnitude coefficients.
    weight_phase : float
        Normalization weight for phase coefficients.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (pupil_upper, pupil_lower) complex pupil functions.
    """
    Nk = min((max_zernike_mag_degree + 1) * (max_zernike_mag_degree + 2) // 2, zernike_basis.shape[0])
    mask = symmetric_zernike_indices < Nk
    symmetric_idx = symmetric_zernike_indices[mask]

    if symmetric_mag:
        pupil_mag_upper = torch.sum(zernike_basis[symmetric_idx] * zcoeff_mag[0][symmetric_idx] * weight_magnitude, dim=0)
    else:
        pupil_mag_upper = torch.sum(zernike_basis[:Nk] * zcoeff_mag[0, :Nk] * weight_magnitude, dim=0)
    pupil_mag_upper = torch.clamp(pupil_mag_upper, min=0)

    if symmetric_mag:
        pupil_mag_lower = torch.sum(zernike_basis[symmetric_idx] * zcoeff_mag[1][symmetric_idx] * weight_magnitude, dim=0)
    else:
        pupil_mag_lower = torch.sum(zernike_basis[:Nk] * zcoeff_mag[1, :Nk] * weight_magnitude, dim=0)
    pupil_mag_lower = torch.clamp(pupil_mag_lower, min=0)

    pupil_phase_upper = torch.sum(zernike_basis[1:] * zcoeff_phase[0, 1:] * weight_phase, dim=0)
    pupil_upper = torch.complex(
        pupil_mag_upper * torch.cos(pupil_phase_upper),
        pupil_mag_upper * torch.sin(pupil_phase_upper),
    ) * pupil_aperture * apodization

    pupil_phase_lower = torch.sum(zernike_basis * zcoeff_phase[1] * weight_phase, dim=0)
    pupil_lower = torch.complex(
        pupil_mag_lower * torch.cos(pupil_phase_lower),
        pupil_mag_lower * torch.sin(pupil_phase_lower),
    ) * pupil_aperture * apodization

    return pupil_upper, pupil_lower


def interpolate_zmap(
    zernike_map: torch.Tensor,
    centers: np.ndarray,
    image_size: tuple,
    batch_indices: list[int],
    weight_zernike: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate Zernike coefficient maps at bead positions using bilinear interpolation.

    Parameters
    ----------
    zernike_map : torch.Tensor
        Spatial Zernike maps, shape (2, Nk, grid_y, grid_x).
    centers : np.ndarray
        Bead center coordinates, shape (N_beads, 2).
    image_size : tuple
        Full image size (used for coordinate normalization).
    batch_indices : list[int]
        [start, end] bead indices for the current batch.
    weight_zernike : float
        Normalization weight for Zernike coefficients.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (Zcoeff_mag, Zcoeff_phase) each of shape (N_batch, Nk, 1, 1).
    """
    weighted_map = zernike_map * weight_zernike
    centers_float = np.float32(centers)
    num_zernike_terms = zernike_map.shape[-3]

    centers_batch = torch.from_numpy(centers_float[batch_indices[0]:batch_indices[1], -2:])
    ny = image_size[-2]
    nx = image_size[-1]
    normalized_y = centers_batch[:, 0] / (ny - 1) * 2 - 1
    normalized_x = centers_batch[:, 1] / (nx - 1) * 2 - 1
    grid = torch.stack([normalized_x, normalized_y], dim=-1).unsqueeze(0).unsqueeze(0)

    zcoeff_mag_list = []
    zcoeff_phase_list = []
    for i in range(num_zernike_terms):
        z_mag = F.grid_sample(
            weighted_map[0, i].unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=True,
        )
        z_phase = F.grid_sample(
            weighted_map[1, i].unsqueeze(0).unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=True,
        )
        zcoeff_mag_list.append(z_mag.squeeze(0).squeeze(0).squeeze(0))
        zcoeff_phase_list.append(z_phase.squeeze(0).squeeze(0).squeeze(0))

    Zcoeff_mag = torch.stack(zcoeff_mag_list, dim=1)
    Zcoeff_mag = Zcoeff_mag.reshape(Zcoeff_mag.shape + (1, 1))
    Zcoeff_phase = torch.stack(zcoeff_phase_list, dim=1)
    Zcoeff_phase = Zcoeff_phase.reshape(Zcoeff_phase.shape + (1, 1))

    return Zcoeff_mag, Zcoeff_phase


def compute_psf_intensity(
    pupil: torch.Tensor,
    defocus_phase: torch.Tensor,
    lateral_shift_phase: torch.Tensor,
    czt_params: tuple,
    normalization_factor: torch.Tensor,
    dipole_field: torch.Tensor | None = None,
    fieldtype: str = "scalar",
) -> torch.Tensor:
    """Compute the PSF intensity via Fourier optics propagation.

    The intensity image of a point source is computed by propagating the
    pupil function through defocus and lateral shift phase factors, then
    computing the squared magnitude of the field in the image plane
    via chirped z-transform (CZT):

        I(z, y, x) = |CZT{ P * exp(defocus_phase) * exp(lateral_shift_phase) }|^2 * norm

    For vector (dipole) imaging, the pupil function is multiplied by the
    dipole radiation pattern before propagation, and intensities from
    all dipole orientations are summed.

    Parameters
    ----------
    pupil : torch.Tensor
        Complex pupil function.
    defocus_phase : torch.Tensor
        Axial defocus phase factor.
    lateral_shift_phase : torch.Tensor
        Lateral shift phase factor.
    czt_params : tuple
        Precomputed CZT parameters (A, Bh, C) as torch tensors or numpy arrays.
    normalization_factor : torch.Tensor
        Normalization factor (0-dim complex tensor).
    dipole_field : torch.Tensor or None
        Dipole radiation pattern, shape (N_dipole, xsz, xsz). None for scalar mode.
    fieldtype : str
        "scalar" for scalar diffraction, "vector" for vectorial (dipole) model.

    Returns
    -------
    torch.Tensor
        PSF intensity, shape (Nz, Ny, Nx) or (N, Nz, Ny, Nx).
    """
    if fieldtype == "vector":
        psf_intensity = torch.zeros_like(
            torch.complex(
                torch.zeros(defocus_phase.shape[0], defocus_phase.shape[1], pupil.shape[-2], pupil.shape[-1]),
                torch.zeros(defocus_phase.shape[0], defocus_phase.shape[1], pupil.shape[-2], pupil.shape[-1]),
            )
        ).to(pupil.device)
        for dipole_component in dipole_field:
            dipole_tensor = dipole_component.to(pupil.device)
            if pupil.dim() == 4:
                pupil_func = pupil * torch.exp(defocus_phase + lateral_shift_phase) * dipole_tensor.unsqueeze(0).unsqueeze(0)
            else:
                pupil_func = pupil * torch.exp(defocus_phase + lateral_shift_phase) * dipole_tensor
            field_at_image = cztfunc(pupil_func, czt_params)
            psf_intensity = psf_intensity + field_at_image * torch.conj(field_at_image) * normalization_factor
    else:
        pupil_func = pupil * torch.exp(defocus_phase + lateral_shift_phase)
        field_at_image = cztfunc(pupil_func, czt_params)
        psf_intensity = field_at_image * torch.conj(field_at_image) * normalization_factor
    return psf_intensity


def apply_blur_and_bead(
    psf_intensity: torch.Tensor,
    sigma: torch.Tensor,
    kspace_x_squared: torch.Tensor,
    kspace_y_squared: torch.Tensor,
    bead_kernel: torch.Tensor,
) -> torch.Tensor:
    """Apply Gaussian detection blur and bead shape convolution in Fourier space.

    Both are applied as multiplicative filters on the 3D FFT of the PSF:

        I_blur = IFFT3D( FFT3D(I_psf) * bead_kernel * gaussian_filter )

    Parameters
    ----------
    psf_intensity : torch.Tensor
        Raw PSF intensity (complex, from Fourier optics).
    sigma : torch.Tensor
        Blur standard deviations [sigma_z, sigma_xy] in pixels (x pi).
    kspace_x_squared : torch.Tensor
        Squared spatial frequency map for lateral x direction (kx^2).
    kspace_y_squared : torch.Tensor
        Squared spatial frequency map for lateral y direction (ky^2).
    bead_kernel : torch.Tensor
        Fourier-space bead shape kernel.

    Returns
    -------
    torch.Tensor
        Blurred PSF (complex-valued, take real part for intensity).
    """
    gaussian_filter_3d = torch.exp(
        -2 * sigma[1] ** 2 * kspace_x_squared - 2 * sigma[0] ** 2 * kspace_y_squared
    )
    gaussian_filter_3d = torch.complex(gaussian_filter_3d / torch.max(gaussian_filter_3d), torch.tensor(0.0, device=gaussian_filter_3d.device))
    blurred_psf = ifft3d(fft3d(psf_intensity) * bead_kernel * gaussian_filter_3d)
    return blurred_psf


def apply_binning(
    blurred_psf: torch.Tensor,
    bin_factor: int,
) -> torch.Tensor:
    """Bin the PSF image by summing bin_factor x bin_factor pixel blocks.

    Simulates camera pixel binning where adjacent pixels are summed
    into a single super-pixel, increasing signal at the cost of
    spatial resolution.

    Parameters
    ----------
    blurred_psf : torch.Tensor
        PSF intensity to bin.
    bin_factor : int
        Binning factor (1 = no binning).

    Returns
    -------
    torch.Tensor
        Binned real-valued PSF intensity.
    """
    if bin_factor <= 1:
        return torch.real(blurred_psf)
    real_psf = torch.real(blurred_psf).unsqueeze(-1)
    bin_kernel = torch.ones(1, bin_factor, bin_factor, 1, 1, dtype=torch.float32, device=real_psf.device)
    binned = F.conv3d(
        real_psf.permute(0, 1, 2, 3, 4).unsqueeze(0),
        bin_kernel.permute(4, 0, 1, 2, 3),
        stride=(1, 1, bin_factor, bin_factor, 1),
        padding="same",
    )
    return binned.squeeze(0).permute(1, 2, 3, 4, 0)[..., 0]


def trim_z(
    psf: torch.Tensor,
    bead_kernel_shape: tuple,
    roi_z: int,
) -> torch.Tensor:
    """Trim the z-axis of the PSF to match the ROI size.

    The bead kernel adds padding in z; this removes the extra slices
    so the output matches the requested ROI depth.

    Parameters
    ----------
    psf : torch.Tensor
        PSF with potentially oversized z dimension.
    bead_kernel_shape : tuple
        Shape of the bead kernel (used to compute padding).
    roi_z : int
        Desired number of z slices in the output.

    Returns
    -------
    torch.Tensor
        Trimmed PSF.
    """
    Nz = psf.shape[-3]
    z_padding = (bead_kernel_shape[0] - roi_z) // 2
    return psf[..., z_padding : Nz - z_padding]


def phase_ramp(
    displacement: torch.Tensor,
    freq_x: torch.Tensor,
    freq_y: torch.Tensor,
    freq_z: torch.Tensor,
) -> torch.Tensor:
    """Compute a linear phase ramp for sub-pixel shift via Fourier shift theorem.

    For a displacement (dx, dy, dz), the image is shifted by multiplying
    the OTF by exp(i * 2pi * (kx*dx + ky*dy + kz*dz)).

    Parameters
    ----------
    displacement : torch.Tensor
        Displacement vector(s), shape (N, 2) or (N, 3).
    freq_x : torch.Tensor
        Spatial frequency coordinates in x.
    freq_y : torch.Tensor
        Spatial frequency coordinates in y.
    freq_z : torch.Tensor
        Spatial frequency coordinates in z.

    Returns
    -------
    torch.Tensor
        Complex phase ramp exp(i * 2pi * k . displacement).
    """
    if displacement.shape[1] == 2:
        shift_phase = 1j * 2 * np.pi * (freq_x * displacement[:, 1] + freq_y * displacement[:, 0])
    else:
        shift_phase = 1j * 2 * np.pi * (freq_x * displacement[:, 2] + freq_y * displacement[:, 1] + freq_z * displacement[:, 0])
    return torch.exp(shift_phase)


def apply_drift(
    psf_input: torch.Tensor,
    drift_rate: torch.Tensor,
    z_positions: torch.Tensor,
    freq_x: torch.Tensor,
    freq_y: torch.Tensor,
    freq_z: torch.Tensor,
    skew_const=None,
) -> torch.Tensor:
    """Apply a z-dependent lateral drift to the PSF via Fourier shift.

    Models sample drift where the lateral offset grows linearly with
    axial position z. For skewed geometries (e.g. light sheet), the
    drift is defined relative to a constant skew angle.

    Parameters
    ----------
    psf_input : torch.Tensor
        Input PSF intensity.
    drift_rate : torch.Tensor
        Drift rate [dx/dz, dy/dz] in pixels per z-slice.
    z_positions : torch.Tensor
        Axial position array for each z slice.
    freq_x, freq_y, freq_z : torch.Tensor
        Spatial frequency coordinates.
    skew_const : float or None
        Constant skew factor (for light sheet / deskewed geometries).

    Returns
    -------
    torch.Tensor
        Drift-corrected PSF intensity.
    """
    otf_2d = fft2d(torch.complex(psf_input, torch.zeros_like(psf_input)))
    if skew_const is not None:
        skew_factor = torch.full(drift_rate.shape, skew_const, dtype=torch.float32, device=drift_rate.device)
        skew_factor = skew_factor.reshape(skew_factor.shape + (1, 1, 1))
        drift_displacement = torch.complex(-skew_factor * z_positions + torch.round(skew_factor * z_positions), torch.tensor(0.0, device=drift_rate.device))
        shift_phase = phase_ramp(drift_displacement, freq_x, freq_y, freq_z)
    else:
        drift_rate_complex = torch.complex(drift_rate.reshape(drift_rate.shape + (1, 1, 1)), torch.tensor(0.0, device=drift_rate.device))
        drift_displacement = drift_rate_complex * z_positions
        shift_phase = phase_ramp(drift_displacement, freq_x, freq_y, freq_z)
    psf_shifted = torch.real(ifft2d(otf_2d * shift_phase))
    return psf_shifted
