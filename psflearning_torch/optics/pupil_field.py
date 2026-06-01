from __future__ import annotations

import numpy as np
import torch

from ..core.zernike import genZern1
from ..core.fourier import prechirpz, cztfunc


class PupilField:
    """Precomputed optical field quantities for PSF forward modeling.

    This class computes and caches all coordinate systems, Zernike polynomial
    bases, Fresnel transmission coefficients, dipole radiation patterns, and
    Fourier transform parameters needed to evaluate the PSF forward model.

    The physical model follows Fourier optics: the PSF is the squared magnitude
    of the inverse Fourier transform (implemented via chirped z-transform) of
    the pupil function, which lies in the back focal plane of the objective.

    Attributes
    ----------
    kspace : np.ndarray
        Squared normalized radial frequency (kx^2 + ky^2) in pupil plane.
    kspace_x, kspace_y : np.ndarray
        Squared frequency components (kx^2, ky^2) for anisotropic blur.
    spherical_terms : np.ndarray
        Noll indices of rotationally symmetric (m=0) Zernike polynomials.
    dipole_field : np.ndarray
        Dipole radiation pattern components for vectorial PSF model.
    apoid : np.ndarray
        Apodization factor: sqrt(cos_theta_imm) / cos_theta_med * Fresnel_T
        (scales pupil to account for aplanatic factor and interface transmission).
    aperture : np.ndarray
        Binary circular aperture: 1 inside the NA circle, 0 outside.
    paramxy : tuple
        Precomputed CZT parameters for pupil-to-image propagation.
    normf : complex
        Normalization factor so that the PSF peak integrates to 1.
    Zrange : np.ndarray
        Axial position array for defocus computation, shape (Nz, 1, 1).
    kx, ky : np.ndarray
        Spatial frequencies in the pupil plane, scaled by NA/lambda * pixel_size.
    kz : np.ndarray
        Axial spatial frequency in immersion medium, kz = sqrt((n/lambda)^2 - k_r^2).
    kz_med : np.ndarray
        Axial spatial frequency in sample medium.
    k : np.ndarray
        Total wavenumber in sample medium (n_med / lambda), scaled by pixel_size_z.
    Zk : np.ndarray
        Zernike polynomial basis, shape (Nk, pupil_size, pupil_size).
    nimm : float
        Refractive index of immersion medium.
    nmed : float
        Refractive index of sample medium.
    zv : np.ndarray
        Axial coordinate array for drift computation.
    kxv, kyv, kzv : np.ndarray
        Frequency coordinate arrays for drift phase ramp computation.
    """

    def __init__(self):
        self.kspace: np.ndarray | None = None
        self.kspace_x: np.ndarray | None = None
        self.kspace_y: np.ndarray | None = None
        self.spherical_terms: np.ndarray | None = None
        self.dipole_field: np.ndarray | None = None
        self.apoid: np.ndarray | None = None
        self.aperture: np.ndarray | None = None
        self.paramxy: tuple | None = None
        self.normf: complex | None = None
        self.Zrange: np.ndarray | None = None
        self.kx: np.ndarray | None = None
        self.ky: np.ndarray | None = None
        self.kz: np.ndarray | None = None
        self.kz_med: np.ndarray | None = None
        self.k: np.ndarray | None = None
        self.Zk: np.ndarray | None = None
        self.nimm: float | None = None
        self.nmed: float | None = None
        self.zv: np.ndarray | None = None
        self.kxv: np.ndarray | None = None
        self.kyv: np.ndarray | None = None
        self.kzv: np.ndarray | None = None

    def compute(
        self,
        options,
        data,
        fieldtype: str = "scalar",
        Nz: int | None = None,
        device: str = "cpu",
    ):
        """Compute all optical field quantities for the given imaging configuration.

        This sets up the complete coordinate system and physical model for PSF
        computation, following scalar or vectorial Fourier optics.

        The optical path is: point source → sample medium → coverslip →
        immersion medium → objective → back focal plane (pupil) → image plane.

        Parameters
        ----------
        options : object
            Configuration options (imaging parameters, model settings).
        data : object
            Experimental data with ROI dimensions and pixel sizes.
        fieldtype : str
            "scalar" for scalar diffraction model, "vector" for vectorial dipole model.
        Nz : int or None
            Number of axial slices (defaults to data ROI depth).
        device : str
            Torch device for computation.
        """
        bin_val = options.model.bin
        Lx = data.rois.shape[-1] * bin_val
        Ly = data.rois.shape[-2] * bin_val
        Lz = data.rois.shape[-3]
        xsz = options.model.pupilsize

        if Nz is None:
            Nz = Lz

        # --- Spatial frequency maps for anisotropic Gaussian blur ---
        xrange = np.linspace(-Lx / 2 + 0.5, Lx / 2 - 0.5, Lx)
        xx, yy = np.meshgrid(xrange, xrange)
        normalized_kx = xx / Lx
        normalized_ky = yy / Lx
        self.kspace = np.float32(normalized_kx * normalized_kx + normalized_ky * normalized_ky)
        self.kspace_x = np.float32(normalized_kx * normalized_kx)
        self.kspace_y = np.float32(normalized_ky * normalized_ky)

        # --- Physical imaging parameters ---
        pixelsize_x = data.pixelsize_x / bin_val
        pixelsize_y = data.pixelsize_y / bin_val
        NA = options.imaging.NA
        emission_wavelength = options.imaging.emission_wavelength
        nimm = options.imaging.RI.imm
        nmed = options.imaging.RI.med
        ncov = options.imaging.RI.cov
        n_max = options.model.n_max
        self.nimm = nimm
        self.nmed = nmed

        # --- Zernike polynomial basis ---
        Zk = genZern1(n_max, xsz)
        # Noll indices of rotationally symmetric (m=0) Zernike modes:
        # these are n=1,3,5,... with odd n → j = (n+1)(n+2)/2
        n_odd = np.arange(-1, n_max, 2)
        self.spherical_terms = ((n_odd + 1) * (n_odd + 2)) // 2

        # --- Pupil-plane coordinates and propagation vectors ---
        # The pupil is sampled on a grid where radius = NA/lambda in k-space,
        # normalized so that the NA circle has radius 1.
        pupilradius = 1
        krange = np.linspace(
            -pupilradius + pupilradius / xsz,
            pupilradius - pupilradius / xsz,
            xsz,
        )
        xx, yy = np.meshgrid(krange, krange)
        kr = np.lib.scimath.sqrt(xx**2 + yy**2)

        # Axial propagation constant in immersion medium:
        #   kz = sqrt((n_imm/lambda)^2 - (k_r * NA/lambda)^2)
        kz = np.lib.scimath.sqrt(
            (nimm / emission_wavelength) ** 2 - (kr * NA / emission_wavelength) ** 2
        )

        # --- Fresnel transmission coefficients at interfaces ---
        # Three-layer model: medium (nmed) → coverslip (ncov) → immersion (nimm)
        # cos(theta) in each layer via Snell's law: n*sin(theta) = NA*k_r
        cos_imm = np.lib.scimath.sqrt(1 - (kr * NA / nimm) ** 2)
        cos_med = np.lib.scimath.sqrt(1 - (kr * NA / nmed) ** 2)
        cos_cov = np.lib.scimath.sqrt(1 - (kr * NA / ncov) ** 2)
        kz_med = nmed / emission_wavelength * cos_med

        # Fresnel coefficients at medium→coverslip interface
        Fresnel_p_medcov = 2 * nmed * cos_med / (nmed * cos_cov + ncov * cos_med)
        Fresnel_s_medcov = 2 * nmed * cos_med / (nmed * cos_med + ncov * cos_cov)
        # Fresnel coefficients at coverslip→immersion interface
        Fresnel_p_covimm = 2 * ncov * cos_cov / (ncov * cos_imm + nimm * cos_cov)
        Fresnel_s_covimm = 2 * ncov * cos_cov / (ncov * cos_cov + nimm * cos_imm)
        # Total transmission through both interfaces
        Tp = Fresnel_p_medcov * Fresnel_p_covimm
        Ts = Fresnel_s_medcov * Fresnel_s_covimm
        Tavg = (Tp + Ts) / 2

        # --- Vectorial (dipole) radiation pattern ---
        # The dipole field decomposes the radiation from an oscillating dipole
        # into p- and s-polarized components, each transmitted through the
        # interface stack. The resulting vector field in the pupil plane has
        # six components (x,y,z for two orthogonal dipole orientations).
        phi = np.arctan2(yy, xx)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        sin_med = kr * NA / nmed

        # p-polarized unit vector (in plane of incidence)
        pvec = Tp * np.stack([cos_med * cos_phi, cos_med * sin_phi, -sin_med])
        # s-polarized unit vector (perpendicular to plane of incidence)
        svec = Ts * np.stack([-sin_phi, cos_phi, np.zeros_like(cos_phi)])
        # Field from x-dipole and y-dipole in pupil coordinates
        hx = cos_phi * pvec - sin_phi * svec
        hy = sin_phi * pvec + cos_phi * svec
        h = np.concatenate((hx, hy), axis=0)
        self.dipole_field = np.complex64(h)

        # --- Apodization factor ---
        # Combines the aplanatic factor sqrt(cos_theta_imm) with the
        # interface transmission and the cos_theta_med correction.
        # For scalar model: also includes the average Fresnel transmission Tavg.
        if options.model.with_apoid:
            apoid = np.lib.scimath.sqrt(cos_imm) / cos_med
            if fieldtype == "scalar":
                apoid = apoid * Tavg
        else:
            apoid = np.ones_like(kr)

        # --- Chirped z-transform parameters for pupil→image propagation ---
        # The CZT computes the Fourier transform from pupil coordinates to
        # image coordinates with arbitrary sampling, enabling pixel-accurate
        # PSF evaluation without zero-padding.
        kpixelsize = 2.0 * NA / emission_wavelength / xsz
        self.paramxy = prechirpz(kpixelsize, pixelsize_x, pixelsize_y, xsz, Lx)

        # --- Circular aperture (NA cutoff) ---
        self.aperture = np.complex64(kr < 1)
        pupil = self.aperture * apoid
        pupil_t = torch.from_numpy(pupil.astype(np.complex64)).to(device)

        # --- Normalization factor ---
        # Computed so that the integrated PSF intensity equals 1 for a
        # perfect (unaberrated) pupil function.
        if fieldtype == "scalar":
            psfA = cztfunc(pupil_t, self.paramxy)
            self.normf = np.complex64(1.0 / float(torch.sum(psfA * torch.conj(psfA)).real))
        else:
            I_res = torch.zeros_like(psfA)
            for hi in self.dipole_field:
                h_t = torch.from_numpy(hi).to(device)
                PupilFunction = pupil_t * h_t
                psfA = cztfunc(PupilFunction, self.paramxy)
                I_res = I_res + psfA * torch.conj(psfA)
            self.normf = np.complex64(1.0 / float(torch.sum(I_res).real))

        # --- Defocus and spatial frequency arrays ---
        # Axial position array for computing defocus phase
        self.Zrange = np.linspace(
            -Nz / 2 + 0.5, Nz / 2 - 0.5, Nz, dtype=np.complex64
        ).reshape(Nz, 1, 1)

        # Spatial frequencies scaled by pixel size for phase computation:
        #   defocus phase:  exp(-i * 2pi * kz * z)
        #   lateral shift:  exp( i * 2pi * (kx * dx + ky * dy))
        self.kx = np.complex64(xx * NA / emission_wavelength) * pixelsize_x
        self.ky = np.complex64(yy * NA / emission_wavelength) * pixelsize_y
        self.kz = np.complex64(kz) * data.pixelsize_z
        self.kz_med = np.complex64(kz_med) * data.pixelsize_z
        self.k = np.complex64(nmed / emission_wavelength) * data.pixelsize_z
        self.apoid = np.complex64(apoid)
        self.Zk = np.float32(Zk)

        # --- Drift phase ramp coordinates ---
        # Used for z-dependent lateral drift correction via Fourier shift theorem
        Lx_roi = data.rois.shape[-1]
        Ly_roi = data.rois.shape[-2]
        Lz_roi = data.rois.shape[-3]

        self.zv = (
            np.linspace(0, Lz_roi - 1, Lz_roi, dtype=np.float32).reshape(
                Lz_roi, 1, 1
            )
            - Lz_roi / 2
        )
        self.kxv = (
            np.linspace(-Lx_roi / 2 + 0.5, Lx_roi / 2 - 0.5, Lx_roi, dtype=np.float32)
            / Lx_roi
        )
        self.kyv = (
            np.linspace(-Ly_roi / 2 + 0.5, Ly_roi / 2 - 0.5, Ly_roi, dtype=np.float32).reshape(
                Ly_roi, 1
            )
            / Ly_roi
        )
        self.kzv = (
            np.linspace(-Lz_roi / 2 + 0.5, Lz_roi / 2 - 0.5, Lz_roi, dtype=np.float32).reshape(
                Lz_roi, 1, 1
            )
            / Lz_roi
        )

    def calnorm(self, pupil: torch.Tensor) -> torch.Tensor:
        """Compute the normalization integral for a given pupil function.

        Returns the total intensity (sum of |PSF|^2) for normalization purposes.

        Parameters
        ----------
        pupil : torch.Tensor
            Complex pupil function.

        Returns
        -------
        torch.Tensor
            Real-valued normalization factor.
        """
        psfA = cztfunc(pupil, self.paramxy)
        return torch.real(torch.sum(psfA * torch.conj(psfA)))


def compute_pupil_field(
    options, data, fieldtype="scalar", Nz=None, device="cpu"
) -> PupilField:
    """Create and compute a PupilField object with all precomputed quantities.

    Convenience factory function for PupilField.

    Parameters
    ----------
    options : object
        Configuration options.
    data : object
        Experimental data.
    fieldtype : str
        "scalar" or "vector" PSF model.
    Nz : int or None
        Number of axial slices.
    device : str
        Torch device.

    Returns
    -------
    PupilField
        Fully initialized PupilField object.
    """
    pf = PupilField()
    pf.compute(options, data, fieldtype, Nz, device)
    return pf
