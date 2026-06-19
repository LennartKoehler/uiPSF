from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import scipy.special as spf
from typing import Any, List, Optional, Tuple

from ..data_representation.ImageData import ImageData
from psflearning.io.param import RunParameters, PSFModelParams
from .. import utilities as utils
from enum import Enum

class ParameterScope(Enum):
    SHARED = 0
    NFIT = 1



class LearnableParameter:
    """A learnable parameter that stores its value as a tf.Variable internally.

    This allows:
    - In-place mutation by the optimizer (via tf.Variable.assign)
    - Differentiable access via .value (returns a tf.Tensor tracked by GradientTape)
    - Numpy access via .numpy() (returns an np.ndarray snapshot)
    """
    scope : ParameterScope
    _variable : tf.Variable
    id : int

    def __init__(self, scope : ParameterScope, value : np.ndarray | tf.Tensor, id : int):
        self.scope = scope
        self._variable = tf.Variable(value, dtype=tf.float32)
        self.id = id

    @property
    def value(self) -> tf.Tensor:
        """Return the current value as a tf.Tensor (differentiable, tracked by GradientTape).

        Uses read_value() instead of the .value attribute because tf.Variable.value
        is a method in many TF versions, and read_value() is the explicit, portable
        way to get a differentiable tensor from a Variable.
        """
        return self._variable.read_value()

    @value.setter
    def value(self, new_val : np.ndarray | tf.Tensor) -> None:
        """Update the value in-place via tf.Variable.assign."""
        self._variable.assign(new_val)

    def numpy(self) -> np.ndarray:
        """Return a numpy snapshot of the current value."""
        return self._variable.numpy()

    @property
    def variable(self) -> tf.Variable:
        """Direct access to the underlying tf.Variable (for optimizer apply_gradients)."""
        return self._variable


class LearnablePSFParameters(ABC):

    @abstractmethod
    def toTensorList(self) -> List[tf.Variable]:
        """Return the underlying tf.Variable objects (for optimizer apply_gradients).

        The returned Variables are the actual mutable storage — the optimizer
        will call .assign() on them in-place, so mutations are visible through
        this object's .value and .numpy() properties.
        """
        ...

    @abstractmethod
    def toNumpy(self) -> dict:
        """Return a snapshot of all parameters as a dict of np.ndarrays."""
        ...

    @classmethod
    @abstractmethod
    def fromTensorList(cls, tensors: List[tf.Tensor]) -> LearnablePSFParameters:
        """Construct an instance from a list of tensors/arrays."""
        ...

    @abstractmethod
    def toLearnableParameterList(self) -> List[LearnableParameter]:
        """Return the LearnableParameter objects in canonical order.

        Used by optimizers that need access to each parameter's scope (SHARED/NFIT)
        and id (batch dimension index) for batching logic.
        """
        ...

    @property
    @abstractmethod
    def n_beads(self) -> int:
        """Number of emitters/beads being fitted."""
        ...


@dataclass
class PupilField:
    """Optical quantities computed from imaging parameters and data dimensions.

    Returned by :meth:`IPSFModel.compute_pupil_field`.  Holds all
    precomputed arrays needed for pupil-based PSF modeling — frequency
    grids, CZT parameters, etc.
    """

    pupil_mask: Any
    apodization: Any
    czt_parameters: Any
    normalization_factor: Any
    z_range: Any
    frequency_x: Any
    frequency_y: Any
    frequency_z: Any
    frequency_z_medium: Any

    dipole_field: Any
    z_positions: Any
    frequency_x_view: Any
    frequency_y_view: Any
    frequency_z_view: Any
    frequency_squared_x: Any
    frequency_squared_y: Any


@dataclass
class PupilGeometry:
    """Intermediate optical quantities derived from pupil coordinates.

    Produced by :meth:`IPSFModel._compute_pupil_geometry` and consumed by
    the dipole-field, apodization, and defocus-frequency subfunctions.
    """

    radial_coordinate: Any
    kz_immersion: Any
    kz_medium: Any
    cos_immersion: Any
    cos_medium: Any
    cos_coverslip: Any
    transmission_p: Any
    transmission_s: Any
    transmission_avg: Any
    azimuthal_angle: Any
    sin_medium: Any
    pupil_x: Any
    pupil_y: Any


class IPSFModel(ABC):
    """
    Interface that ensures consistency and compatability between all old and new implementations of data classes, fitters and psfs.
    Classes implementing this interafce define a psf model/parametrization. They describe how the parameters of the psf are used to calculate a forward image
    at a specific position. They also provide initial values and postprocessing of the variables for the fitter,
    since they depend on the nature of the psf model/parametrization.

    PSF classes are stateless — all operational data is carried in a context
    object that is created by :meth:`calc_initials` and passed to
    :meth:`calc_forward_images` and :meth:`postprocess`.
    """

    @staticmethod
    def compute_pupil_field(
        data: ImageData,
        params: RunParameters,
        psf_type: str = 'vector',
        Nz: int = 21,
    ) -> PupilField:
        """Compute pupil field and related optical quantities for PSF modeling.

        Pure function — no side effects.  Replaces the old
        ``calpupilfield`` method that mutated ``self``.

        Args:
            data: Preprocessed image data providing pixel sizes and ROI dimensions.
            params: Imaging and model parameters.
            psf_type: ``'scalar'`` or ``'vector'``.
            Nz: Number of z-slices (required; previously defaulted from
                ``bead_kernel.shape[0]``).

        Returns:
            PupilField with all precomputed optical quantities.
        """
        pixel_upsampling_factor = params.model.psf.pixel_upsampling_factor
        pupil_size = params.model.psf.pupil_size
        NA = params.data.numerical_aperture
        wavelength = params.data.emission_wavelength
        n_immersion = params.data.refractive_indices.immersion
        n_medium = params.data.refractive_indices.medium
        n_coverslip = params.data.refractive_indices.coverslip

        roi_shape = data.measured_roi_images.shape
        roi_depth, roi_height, roi_width = roi_shape[-3], roi_shape[-2], roi_shape[-1]
        image_size = roi_width * pixel_upsampling_factor

        pixelsize_x = data.pixelsize_x / pixel_upsampling_factor
        pixelsize_y = data.pixelsize_y / pixel_upsampling_factor
        pixelsize_z = data.pixelsize_z

        frequency_squared_x, frequency_squared_y = IPSFModel._compute_image_frequency_grid(image_size)


        geometry = IPSFModel._compute_pupil_geometry(
            pupil_size, NA, wavelength, n_immersion, n_medium, n_coverslip,
        )

        dipole_field = IPSFModel._compute_dipole_field(geometry)

        apodization = IPSFModel._compute_apodization(
            geometry, psf_type, params.model.psf.include_apodization,
        )

        czt_parameters = IPSFModel._compute_czt_parameters(
            NA, wavelength, pupil_size, pixelsize_x, pixelsize_y, image_size,
        )

        pupil_mask = np.complex64(geometry.radial_coordinate < 1)
        normalization_factor = IPSFModel._compute_normalization(
            pupil_mask, apodization, dipole_field, czt_parameters, psf_type,
        )

        frequency_x, frequency_y, frequency_z, frequency_z_medium = (
            IPSFModel._compute_defocus_frequencies(
                geometry, NA, wavelength, pixelsize_x, pixelsize_y, pixelsize_z,
            )
        )

        z_positions, frequency_x_view, frequency_y_view, frequency_z_view = (
            IPSFModel._compute_view_grids(roi_width, roi_height, roi_depth)
        )

        z_range = np.linspace(
            -Nz / 2 + 0.5, Nz / 2 - 0.5, Nz, dtype=np.complex64,
        ).reshape((Nz, 1, 1))

        return PupilField(
            pupil_mask=pupil_mask,
            apodization=apodization,
            czt_parameters=czt_parameters,
            normalization_factor=normalization_factor,
            z_range=z_range,
            frequency_x=frequency_x,
            frequency_y=frequency_y,
            frequency_z=frequency_z,
            frequency_z_medium=frequency_z_medium,
            dipole_field=dipole_field,
            z_positions=z_positions,
            frequency_x_view=frequency_x_view,
            frequency_y_view=frequency_y_view,
            frequency_z_view=frequency_z_view,
            frequency_squared_x=frequency_squared_x,
            frequency_squared_y=frequency_squared_y,
        )

    @staticmethod
    def _compute_image_frequency_grid(image_size: int) -> tuple[np.ndarray, np.ndarray]:
        coord_range = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
        grid_x, grid_y = np.meshgrid(coord_range, coord_range)
        freq_x = grid_x / image_size
        freq_y = grid_y / image_size
        return np.float32(freq_x * freq_x), np.float32(freq_y * freq_y)


    @staticmethod
    def _compute_pupil_geometry(
        pupil_size: int,
        NA: float,
        wavelength: float,
        n_immersion: float,
        n_medium: float,
        n_coverslip: float,
    ) -> PupilGeometry:
        pupil_edge = 1.0
        krange = np.linspace(
            -pupil_edge + pupil_edge / pupil_size,
            pupil_edge - pupil_edge / pupil_size,
            pupil_size,
        )
        pupil_x, pupil_y = np.meshgrid(krange, krange)

        radial_coordinate = np.lib.scimath.sqrt(pupil_x ** 2 + pupil_y ** 2)
        kz_immersion = np.lib.scimath.sqrt(
            (n_immersion / wavelength) ** 2 - (radial_coordinate * NA / wavelength) ** 2
        )

        cos_immersion = np.lib.scimath.sqrt(1 - (radial_coordinate * NA / n_immersion) ** 2)
        cos_medium = np.lib.scimath.sqrt(1 - (radial_coordinate * NA / n_medium) ** 2)
        cos_coverslip = np.lib.scimath.sqrt(1 - (radial_coordinate * NA / n_coverslip) ** 2)
        kz_medium = n_medium / wavelength * cos_medium

        fresnel_p_medium_to_coverslip = (
            2 * n_medium * cos_medium / (n_medium * cos_coverslip + n_coverslip * cos_medium)
        )
        fresnel_s_medium_to_coverslip = (
            2 * n_medium * cos_medium / (n_medium * cos_medium + n_coverslip * cos_coverslip)
        )
        fresnel_p_coverslip_to_immersion = (
            2 * n_coverslip * cos_coverslip
            / (n_coverslip * cos_immersion + n_immersion * cos_coverslip)
        )
        fresnel_s_coverslip_to_immersion = (
            2 * n_coverslip * cos_coverslip
            / (n_coverslip * cos_coverslip + n_immersion * cos_immersion)
        )

        transmission_p = fresnel_p_medium_to_coverslip * fresnel_p_coverslip_to_immersion
        transmission_s = fresnel_s_medium_to_coverslip * fresnel_s_coverslip_to_immersion
        transmission_avg = (transmission_p + transmission_s) / 2

        azimuthal_angle = np.arctan2(pupil_y, pupil_x)
        sin_medium = radial_coordinate * NA / n_medium

        return PupilGeometry(
            radial_coordinate=radial_coordinate,
            kz_immersion=kz_immersion,
            kz_medium=kz_medium,
            cos_immersion=cos_immersion,
            cos_medium=cos_medium,
            cos_coverslip=cos_coverslip,
            transmission_p=transmission_p,
            transmission_s=transmission_s,
            transmission_avg=transmission_avg,
            azimuthal_angle=azimuthal_angle,
            sin_medium=sin_medium,
            pupil_x=pupil_x,
            pupil_y=pupil_y,
        )

    @staticmethod
    def _compute_dipole_field(geometry: PupilGeometry) -> np.ndarray:
        cos_phi = np.cos(geometry.azimuthal_angle)
        sin_phi = np.sin(geometry.azimuthal_angle)

        p_pol_vector = geometry.transmission_p * np.stack([
            geometry.cos_medium * cos_phi,
            geometry.cos_medium * sin_phi,
            -geometry.sin_medium,
        ])
        s_pol_vector = geometry.transmission_s * np.stack([
            -sin_phi,
            cos_phi,
            np.zeros(cos_phi.shape),
        ])

        dipole_field_x = cos_phi * p_pol_vector - sin_phi * s_pol_vector
        dipole_field_y = sin_phi * p_pol_vector + cos_phi * s_pol_vector
        dipole_field = np.concatenate((dipole_field_x, dipole_field_y), axis=0)
        return np.complex64(dipole_field)

    @staticmethod
    def _compute_apodization(
        geometry: PupilGeometry,
        psf_type: str,
        include_apodization: bool,
    ) -> Any:
        if not include_apodization:
            return np.complex64(1.0)
        apodization = np.lib.scimath.sqrt(geometry.cos_immersion) / geometry.cos_medium
        if psf_type == 'scalar':
            apodization = apodization * geometry.transmission_avg
        return np.complex64(apodization)

    @staticmethod
    def _compute_czt_parameters(
        NA: float,
        wavelength: float,
        pupil_size: int,
        pixelsize_x: float,
        pixelsize_y: float,
        image_size: int,
    ) -> tuple:
        pupil_freq_pixel_size = 2.0 * NA / wavelength / pupil_size
        return utils.prechirpz1(
            pupil_freq_pixel_size, pixelsize_x, pixelsize_y, pupil_size, image_size,
        )

    @staticmethod
    def _compute_normalization(
        pupil_mask: np.ndarray,
        apodization: Any,
        dipole_field: np.ndarray,
        czt_parameters: tuple,
        psf_type: str,
    ) -> np.complex64:
        pupil = tf.cast(pupil_mask * apodization, tf.complex64)
        if psf_type == 'scalar':
            propagated = utils.cztfunc1(pupil, czt_parameters)
            intensity = propagated * np.conj(propagated)
            return np.complex64(1.0 / np.sum(intensity))
        I_total = 0.0
        for h in dipole_field:
            pupil_function = pupil * h
            propagated = utils.cztfunc1(pupil_function, czt_parameters)
            I_total += propagated * tf.math.conj(propagated)
        return np.complex64(1.0 / np.sum(I_total))

    @staticmethod
    def _compute_defocus_frequencies(
        geometry: PupilGeometry,
        NA: float,
        wavelength: float,
        pixelsize_x: float,
        pixelsize_y: float,
        pixelsize_z: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frequency_x = np.complex64(geometry.pupil_x * NA / wavelength) * pixelsize_x
        frequency_y = np.complex64(geometry.pupil_y * NA / wavelength) * pixelsize_y
        frequency_z = np.complex64(geometry.kz_immersion) * pixelsize_z
        frequency_z_medium = np.complex64(geometry.kz_medium) * pixelsize_z
        return frequency_x, frequency_y, frequency_z, frequency_z_medium

    @staticmethod
    def _compute_view_grids(
        roi_width: int,
        roi_height: int,
        roi_depth: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z_positions = (
            np.linspace(0, roi_depth - 1, roi_depth, dtype=np.float32)
            .reshape(roi_depth, 1, 1)
            - roi_depth / 2
        )
        frequency_x_view = (
            np.linspace(-roi_width / 2 + 0.5, roi_width / 2 - 0.5, roi_width, dtype=np.float32)
            / roi_width
        )
        frequency_y_view = (
            np.linspace(-roi_height / 2 + 0.5, roi_height / 2 - 0.5, roi_height, dtype=np.float32)
            .reshape(roi_height, 1)
            / roi_height
        )
        frequency_z_view = (
            np.linspace(-roi_depth / 2 + 0.5, roi_depth / 2 - 0.5, roi_depth, dtype=np.float32)
            .reshape(roi_depth, 1, 1)
            / roi_depth
        )
        return z_positions, frequency_x_view, frequency_y_view, frequency_z_view


    @staticmethod
    def gen_bead_kernel(data: ImageData, params: RunParameters, isVolume: bool = False) -> tf.Tensor:
        """Generate a bead kernel for convolution with the PSF model.
        """
        pixelsize_z = data.pixelsize_z
        bead_radius = data.bead_radius
        if isVolume:
            Nz = data.measured_roi_images.shape[-3]
            pixel_upsampling_factor = 1
        else:
            Nz = data.measured_roi_images.shape[-3]+np.int32(bead_radius//pixelsize_z)*2+4
        pixel_upsampling_factor = params.model.psf.pixel_upsampling_factor

        Lx = data.measured_roi_images.shape[-1]*pixel_upsampling_factor
        pixelsize_x = data.pixelsize_x/pixel_upsampling_factor
        pixelsize_y = data.pixelsize_y/pixel_upsampling_factor

        xrange = np.linspace(-Lx/2+0.5,Lx/2-0.5,Lx)+1e-6
        zrange = np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz)
        [xx,yy,zz] = np.meshgrid(xrange,xrange,zrange)
        xx = np.swapaxes(xx,0,2)
        yy = np.swapaxes(yy,0,2)
        zz = np.swapaxes(zz,0,2)

        pkx = 1/Lx/pixelsize_x
        pky = 1/Lx/pixelsize_y
        pkz = 1/Nz/pixelsize_z
        if bead_radius>0:
            Zk0 = np.sqrt((xx*pkx)**2+(yy*pky)**2+(zz*pkz)**2)*bead_radius
            mu = 1.5
            kernel = spf.jv(mu,2*np.pi*Zk0)/(Zk0**mu)*bead_radius**3
            kernel = kernel/np.max(kernel)
            kernel = np.float32(kernel)
        else:
            kernel = np.ones((Nz,Lx,Lx),dtype=np.float32)
        return tf.complex(kernel,0.0)


    @staticmethod
    def phaseRamp(pos: tf.Tensor, pupil_field: PupilField) -> tf.Tensor:
        """Compute a phase ramp factor for the given positions."""
        if pos.shape[1]==2:
            shiftphase = 1j*2*np.pi*(pupil_field.frequency_x_view*pos[:,1]+pupil_field.frequency_y_view*pos[:,0])
        if pos.shape[1]==3:
            shiftphase = 1j*2*np.pi*(pupil_field.frequency_x_view*pos[:,2]+pupil_field.frequency_y_view*pos[:,1]+pupil_field.frequency_z_view*pos[:,0])

        return tf.exp(shiftphase)

    @staticmethod
    def applyDrift(psfin: tf.Tensor, gxy: tf.Tensor, data: ImageData, pupil_field: PupilField) -> tf.Tensor:
        """Apply drift or shift correction to a PSF using skew or linear drift."""
        otf2d = utils.fft2d(tf.complex(psfin,0.0))
        if data.skew_const:
            # drift
            sk = np.array([data.skew_const],dtype=np.float32)+np.zeros(gxy.shape,dtype=np.float32)
            sk = np.reshape(sk,sk.shape+(1,1,1))
            dxy = tf.complex(-sk*pupil_field.z_positions+tf.round(sk*pupil_field.z_positions),0.0)
            shiftphase = IPSFModel.phaseRamp(dxy, pupil_field)

        else:
            # shift
            gxy = tf.complex(tf.reshape(gxy,gxy.shape+(1,1,1)),0.0)*pupil_field.z_positions
            shiftphase = IPSFModel.phaseRamp(gxy, pupil_field)
        psf_shift = tf.math.real(utils.ifft2d(otf2d*shiftphase))

        return psf_shift

    @abstractmethod
    def calc_initials(self, data: ImageData, params: RunParameters, **kwargs) -> tuple:
        """
        Calculates the initial values for the optimizable variables.

        Args:
            data: Preprocessed image data.
            params: Imaging and model parameters.

        Returns:
            tuple of (variables, context)
        """
        raise NotImplementedError("You need to implement a 'calc_initials' method in your psf class.")

    @abstractmethod
    def calc_forward_images(self, variables, context, data=None) -> tf.Tensor:
        """
        Calculates the forward images.

        Args:
            variables: Learnable PSF variables.
            context: PSF context carrying all operational state.
            data: Optional data object.
        """
        raise NotImplementedError("You need to implement a 'calc_forward_images' method in your psf class.")

    @abstractmethod
    def postprocess(self, data, variables, context) -> Any:
        """
        Postprocesses the optimized variables. For example, normalizes the psf or calculates global positions.

        Args:
            data: Preprocessed image data.
            variables: Optimized learnable variables.
            context: PSF context carrying all operational state.
        """
        raise NotImplementedError("You need to implement a 'postprocess' method in your psf class.")
