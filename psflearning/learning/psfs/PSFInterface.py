from __future__ import annotations

from abc import ABC, abstractmethod
import pickle

import numpy as np
import tensorflow as tf
import scipy.special as spf
from typing import Any, List

from ..data_representation.PreprocessedImageDataInterface import PreprocessedImageDataInterface
from psflearning.io.param import OptionParams
from .. import utilities as im
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




class PSFInterface(ABC):
    """
    Interface that ensures consistency and compatability between all old and new implementations of data classes, fitters and psfs.
    Classes implementing this interafce define a psf model/parametrization. They describe how the parameters of the psf are used to calculate a forward image
    at a specific position. They also provide initial values and postprocessing of the variables for the fitter,
    since they depend on the nature of the psf model/parametrization.
    """
    def __init__(self, params: OptionParams):
        self.params: OptionParams = params
        pass
    bead_kernel: Any
    pupil_mask: Any
    apodization: Any
    czt_parameters: Any
    normalization_factor: Any
    z_range: Any
    frequency_x: Any
    frequency_y: Any
    frequency_z: Any
    frequency_z_medium: Any
    zernike_polynomial_basis: Any
    z_positions: Any
    frequency_x_view: Any
    frequency_y_view: Any
    frequency_z_view: Any
    frequency_squared_x: Any
    frequency_squared_y: Any
    spherical_noll_indices: Any
    dipole_field: Any
    defocus: Any
    psf_type: str

    @abstractmethod
    def calc_initials(self, data: PreprocessedImageDataInterface) -> list:
        """
        Calculates the initial values for the optimizable variables.
        """
        raise NotImplementedError("You need to implement a 'calc_initials' method in your psf class.")

    @abstractmethod
    def calc_forward_images(self, variables, data=None) -> tf.Tensor:
        """
        Calculates the forward images.
        """
        raise NotImplementedError("You need to implement a 'calc_forward_images' method in your psf class.")

    @abstractmethod
    def postprocess(self, data, variables) -> Any:
        """
        Postprocesses the optimized variables. For example, normalizes the psf or calculates global positions.
        """
        raise NotImplementedError("You need to implement a 'postprocess' method in your psf class.")

    def save(self, filename: str) -> None:
        """
        Save object to file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> PSFInterface:
        """
        Load object from file.
        """
        with open(filename, "rb") as f:
            self = pickle.load(f)
        return self

    def calpupilfield(self, data: PreprocessedImageDataInterface, fieldtype: str = 'vector', Nz: int | None = None, datatype: str = 'bead') -> None:
        """
        Calculate pupil field and related optical quantities for PSF modeling.
        """
        if Nz is None:
            Nz = self.bead_kernel.shape[0]
        assert Nz is not None
        bin = self.params.model.bin
        Lx = data.measured_roi_images.shape[-1]*bin
        Ly = data.measured_roi_images.shape[-2]*bin
        Lz = data.measured_roi_images.shape[-3]
        xsz =self.params.model.pupilsize

        xrange = np.linspace(-Lx/2+0.5,Lx/2-0.5,Lx)
        [xx,yy] = np.meshgrid(xrange,xrange)
        pkx = xx/Lx
        self.frequency_squared_x = np.float32(pkx*pkx)
        pky = yy/Lx
        self.frequency_squared_y = np.float32(pky*pky)

        pixelsize_x = data.pixelsize_x/bin
        pixelsize_y = data.pixelsize_y/bin
        NA = self.params.imaging.NA
        emission_wavelength = self.params.imaging.emission_wavelength
        nimm = self.params.imaging.RI.imm
        nmed = self.params.imaging.RI.med
        ncov = self.params.imaging.RI.cov
        n_max = self.params.model.n_max
        Zk = im.genZern1(n_max,xsz)

        n1 = np.array(range(-1,n_max,2))
        self.spherical_noll_indices = (n1+1)*(n1+2)//2

        pupilradius = 1
        krange = np.linspace(-pupilradius+pupilradius/xsz,pupilradius-pupilradius/xsz,xsz)
        [xx,yy] = np.meshgrid(krange,krange)
        kr = np.lib.scimath.sqrt(xx**2+yy**2)
        kz = np.lib.scimath.sqrt((nimm/emission_wavelength)**2-(kr*NA/emission_wavelength)**2)

        cos_imm = np.lib.scimath.sqrt(1-(kr*NA/nimm)**2)
        cos_med = np.lib.scimath.sqrt(1-(kr*NA/nmed)**2)
        cos_cov = np.lib.scimath.sqrt(1-(kr*NA/ncov)**2)
        kz_med = nmed/emission_wavelength*cos_med
        FresnelPmedcov = 2*nmed*cos_med/(nmed*cos_cov+ncov*cos_med)
        FresnelSmedcov = 2*nmed*cos_med/(nmed*cos_med+ncov*cos_cov)
        FresnelPcovimm = 2*ncov*cos_cov/(ncov*cos_imm+nimm*cos_cov)
        FresnelScovimm = 2*ncov*cos_cov/(ncov*cos_cov+nimm*cos_imm)
        Tp = FresnelPmedcov*FresnelPcovimm
        Ts = FresnelSmedcov*FresnelScovimm
        Tavg = (Tp+Ts)/2

        phi = np.arctan2(yy,xx)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        sin_med = kr*NA/nmed

        pvec = Tp*np.stack([cos_med*cos_phi,cos_med*sin_phi,-sin_med])
        svec = Ts*np.stack([-sin_phi,cos_phi,np.zeros(cos_phi.shape)])

        hx = cos_phi*pvec-sin_phi*svec
        hy = sin_phi*pvec+cos_phi*svec
        h = np.concatenate((hx,hy),axis=0)
        self.dipole_field = np.complex64(h)
        if self.params.model.with_apoid:
            apoid = np.lib.scimath.sqrt(cos_imm)/cos_med
            if fieldtype=='scalar':
                apoid=apoid*Tavg
        else:
            apoid = 1

        kpixelsize = 2.0*NA/emission_wavelength/xsz
        self.czt_parameters = im.prechirpz1(kpixelsize,pixelsize_x,pixelsize_y,xsz,Lx)

        self.pupil_mask = np.complex64(kr<1)
        pupil = self.pupil_mask*apoid
        pupil = tf.cast(pupil,tf.complex64)
        if fieldtype=='scalar':
            propagated_psf_amplitude = im.cztfunc1(pupil,self.czt_parameters)
            self.normalization_factor = np.complex64(1/np.sum(propagated_psf_amplitude*np.conj(propagated_psf_amplitude)))
        else:
            I_res = 0.0
            for h in self.dipole_field:
                PupilFunction = pupil*h
                propagated_psf_amplitude = im.cztfunc1(PupilFunction,self.czt_parameters)
                I_res += propagated_psf_amplitude*tf.math.conj(propagated_psf_amplitude)
            self.normalization_factor = np.complex64(1/np.sum(I_res))
        self.z_range = np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz,dtype=np.complex64).reshape((Nz,1,1))
        self.frequency_x = np.complex64(xx*NA/emission_wavelength)*pixelsize_x
        self.frequency_y = np.complex64(yy*NA/emission_wavelength)*pixelsize_y
        self.frequency_z = np.complex64(kz)*data.pixelsize_z
        self.frequency_z_medium = np.complex64(kz_med)*data.pixelsize_z
        self.apodization = np.complex64(apoid)
        self.zernike_polynomial_basis = np.float32(Zk)

        Lx = data.measured_roi_images.shape[-1]
        Ly = data.measured_roi_images.shape[-2]
        Lz = data.measured_roi_images.shape[-3]

        self.z_positions = np.linspace(0,Lz-1,Lz,dtype=np.float32).reshape(Lz,1,1)-Lz/2
        self.frequency_x_view = np.linspace(-Lx/2+0.5,Lx/2-0.5,Lx,dtype=np.float32)/Lx
        self.frequency_y_view = (np.linspace(-Ly/2+0.5,Ly/2-0.5,Ly,dtype=np.float32).reshape(Ly,1))/Ly
        self.frequency_z_view = (np.linspace(-Lz/2+0.5,Lz/2-0.5,Lz,dtype=np.float32).reshape(Lz,1,1))/Lz


    def gen_bead_kernel(self, data: PreprocessedImageDataInterface, isVolume: bool = False) -> None:
        """
        Generate a bead kernel for convolution with the PSF model.
        """
        pixelsize_z = data.pixelsize_z
        bead_radius = data.bead_radius
        if isVolume:
            Nz = data.measured_roi_images.shape[-3]
            bin = 1
        else:
            Nz = data.measured_roi_images.shape[-3]+np.int32(bead_radius//pixelsize_z)*2+4
            bin = self.params.model.bin

        Lx = data.measured_roi_images.shape[-1]*bin
        pixelsize_x = data.pixelsize_x/bin
        pixelsize_y = data.pixelsize_y/bin

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
        self.bead_kernel = tf.complex(kernel,0.0)

        return


    def phaseRamp(self, pos: tf.Tensor) -> tf.Tensor:
        """
        Compute a phase ramp factor for the given positions.
        """
        if pos.shape[1]==2:
            shiftphase = 1j*2*np.pi*(self.frequency_x_view*pos[:,1]+self.frequency_y_view*pos[:,0])
        if pos.shape[1]==3:
            shiftphase = 1j*2*np.pi*(self.frequency_x_view*pos[:,2]+self.frequency_y_view*pos[:,1]+self.frequency_z_view*pos[:,0])

        return tf.exp(shiftphase)

    def applyDrift(self, psfin: tf.Tensor, gxy: tf.Tensor, data: PreprocessedImageDataInterface) -> tf.Tensor:
        """
        Apply drift or shift correction to a PSF using skew or linear drift.
        """
        otf2d = im.fft2d(tf.complex(psfin,0.0))
        if data.skew_const:
            # drift
            sk = np.array([data.skew_const],dtype=np.float32)+np.zeros(gxy.shape,dtype=np.float32)
            sk = np.reshape(sk,sk.shape+(1,1,1))
            dxy = tf.complex(-sk*self.z_positions+tf.round(sk*self.z_positions),0.0)
            shiftphase = self.phaseRamp(dxy)

        else:
            # shift
            gxy = tf.complex(tf.reshape(gxy,gxy.shape+(1,1,1)),0.0)*self.z_positions
            shiftphase = self.phaseRamp(gxy)
        psf_shift = tf.math.real(im.ifft2d(otf2d*shiftphase))

        return psf_shift
