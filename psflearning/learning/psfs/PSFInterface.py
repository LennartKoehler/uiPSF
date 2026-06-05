from __future__ import annotations

from abc import ABC, abstractmethod
import pickle

import numpy as np
import tensorflow as tf
import scipy.special as spf
from typing import Any, List

from ..data_representation.PreprocessedImageDataInterface import PreprocessedImageDataInterface
from .. import utilities as im
from .. import imagetools as nip
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

    @abstractmethod
    def filter_by_mask(self, mask: np.ndarray) -> LearnablePSFParameters:
        """Return a new instance with per-bead (NFIT) parameters filtered by *mask*.

        Shared parameters are kept unchanged.  *mask* is a boolean array
        of shape ``(n_beads,)``.
        """
        ...



class PSFInterface(ABC):
    """
    Interface that ensures consistency and compatability between all old and new implementations of data classes, fitters and psfs.
    Classes implementing this interafce define a psf model/parametrization. They describe how the parameters of the psf are used to calculate a forward image
    at a specific position. They also provide initial values and postprocessing of the variables for the fitter,
    since they depend on the nature of the psf model/parametrization.
    """

    data: PreprocessedImageDataInterface
    bead_kernel: Any
    options: Any
    aperture: Any
    apoid: Any
    paramxy: Any
    normf: Any
    Zrange: Any
    kx: Any
    ky: Any
    kz: Any
    kz_med: Any
    Zk: Any
    zv: Any
    kxv: Any
    kyv: Any
    kzv: Any
    kspace: Any
    kspace_x: Any
    kspace_y: Any
    spherical_terms: Any
    dipole_field: Any
    defocus: Any
    psftype: str
    imgcenter: Any
    sub_psfs: Any
    psfnorm: Any

    @abstractmethod
    def calc_initials(self, data: PreprocessedImageDataInterface) -> list:
        """
        Calculates the initial values for the optimizable variables.
        """
        raise NotImplementedError("You need to implement a 'calc_initials' method in your psf class.")

    @abstractmethod
    def calc_forward_images(self, variables) -> tf.Tensor:
        """
        Calculates the forward images.
        """
        raise NotImplementedError("You need to implement a 'calc_forward_images' method in your psf class.")

    @abstractmethod
    def postprocess(self, variables) -> Any:
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

    def calpupilfield(self, fieldtype: str = 'vector', Nz: int | None = None, datatype: str = 'bead') -> None:
        """
        Calculate pupil field and related optical quantities for PSF modeling.
        """
        if Nz is None:
            Nz = self.bead_kernel.shape[0]
        assert Nz is not None
        bin = self.options.model.bin
        Lx = self.data.rois.shape[-1]*bin
        Ly = self.data.rois.shape[-2]*bin
        Lz = self.data.rois.shape[-3]
        xsz =self.options.model.pupilsize

        xrange = np.linspace(-Lx/2+0.5,Lx/2-0.5,Lx)
        [xx,yy] = np.meshgrid(xrange,xrange)
        pkx = xx/Lx
        pky = yy/Lx
        self.kspace = np.float32(pkx*pkx+pky*pky)
        self.kspace_x = np.float32(pkx*pkx)
        self.kspace_y = np.float32(pky*pky)

        pixelsize_x = self.data.pixelsize_x/bin
        pixelsize_y = self.data.pixelsize_y/bin
        NA = self.options.imaging.NA
        emission_wavelength = self.options.imaging.emission_wavelength
        nimm = self.options.imaging.RI.imm
        nmed = self.options.imaging.RI.med
        ncov = self.options.imaging.RI.cov
        n_max = self.options.model.n_max
        Zk = im.genZern1(n_max,xsz)

        n1 = np.array(range(-1,n_max,2))
        self.spherical_terms = (n1+1)*(n1+2)//2

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
        if self.options.model.with_apoid:
            apoid = np.lib.scimath.sqrt(cos_imm)/cos_med
            if fieldtype=='scalar':
                apoid=apoid*Tavg
        else:
            apoid = 1

        kpixelsize = 2.0*NA/emission_wavelength/xsz
        self.paramxy = im.prechirpz1(kpixelsize,pixelsize_x,pixelsize_y,xsz,Lx)

        self.aperture = np.complex64(kr<1)
        pupil = self.aperture*apoid
        pupil = tf.cast(pupil,tf.complex64)
        if fieldtype=='scalar':
            psfA = im.cztfunc1(pupil,self.paramxy)
            self.normf = np.complex64(1/np.sum(psfA*np.conj(psfA)))
        else:
            I_res = 0.0
            for h in self.dipole_field:
                PupilFunction = pupil*h
                psfA = im.cztfunc1(PupilFunction,self.paramxy)
                I_res += psfA*tf.math.conj(psfA)
            self.normf = np.complex64(1/np.sum(I_res))
        self.Zrange = np.linspace(-Nz/2+0.5,Nz/2-0.5,Nz,dtype=np.complex64).reshape((Nz,1,1))
        self.kx = np.complex64(xx*NA/emission_wavelength)*pixelsize_x
        self.ky = np.complex64(yy*NA/emission_wavelength)*pixelsize_y
        self.kz = np.complex64(kz)*self.data.pixelsize_z
        self.kz_med = np.complex64(kz_med)*self.data.pixelsize_z
        self.k = np.complex64(nmed/emission_wavelength)*self.data.pixelsize_z
        self.apoid = np.complex64(apoid)
        self.nimm = nimm
        self.nmed = nmed
        self.Zk = np.float32(Zk)

        Lx = self.data.rois.shape[-1]
        Ly = self.data.rois.shape[-2]
        Lz = self.data.rois.shape[-3]

        self.zv = np.linspace(0,Lz-1,Lz,dtype=np.float32).reshape(Lz,1,1)-Lz/2
        self.kxv = np.linspace(-Lx/2+0.5,Lx/2-0.5,Lx,dtype=np.float32)/Lx
        self.kyv = (np.linspace(-Ly/2+0.5,Ly/2-0.5,Ly,dtype=np.float32).reshape(Ly,1))/Ly
        self.kzv = (np.linspace(-Lz/2+0.5,Lz/2-0.5,Lz,dtype=np.float32).reshape(Lz,1,1))/Lz


    def calnorm(self, pupil: tf.Tensor) -> tf.Tensor:
        """
        Calculate the normalization factor from a pupil function.
        """
        psfA = im.cztfunc1(pupil,self.paramxy)
        normf = tf.math.real(tf.reduce_sum(psfA*tf.math.conj(psfA)))
        return normf

    def gen_bead_kernel(self, isVolume: bool = False) -> None:
        """
        Generate a bead kernel for convolution with the PSF model.
        """
        pixelsize_z = self.data.pixelsize_z
        bead_radius = self.data.bead_radius
        if isVolume:
            Nz = self.data.rois.shape[-3]
            bin = 1
        else:
            Nz = self.data.rois.shape[-3]+np.int32(bead_radius//pixelsize_z)*2+4
            bin = self.options.model.bin

        Lx = self.data.rois.shape[-1]*bin
        pixelsize_x = self.data.pixelsize_x/bin
        pixelsize_y = self.data.pixelsize_y/bin

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


    def applyPhaseRamp(self, img: tf.Tensor, shiftvec: tf.Tensor) -> tf.Tensor:
        """
        Applies a frequency ramp as a phase factor according to the shiftvec to a Fourier transform to shift the image.
        Identical to implementation in InverseModelling. Just removed if-statement (0) that does not make sense for me and prevent my code to work correctly.
        img: input Fourier transform tensor
        shiftvec: real-space shifts
        """
        res = im.totensor(img)
        myshape = im.shapevec(res)
        ShiftDims = int(shiftvec.shape[-1])
        for d in range(1, ShiftDims+1):
            myshifts = shiftvec[..., -d]
            for ed in range(len(myshape) - len(myshifts.shape)):
                myshifts = tf.expand_dims(myshifts,-1)
            res = res * tf.exp(tf.complex(im.totensor(0.0), 2.0 * np.pi * myshifts * nip.ramp1D(myshape[-d], ramp_dim = -d, freq='ftfreq')))
        return res

    def phaseRamp(self, pos: tf.Tensor) -> tf.Tensor:
        """
        Compute a phase ramp factor for the given positions.
        """
        if pos.shape[1]==2:
            shiftphase = 1j*2*np.pi*(self.kxv*pos[:,1]+self.kyv*pos[:,0])
        if pos.shape[1]==3:
            shiftphase = 1j*2*np.pi*(self.kxv*pos[:,2]+self.kyv*pos[:,1]+self.kzv*pos[:,0])

        return tf.exp(shiftphase)

    def applyDrift(self, psfin: tf.Tensor, gxy: tf.Tensor) -> tf.Tensor:
        """
        Apply drift or shift correction to a PSF using skew or linear drift.
        """
        otf2d = im.fft2d(tf.complex(psfin,0.0))
        if self.data.skew_const:
            # drift
            sk = np.array([self.data.skew_const],dtype=np.float32)+np.zeros(gxy.shape,dtype=np.float32)
            sk = np.reshape(sk,sk.shape+(1,1,1))
            dxy = tf.complex(-sk*self.zv+tf.round(sk*self.zv),0.0)
            shiftphase = self.phaseRamp(dxy)

        else:
            # shift
            gxy = tf.complex(tf.reshape(gxy,gxy.shape+(1,1,1)),0.0)*self.zv
            shiftphase = self.phaseRamp(gxy)
        psf_shift = tf.math.real(im.ifft2d(otf2d*shiftphase))

        return psf_shift

    def psf2IAB(self, ROIs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert PSF intensity measurements at three phases to IAB model components.
        """
        G = np.zeros(ROIs.shape, dtype = np.complex64)
        G[:,0] = ROIs[:,0]*np.exp(-2*np.pi/3*1j)+ROIs[:,1]+ROIs[:,2]*np.exp(2*np.pi/3*1j)
        G[:,1] = np.sum(ROIs,axis=1)
        G[:,2] = ROIs[:,0]*np.exp(2*np.pi/3*1j)+ROIs[:,1]+ROIs[:,2]*np.exp(-2*np.pi/3*1j) # G[:,2] = np.conj(G[:,0])
        # solving above equations for ROIs and redefine it as O
        O = np.zeros(ROIs.shape, dtype = np.complex64)
        O[:,0] = 1/3*(G[:,0]*np.exp(2*np.pi/3*1j)+G[:,1]+G[:,2]*np.exp(-2*np.pi/3*1j))
        O[:,1] = 1/3*np.sum(G,axis=1)
        O[:,2] = 1/3*(G[:,0]*np.exp(-2*np.pi/3*1j)+G[:,1]+G[:,2]*np.exp(2*np.pi/3*1j)) # O[:,2] = np.conj(O[:,0])
        # above derivation is purely based on the definition of FFT and the fact that cos(2pi/3) and cos(4pi/3) are all equal to -0.5.
        # it is true for PSF at any 3 phases, however, if the 3 phases are exactly at [-2pi/3, 0, 2pi/3], then G can be used to represent the complex IAB model, where
        I = np.real(G[:,1])/3
        A = G[:,0]/3
        B = G[:,2]/3 # B = np.conj(A)

        a = np.squeeze(np.sum(np.real(A[0]),axis = (-1,-2)))
        b = np.squeeze(np.sum(np.imag(A[0]),axis = (-1,-2)))

        y1 = np.squeeze(np.sum((ROIs[:,2]-ROIs[:,0])/np.sqrt(3),axis = (-1,-2)))
        y2 = np.squeeze(np.sum(ROIs[:,1]-np.sum(ROIs,axis = 1)/3,axis = (-1,-2)))

        q = np.squeeze(1j*(a*y1-b*y2) + (a*y2+b*y1))
        if len(q.shape)>1:
            phi = 1*np.median(np.angle(q),axis=1)
        else:
            phi = 1*np.median(np.angle(q))


        return I, A, B, phi
