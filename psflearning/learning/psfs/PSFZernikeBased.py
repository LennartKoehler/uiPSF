from __future__ import annotations

from typing import Any

import numpy as np
import scipy as sp
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from .PSFZernikeBase import PSFZernikeBase
from ..data_representation.PreprocessedImageDataInterface import PreprocessedImageDataInterface
from ..loss_functions import mse_real_zernike
from .. import utilities as im
from .. import imagetools as nip

class PSFZernikeBased(PSFZernikeBase):
    """
    PSF class that uses a 3D volume to describe the PSF.
    Should only be used with single-channel data.
    """
    def __init__(self, options: Any = None) -> None:
        self.parameters = None
        self.data = None
        self.Zphase = None
        self.zT = None
        self.bead_kernel = None
        self.options = options
        self.initpupil = None
        self.defocus = np.float32(0)
        self.default_loss_func = mse_real_zernike
        self.psftype = 'scalar'
        return

    def calc_initials(self, data: PreprocessedImageDataInterface, start_time: Any = None) -> tuple[list, Any]:
        """
        Provides initial values for the optimizable varibales for the fitter class.
        """
        self.data = data
        _, rois, _, _ = self.data.get_image_data()

        options = self.options
        if options.model.with_IMM:
            init_positions = np.zeros((rois.shape[0], len(rois.shape)))
        else:
            init_positions = np.zeros((rois.shape[0], len(rois.shape)-1))

        init_backgrounds = np.array(np.min(gaussian_filter(rois, [0, 2, 2, 2]), axis=(-3, -2, -1), keepdims=True))
        init_intensitiesL = np.sum(rois - init_backgrounds, axis=(-2, -1), keepdims=True)
        init_intensities = np.mean(init_intensitiesL,axis=1,keepdims=True)

        self.gen_bead_kernel()
        N = rois.shape[0]
        Nz = rois.shape[-3]
        Lx = rois.shape[-1]

        if self.psftype=='vector':
            self.calpupilfield('vector')
        else:
            self.calpupilfield('scalar')
        if options.model.const_pupilmag:
            self.n_max_mag = 0
        else:
            self.n_max_mag = 100


        init_backgrounds[init_backgrounds<0.1] = 0.1
        bgmean = np.median(init_backgrounds)
        wI = np.lib.scimath.sqrt(np.median(init_intensities))

        self.weight = {"intensity" : wI * 100,
                       "background" : bgmean,
                       "drift" : 1 / wI * 40,
                       "zernikeMagnitude" : 0.5 / wI * 40,
                       "zernikePhase" : 0.5 / wI * 40}

        sigma = np.ones((2,))*self.options.model.blur_sigma*np.pi
        self.init_sigma = sigma
        # init_Zcoeff = np.zeros((2,self.Zk.shape[0],1,1))
        # init_Zcoeff[:,0,0,0] = [1,0]/self.weight["zernikePhase"]

        init_Zcoeff_magnitude = np.zeros((self.Zk.shape[0],1,1))
        init_Zcoeff_phase = np.zeros((self.Zk.shape[0],1,1))

        init_Zcoeff_magnitude[0,0,0] = 1/self.weight["zernikePhase"]
        # init_Zcoeff_phase[0,0,0] = 0

        init_backgrounds = np.ones((N,1,1,1),dtype = np.float32)*np.median(init_backgrounds,axis=0, keepdims=True) / self.weight["background"]
        gxy = np.zeros((N,2),dtype=np.float32)
        gI = np.ones((N,Nz,1,1),dtype = np.float32)*init_intensities

        self.varinfo = [dict(type='Nfit',id=0),
                   dict(type='Nfit',id=0),
                   dict(type='Nfit',id=0),
                   dict(type='shared'),
                   dict(type='shared'),
                   dict(type='Nfit',id=0)]

        if options.model.var_photon:
            init_Intensity = gI/self.weight["intensity"]
        else:
            init_Intensity = init_intensities / self.weight["intensity"]
        return [init_positions.astype(np.float32),
                init_backgrounds.astype(np.float32),
                init_Intensity.astype(np.float32),
                init_Zcoeff_magnitude.astype(np.float32),
                init_Zcoeff_phase.astype(np.float32),
                sigma.astype(np.float32),
                gxy], start_time

    def calc_forward_images(self, variables: list) -> tf.Tensor:
        """
        Calculate forward images from the current guess of the variables.
        Shifting is done by Fourier transform and applying a phase ramp.
        """
        pos, backgrounds, intensities, zCoeff_magnitude, zCoeff_phase, sigma, gxy = variables

        if self.initpupil is not None:
            pupil = self.initpupil
        else:
            pupil = self.compute_pupil_from_zernike(
                zCoeff_magnitude, zCoeff_phase, self.weight["zernikeMagnitude"], self.weight["zernikePhase"]
            )

        pos = tf.complex(tf.reshape(pos, pos.shape + (1, 1, 1)), 0.0)

        phase_z, phase_xy = self.__compute_phase(
                pos,
                self.Zrange,
                self.defocus,
                self.kx,
                self.ky,
                self.kz,
                self.kz_med)

        I_res = self.propagate_pupil(pupil, phase_z, phase_xy)

        bin = self.options.model.bin
        if not self.options.model.var_blur:
            sigma = self.init_sigma

        I_blur = self.apply_blur_3d(I_res, sigma, use_bead_kernel=True)
        I_blur_bin = self.bin_image_3d(I_blur, bin)
        psf_fit = I_blur_bin[..., 0]
        psf_fit = self.trim_z_padding(psf_fit)

        if self.options.model.estimate_drift:
            gxy = gxy * self.weight["drift"]
            psf_shift = self.applyDrift(psf_fit, gxy)
            forward_images = psf_shift * intensities * self.weight["intenity"] + backgrounds * self.weight["background"]
        else:
            forward_images = psf_fit * intensities * self.weight["intensity"] + backgrounds * self.weight["background"]

        return forward_images


    def genpsfmodel(self, sigma: np.ndarray, Zcoeff_magnitude: tf.Tenosor = None, Zcoeff_phase: tf.Tensor = None, pupil: Any = None, addbead: bool = False) -> tuple[tf.Tensor, Any]:
        """Generate a PSF model from Zernike coefficients or a given pupil function."""
        if pupil is None:
            pupil_mag = tf.reduce_sum(self.Zk * Zcoeff_magnitude, axis=0)
            pupil_mag = tf.math.maximum(pupil_mag, 0)
            pupil_phase = tf.reduce_sum(self.Zk * Zcoeff_phase, axis=0)
            pupil = self.magnitude_phase_to_pupil(pupil_mag, pupil_phase)

        phiz = -1j * 2 * np.pi * self.kz * (self.Zrange + self.defocus)
        phase_z = tf.exp(phiz)
        I_res = self.propagate_pupil(pupil, phase_z)

        bin = self.options.model.bin
        I_blur = self.apply_blur_3d(I_res, sigma, use_bead_kernel=addbead)

        if len(I_blur.shape) == 5:
            I_model = self.bin_image_3d(I_blur, bin)
        else:
            kernel = np.ones((bin, bin, 1, 1), dtype=np.float32)
            I_model = tf.nn.convolution(I_blur, kernel, strides=(1, bin, bin, 1), padding='SAME', data_format='NHWC')
        I_model = I_model[..., 0]

        return I_model, pupil

    def postprocess(self, variables: list) -> list:
        """
        Applies postprocessing to the optimized variables. In this case calculates
        real positions in the image from the positions in the roi. Also, normalizes
        psf and adapts intensities and background accordingly.
        """
        positions, backgrounds, intensities, Zcoeff_magnitude, Zcoeff_phase, sigma,gxy = variables
        z_center = (self.Zrange.shape[-3] - 1) // 2

        Zcoeff_magnitude = Zcoeff_magnitude * self.weight["zernikeMagnitude"]
        Zcoeff_phase = Zcoeff_phase * self.weight["zernikePhase"]

        bin = self.options.model.bin
        positions[:,1:] = positions[:,1:]/bin
        if self.initpupil is not None:
            pupil = self.initpupil
            I_model, _ = self.genpsfmodel(sigma,pupil=pupil)
            I_model_bead, _ = self.genpsfmodel(sigma,pupil=pupil,addbead=True)
        else:
            I_model, pupil = self.genpsfmodel(sigma,Zcoeff_magnitude=Zcoeff_magnitude, Zcoeff_phase=Zcoeff_phase)
            I_model_bead, _ = self.genpsfmodel(sigma,Zcoeff_magnitude=Zcoeff_magnitude, Zcoeff_phase=Zcoeff_phase, addbead=True)

        images, _, centers, _ = self.data.get_image_data()
        original_shape = images.shape[-3:]
        Nbead = centers.shape[0]
        if positions.shape[1]>3:
            global_positions = np.swapaxes(np.vstack((positions[:,0]+z_center,positions[:,1],centers[:,-2]-positions[:,-2],centers[:,-1]-positions[:,-1])),1,0)
        else:
            global_positions = np.swapaxes(np.vstack((positions[:,0]+z_center,centers[:,-2]-positions[:,-2],centers[:,-1]-positions[:,-1])),1,0)

        return [global_positions.astype(np.float32),
                backgrounds*self.weight["background"],
                intensities*self.weight["intensity"],
                I_model_bead,
                I_model,
                np.complex64(pupil),
                Zcoeff_magnitude,
                Zcoeff_phase,
                sigma,
                gxy*self.weight["drift"],
                np.flip(I_model,axis=-3),
                variables]

    def res2dict(self, res: list) -> dict[str, Any]:
        """Convert optimization results to a dictionary with named fields."""
        res_dict = dict(pos=res[0],
                        bg=np.squeeze(res[1]),
                        intensity=np.squeeze(res[2]),
                        I_model_bead = res[3],
                        I_model = res[4],
                        pupil = res[5],
                        zernike_coeff = np.squeeze(res[6]),
                        sigma = np.squeeze(res[7])/np.pi,
                        drift_rate=res[8],
                        I_model_reverse = res[9],
                        offset=np.min(res[4]),
                        zernike_polynomial = self.Zk,
                        apodization = self.apoid,
                        cor_all = self.data.centers_all,
                        cor = self.data.centers)

        return res_dict
