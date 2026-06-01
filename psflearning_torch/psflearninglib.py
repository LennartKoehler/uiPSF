from __future__ import annotations

import json
import numpy as np
import h5py as h5
from omegaconf import OmegaConf
from tqdm import tqdm
from dotted_dict import DottedDict

from .core.psf_model import ZernikePSF, ZernikePSF4Pi, ZernikePSFFD
from .core.spline import psf2cspline_np
from .learning.fitter import Fitter
from .learning.loss import (
    mse_real_zernike,
    mse_real_zernike_FD,
    mse_zernike_4pi,
    mse_real_All,
    mse_real_4pi_All,
)
from .data.loader import DataLoader

PSF_DICT = dict(
    zernike=ZernikePSF,
    zernike_vector=ZernikePSF,
    zernike_FD=ZernikePSFFD,
    zernike_vector_FD=ZernikePSFFD,
)

LOSSFUN_DICT = dict(
    zernike=mse_real_zernike,
    zernike_vector=mse_real_zernike,
    zernike_FD=mse_real_zernike_FD,
    zernike_vector_FD=mse_real_zernike_FD,
)

PSF_DICT_4pi = dict(
    zernike=ZernikePSF4Pi,
)

LOSSFUN_DICT_4pi = dict(
    zernike=mse_zernike_4pi,
)


class PSFLearning:
    def __init__(self, param=None):
        self.param = param
        self.loc_FD = None

    def getpsfclass(self):
        param = self.param
        PSFtype = param.PSFtype
        channeltype = param.channeltype
        lossfun = LOSSFUN_DICT.get(PSFtype, mse_real_zernike)
        lossfunmulti = None

        if channeltype == "single":
            psfclass = PSF_DICT.get(PSFtype, ZernikePSF)
            psfmulticlass = None
        elif channeltype == "multi":
            psfclass = PSF_DICT.get(PSFtype, ZernikePSF)
            psfmulticlass = None
            lossfunmulti = mse_real_All
        elif channeltype == "4pi":
            psfclass = PSF_DICT_4pi.get(PSFtype, ZernikePSF4Pi)
            lossfun = LOSSFUN_DICT_4pi.get(PSFtype, mse_zernike_4pi)
            psfmulticlass = None
            lossfunmulti = mse_real_4pi_All

        self.psf_class = psfclass
        self.psf_class_multi = psfmulticlass
        self.loss_fun = lossfun
        self.loss_fun_multi = lossfunmulti

    def load_data(self, frange=None):
        param = self.param
        filelist = param.filelist if param.filelist else DataLoader(param).getfilelist()
        if frange:
            filelist = filelist[frange[0] : frange[1]]

        loader = DataLoader(param)
        fmt = param.format
        if fmt == ".mat":
            imagesall = loader.loadmat(filelist)
        elif fmt in (".tif", ".tiff"):
            imagesall = loader.loadtiff(filelist)
        elif fmt == ".czi":
            imagesall = loader.loadczi(filelist)
        elif fmt == ".h5":
            imagesall = loader.loadh5(filelist)
        else:
            raise TypeError(f"Unsupported format: {fmt}")

        channeltype = param.channeltype
        PSFtype = param.PSFtype
        ref_channel = param.ref_channel

        if channeltype == "4pi":
            images = np.transpose(imagesall, (1, 0, 2, 3, 4))
        elif channeltype == "multi":
            images = np.transpose(imagesall, (1, 0, 2, 3, 4))
            Nchannel = images.shape[0]
            defocus = []
            for i in range(Nchannel):
                defocus.append(param.option.multi.defocus_offset + i * param.option.multi.defocus_delay)
            defocus[0], defocus[ref_channel] = defocus[ref_channel], defocus[0]
            self.param.option.multi.defocus = defocus
            id_list = list(range(images.shape[0]))
            id_list[0], id_list[ref_channel] = id_list[ref_channel], id_list[0]
            images = images[id_list]
        else:
            images = imagesall

        if param.swapxy:
            tmp = np.zeros(images.shape[:-2] + (images.shape[-1], images.shape[-2]), dtype=np.float32)
            tmp[0:] = np.swapaxes(images[0:], -1, -2)
            images = tmp

        if (param.stage_mov_dir == "reverse") & (param.datatype == "bead"):
            images = np.flip(images, axis=-3)

        print(images.shape)
        return images

    def prep_data(self, images):
        from .data.representation import PreprocessedImageDataSingleChannel, PreprocessedImageDataMultiChannel

        param = self.param
        PSFtype = param.PSFtype
        channeltype = param.channeltype
        fov = list(param.FOV.values())
        skew_const = param.LLS.skew_const

        zstart = fov[-3]
        zend = images.shape[-3] + fov[-2]
        zstep = fov[-1]
        zind = range(zstart, zend, zstep)
        ims = np.swapaxes(images, 0, -3)
        ims = ims[zind]
        images = np.swapaxes(ims, 0, -3)

        isvolume = PSFtype == "voxel"

        if channeltype == "single":
            dataobj = PreprocessedImageDataSingleChannel(images)
        elif channeltype == "4pi":
            dataobj = PreprocessedImageDataMultiChannel(images, PreprocessedImageDataSingleChannel, is4pi=True)
        elif channeltype == "multi":
            dataobj = PreprocessedImageDataMultiChannel(images, PreprocessedImageDataSingleChannel)

        fov_param = None if fov[2] == 0 else fov[:3]
        skew_param = None if (skew_const[0] == 0.0 and skew_const[1] == 0.0) else skew_const

        dataobj.process(
            roi_size=param.roi.roi_size,
            gaus_sigma=param.roi.gauss_sigma,
            min_border_dist=list(np.array(param.roi.roi_size) // 2 + 1),
            min_center_dist=np.max(param.roi.roi_size),
            FOV=fov_param,
            max_threshold=param.roi.peak_height,
            max_kernel=param.roi.max_kernel,
            pixelsize_x=param.pixel_size.x,
            pixelsize_y=param.pixel_size.y,
            pixelsize_z=param.pixel_size.z,
            bead_radius=param.roi.bead_radius,
            modulation_period=param.fpi.modulation_period,
            plot=param.plotall,
            padPSF=False if PSFtype == "voxel" else True,
            isVolume=isvolume,
            skew_const=skew_param,
            max_bead_number=param.roi.max_bead_number,
        )

        return dataobj

    def initializepsf(self):
        param = self.param
        w = list(param.loss_weight.values())
        optionparam = param.option
        batchsize = param.batch_size

        if self.psf_class_multi is None:
            psfobj = self.psf_class(options=optionparam)
            if "vector" in param.PSFtype:
                psfobj.psftype = "vector"
        else:
            psfobj = self.psf_class_multi(self.psf_class, options=optionparam, loss_weight=w)
            if "vector" in param.PSFtype:
                psfobj.PSFtype = "vector"

        return psfobj

    def learn_psf(self, dataobj, time=None):
        param = self.param
        maxiter = param.iteration
        w = list(param.loss_weight.values())
        batchsize = param.batch_size
        optionparam = param.option
        PSFtype = param.PSFtype
        channeltype = param.channeltype
        roi_size = param.roi.roi_size

        psfobj = self.initializepsf()
        pupilfile = optionparam.model.init_pupil_file

        if pupilfile:
            f = h5.File(pupilfile, "r")
            if channeltype == "single":
                try:
                    psfobj.initial_pupil = np.array(f["res"]["pupil"])
                except KeyError:
                    pass
                try:
                    psfobj.Zoffset = np.array(f["res"]["zoffset"])
                except KeyError:
                    pass
                try:
                    psfobj.initpsf = np.array(f["res"]["I_model_reverse"]).astype(np.float32)
                except KeyError:
                    try:
                        psfobj.initpsf = np.array(f["res"]["I_model"]).astype(np.float32)
                    except KeyError:
                        pass
                try:
                    psfobj.initzcoeff = np.array(f["res"]["zernike_coeff"]).astype(np.float32)
                except KeyError:
                    pass

        fitter = Fitter(
            dataobj, psfobj, self.loss_fun,
            loss_func_single=self.loss_fun_multi,
            loss_weight=w,
            maxiter=maxiter,
            batch_size=batchsize,
        )

        _, _, centers, file_idxs = dataobj.get_image_data()
        centers = np.stack(centers)
        res, toc = fitter.learn_psf(start_time=time)

        pos = res[-1][0]
        zpos = pos[:, 0:1]
        zpos = zpos - np.mean(zpos)

        self.learning_result = res
        self.loc_result = [0, 0, 0, 0, 0, toc, {}]
        return psfobj, fitter

    def save_result(self, psfobj, dataobj, fitter):
        param = self.param
        res = self.learning_result
        locres = self.loc_result if hasattr(self, "loc_result") else [0, 0, 0, 0, 0, 0, {}]
        toc = locres[-2]

        folder = param.datapath
        savename = param.savename + "_" + param.PSFtype + "_" + param.channeltype
        res_dict = psfobj.res2dict(res)

        coeff_reverse = self._gencspline(res_dict, psfobj, keyname="I_model_reverse") if "I_model_reverse" in res_dict else np.array([])
        coeff = self._gencspline(res_dict, psfobj)

        if self.loc_FD is not None:
            locres_dict = dict(
                P=locres[0], CRLB=locres[1], LL=locres[2],
                coeff=coeff, coeff_bead=locres[3], loc=locres[-1],
                loc_FD=self.loc_FD, coeff_reverse=coeff_reverse,
            )
        else:
            locres_dict = dict(
                P=locres[0], CRLB=locres[1], LL=locres[2],
                coeff=coeff, coeff_bead=locres[3], loc=locres[-1],
                coeff_reverse=coeff_reverse,
            )

        img, _, centers, file_idxs = dataobj.get_image_data()
        img = np.stack(img)
        rois_dict = dict(
            cor=np.stack(centers), fileID=np.stack(file_idxs),
            psf_data=fitter.rois, psf_fit=fitter.forward_images,
            image_size=img.shape,
        )

        resfile = savename + ".h5"
        self._writeh5file(resfile, res_dict, locres_dict, rois_dict)
        self.result_file = resfile
        return resfile

    def _writeh5file(self, filename, res_dict, locres_dict, rois_dict):
        with h5.File(filename, "w") as f:
            f.attrs["params"] = json.dumps(OmegaConf.to_container(self.param))
            g3 = f.create_group("rois")
            g1 = f.create_group("res")
            g2 = f.create_group("locres")

            for k, v in locres_dict.items():
                if isinstance(v, dict):
                    gi = g2.create_group(k)
                    for ki, vi in v.items():
                        gi[ki] = vi
                else:
                    g2[k] = v
            for k, v in res_dict.items():
                if isinstance(v, dict):
                    gi = g1.create_group(k)
                    for ki, vi in v.items():
                        gi[ki] = vi
                else:
                    g1[k] = v
            for k, v in rois_dict.items():
                g3[k] = v

    def _gencspline(self, res_dict, psfobj, keyname="I_model"):
        param = self.param
        channeltype = param.channeltype
        coeff = []

        if channeltype == "single":
            if keyname in res_dict:
                I_model = res_dict[keyname]
                offset = np.min(I_model)
                Imd = I_model - offset
                normf = np.median(np.sum(Imd, axis=(-1, -2)))
                Imd = Imd / normf
                coeff = psf2cspline_np(Imd).astype(np.float32)
        elif channeltype == "multi":
            if "channel0" in res_dict and keyname in res_dict["channel0"]:
                Nchannel = len(psfobj.sub_psfs) if hasattr(psfobj, "sub_psfs") else 2
                I_model = []
                for i in range(Nchannel):
                    I_model.append(res_dict["channel" + str(i)][keyname])
                I_model = np.stack(I_model)
                offset = np.min(I_model)
                Imd = I_model - offset
                normf = np.max(np.median(np.sum(Imd, axis=(-1, -2)), axis=-1))
                Imd = Imd / normf
                Iall = []
                for i in range(Nchannel):
                    Iall.append(psf2cspline_np(Imd[i]))
                coeff = np.stack(Iall).astype(np.float32)
        elif channeltype == "4pi":
            if "channel0" in res_dict and keyname in res_dict["channel0"]:
                Nchannel = len(psfobj.sub_psfs) if hasattr(psfobj, "sub_psfs") else 2
                I_model = []
                A_model = []
                for i in range(Nchannel):
                    I_model.append(res_dict["channel" + str(i)][keyname])
                    if keyname == "I_model":
                        A_model.append(res_dict["channel" + str(i)]["A_model"])
                    else:
                        A_model.append(res_dict["channel" + str(i)]["A_model_reverse"])
                I_model = np.stack(I_model)
                A_model = np.stack(A_model)
                offset = np.min(I_model - 2 * np.abs(A_model))
                Imd = I_model - offset
                normf = np.max(np.median(np.sum(Imd[:, 1:-1], axis=(-1, -2)), axis=-1)) * 2.0
                Imd = Imd / normf
                Amd = A_model / normf
                IABall = []
                for i in range(Nchannel):
                    Ai = 2 * np.real(Amd[i])
                    Bi = -2 * np.imag(Amd[i])
                    IAB = [psf2cspline_np(Ai), psf2cspline_np(Bi), psf2cspline_np(Imd[i])]
                    IABall.append(np.stack(IAB))
                coeff = np.stack(IABall).astype(np.float32)

        return coeff

    def genpsf(self, f, Nz=21, xsz=21, stagepos=1.0):
        p = self.param
        self.getpsfclass()
        psfobj = self.initializepsf()
        return psfobj

    def calfwhm(self, f):
        p = self.param
        I_model = f.res[4] if len(f.res) > 4 else f.res[3]
        Imaxh = np.max(I_model) / 2
        Ix, xh, Iy, yh, Iz, zh = self._getfwhm(I_model)
        fwhmx = np.diff(xh) * p.pixel_size.x * 1e3
        fwhmy = np.diff(yh) * p.pixel_size.y * 1e3
        fwhmz = np.diff(zh) * p.pixel_size.z * 1e3
        print(f"FWHMxy: {(fwhmx[0]+fwhmy[0])/2:.2f} nm, FWHMz: {fwhmz[0]:.2f} nm")
        return fwhmx, fwhmy, fwhmz

    def _getfwhm(self, I_model):
        cor = np.unravel_index(np.argmax(I_model), I_model.shape)
        Ix = I_model[cor[0], cor[1]]
        xh = self._get1dfwhm(Ix, cor[2])
        Iy = I_model[cor[0], :, cor[2]]
        yh = self._get1dfwhm(Iy, cor[1])
        Iz = I_model[:, cor[1], cor[2]]
        zh = self._get1dfwhm(Iz, cor[0])
        return Ix, xh, Iy, yh, Iz, zh

    def _get1dfwhm(self, I, cor):
        Imaxh = np.max(I) / 2
        x1 = np.argsort(np.abs(I[:cor] - Imaxh))[0]
        if I[x1] > Imaxh:
            x1 = [x1, x1 - 1]
        else:
            x1 = [x1, x1 + 1]
        x2 = np.argsort(np.abs(I[cor:] - Imaxh))[0] + cor
        if I[x2] > Imaxh:
            x2 = [x2, x2 + 1]
        else:
            x2 = [x2, x2 - 1]
        g = np.diff(x1) / np.diff(I[x1])
        xh1 = g * (Imaxh - I[x1[0]]) + x1[0]
        x1 = np.array(x1, dtype=np.float64)
        xh1 = np.minimum(np.maximum(xh1, np.min(x1)), np.max(x1))
        g = np.diff(x2) / np.diff(I[x2])
        xh2 = g * (Imaxh - I[x2[0]]) + x2[0]
        x2 = np.array(x2, dtype=np.float64)
        xh2 = np.minimum(np.maximum(xh2, np.min(x2)), np.max(x2))
        return np.hstack([xh1, xh2])
