from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..data_representation.PreprocessedImageDataSingleChannel import PreprocessedImageDataSingleChannel
from ..psfs.PSFInterface import PSFInterface
from ..loclib import localizationlib
from typing import Any, Dict, List, Optional


class Localizer:
    """
    Handles localization given a learned PSF model.
    This class takes fitted PSF results and performs localization on image data.
    """
    def __init__(
        self,
        data: PreprocessedImageDataSingleChannel,
        psf: PSFInterface,
        rois: np.ndarray,
        forward_images: np.ndarray,
    ) -> None:
        self.data = data
        self.psf = psf
        self.rois = rois
        self.forward_images = forward_images
        self._reject_metric: Optional[List[np.ndarray]] = None
        self._minI: Optional[np.ndarray] = None

    def localize(
        self,
        res: Any,
        channeltype: str,
        usecuda: bool = True,
        initz: Optional[Any] = None,
        plot: bool = True,
        start_time: Optional[float] = None,
    ) -> Any:
        """Localize emitters using the fitted PSF model and compute rejection metrics."""
        intensity = np.abs(np.squeeze(res.intensities, axis=(-1, -2)))
        if res.intensities.dtype == 'complex64':
            intensityR = intensity
        else:
            intensityR = np.real(np.squeeze(res.intensities, axis=(-1, -2)))
        I_model = res.model_bead
        psf_fit = self.forward_images
        psf_data = self.rois
        pz = self.data.pixelsize_z

        dll = localizationlib(usecuda=usecuda)
        if channeltype == 'single':
            locres = dll.loc_ast(psf_data, I_model, pz, initz=initz, plot=plot, start_time=start_time)
            mydiff = psf_fit[:, 1:-1] - psf_data[:, 1:-1]
            mse1 = np.mean(np.square(mydiff), axis=(-3, -2, -1)) / np.mean(psf_data, axis=(-3, -2, -1))

        elif channeltype == 'multi':
            _, _, centers, _ = self.data.get_image_data()
            cor = np.stack(centers)[..., -2:]  # pyright: ignore[reportCallIssue]
            imgcenter = self.psf.imgcenter
            T = res.drift_xy
            locres = dll.loc_ast_dual(psf_data, I_model, pz, cor, imgcenter, T, initz=initz, plot=plot, start_time=start_time)
            mydiff = psf_fit[:, :, 1:-1] - psf_data[:, :, 1:-1]
            mse1 = np.mean(np.mean(np.square(mydiff), axis=(-3, -2, -1)) / np.mean(psf_data, axis=(-3, -2, -1)), axis=0)

        elif channeltype == '4pi':
            _, _, centers, _ = self.data.get_image_data()
            A_model = res.model
            cor = np.stack(centers)
            imgcenter = self.psf.imgcenter
            T = np.squeeze(res.drift_xy)
            zT = np.array([self.psf.sub_psfs[0].zT])
            locres = dll.loc_4pi(psf_data, I_model, A_model, pz, cor, imgcenter, T, zT, initz=initz, plot=plot, start_time=start_time)
            mydiff = psf_fit[:, :, :, 1:-1] - psf_data[:, :, :, 1:-1]
            mse1 = np.mean(np.mean(np.square(mydiff), axis=(-4, -3, -2, -1)) / np.mean(psf_data, axis=(-4, -3, -2, -1)), axis=0)
        else:
            raise TypeError('supported channeltype is:', str(['single', 'multi', '4pi']))

        if channeltype == 'single':
            if len(intensity.shape) < 2:
                avgI = intensity
                minI = intensityR
            else:
                avgI = np.median(intensity, axis=1)
                minI = np.min(intensityR, axis=1)
        else:
            if len(intensity.shape) < 3:
                avgI = intensity[0]
                minI = intensityR[0]
            else:
                avgI = np.median(intensity[0], axis=1)
                minI = np.min(intensityR[0], axis=1)

        if psf_data.shape[0] == 1:
            intRatio = np.array([1.0])
            mseRatio = np.array([1.0])
        else:
            intRatio = np.square(avgI - np.median(avgI)) / np.median(avgI) / avgI
            mseRatio = mse1 / np.median(mse1)
        msezRatio = locres.mse_z_ratio
        metric = [msezRatio, mseRatio, intRatio]
        label = ['relative MSE in z', 'relative MSE']
        if plot & (mseRatio.size > 1):
            fig = plt.figure(figsize=[8, 8])
            for i, val in enumerate(metric[:-1]):
                plt.plot(intRatio, val, '.')
            plt.xlabel('relative intensity')
            plt.ylabel('relative MSE')
            plt.xlim([0, 3])
            plt.ylim([0, 5])
            plt.legend(label)
            plt.grid(True)
            plt.show()

        self._reject_metric = metric
        self._minI = minI
        return locres

    def localize_smlm(
        self,
        res,
        channeltype: str,
        usecuda: bool = True,
        initz: Optional[Any] = None,
        plot: bool = True,
    ) -> Any:
        """Localize emitters in SMLM data using the fitted PSF model."""
        I_model = res.model_bead

        psf_data = self.rois
        pz = self.data.pixelsize_z

        dll = localizationlib(usecuda=usecuda)
        if channeltype == 'single':
            locres = dll.loc_ast(psf_data, I_model, pz, initz=initz, plot=plot)

        elif channeltype == 'multi':
            _, _, centers, _ = self.data.get_image_data()
            cor = np.stack(centers)
            imgcenter = self.psf.imgcenter
            T = res.drift_xy
            locres = dll.loc_ast_dual(psf_data, I_model, pz, cor, imgcenter, T, initz=initz, plot=plot)

        elif channeltype == '4pi':
            _, _, centers, _ = self.data.get_image_data()
            A_model = res.model
            cor = np.stack(centers)
            imgcenter = self.psf.imgcenter
            T = np.squeeze(res.drift_xy)
            zT = np.array([self.data.channels[0].zT])
            locres = dll.loc_4pi(psf_data, I_model, A_model, pz, cor, imgcenter, T, zT, initz=initz, plot=plot)

        else:
            raise TypeError('supported psftype is:', str(['single', 'multi', '4pi']))

        return locres

    def localize_FD(
        self,
        res: Any,
        channeltype: str,
        usecuda: bool = True,
        initz: Optional[Any] = None,
        plot: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Localize emitters frame-by-frame and compute z-bias drift over the acquisition."""
        I_model_all = self.forward_images
        psf_data = self.rois
        pz = self.data.pixelsize_z
        if len(psf_data.shape) > 3:
            Nz = psf_data.shape[-3]
        else:
            Nz = 1
        _, _, centers, _ = self.data.get_image_data()
        cor = np.stack(centers)
        dll = localizationlib(usecuda=usecuda)
        x = []
        y = []
        z = []
        for i in range(psf_data.shape[-4]):
            if channeltype == 'single':
                loci = dll.loc_ast(psf_data[i], I_model_all[i], pz, initz=initz, start_time=0)
            elif channeltype == 'multi':

                imgcenter = self.psf.imgcenter
                T = res.drift_xy
                loci = dll.loc_ast_dual(psf_data[:, i:i + 1], I_model_all[:, i], pz, cor[:, i:i + 1], imgcenter, T, initz=initz, start_time=0)

            x.append(np.squeeze(loci.positions['x']))
            y.append(np.squeeze(loci.positions['y']))
            z.append(np.squeeze(loci.positions['z']))

        xf = np.stack(x)
        yf = np.stack(y)
        zf = np.stack(z)

        zg = np.linspace(0, Nz - 1, Nz)
        if Nz > 1:
            zf = zf - np.median(zf - zg, axis=1, keepdims=True)
            zdiff = zf - zg
            xf = xf - np.median(xf, axis=1, keepdims=True)
            yf = yf - np.median(yf, axis=1, keepdims=True)
            if Nz > 4:
                zind = range(2, Nz - 2, 1)
            else:
                zind = range(0, Nz, 1)

            zdiff = zdiff - np.mean(zdiff[:, zind], axis=1, keepdims=True)
        else:
            zdiff = zf
        if plot & (Nz > 1):
            fig = plt.figure(figsize=[12, 6])
            ax = fig.add_subplot(1, 2, 1)
            plt.plot(zf.transpose(), color=(0.6, 0.6, 0.6))
            plt.plot(zg)
            ax = fig.add_subplot(1, 2, 2)
            plt.plot((zdiff).transpose(), 'k', alpha=0.1)
            plt.plot(np.median(zdiff, axis=0), color='r')
            plt.plot(zg - zg, color='k')
            ax.set_ylabel('z bias')
            ax.set_ylim([-0.1, 0.1] / np.array(pz))

        loc_FD = dict(x=xf, y=yf, z=zf)
        return loc_FD

    @property
    def reject_metric(self) -> List[np.ndarray]:
        if self._reject_metric is None:
            raise AttributeError("reject_metric not set. Run localize() first.")
        return self._reject_metric

    @property
    def minI(self) -> np.ndarray:
        if self._minI is None:
            raise AttributeError("minI not set. Run localize() first.")
        return self._minI