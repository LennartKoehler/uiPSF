from __future__ import annotations

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from psflearning.learning.psfs.PSFVolumeBased4pi import PSFVolumeBased4pi
from .FitterInterface import FitterInterface
from ..data_representation.PreprocessedImageDataSingleChannel import PreprocessedImageDataSingleChannel
from ..psfs.PSFInterface import PSFInterface
from ..optimizers import OptimizerABC
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from .Localizer import Localizer


class PSFLearner(FitterInterface):
    """
    This class combines data, psf, optimizer and loss function and defines the actual learning process.
    Responsible only for learning a PSF model, not for localization.
    """
    def __init__(
        self,
        data: PreprocessedImageDataSingleChannel,
        psf: PSFInterface,
        optimizer: OptimizerABC,
        loss_func: Callable,
        loss_func_single: Optional[Callable] = None,
        loss_weight: Optional[Any] = None,
    ) -> None:
        self.data = data
        self.psf: PSFInterface = psf
        self.loss_func = loss_func
        self.loss_func_single = loss_func_single
        self.optimizer = optimizer

        self.rois: Optional[np.ndarray] = None
        self.forward_images: Optional[np.ndarray] = None
        self.mu: float = 1
        self.rate: float = 1.1
        self.loss_weight = loss_weight
        return

    def __objective(
        self,
        variables: List[np.ndarray],
        mu: float,
        ind: Optional[List[int]] = None,
    ) -> Any:
        """
        Define the objective that should be optimized.
        Basically asks the psf to calculate forward_images and combines those with the data
        and the loss function to calculate the loss.
        """
        if ind is None:
            ind = [0, variables[0].shape[0]]
        self.psf.ind = ind
        forward_images = self.psf.calc_forward_images(variables)
        if self.loss_func_single:
            psfnorm = [None] * len(self.psf.sub_psfs)
            if hasattr(self.psf.sub_psfs[0], 'psfnorm'):
                for i, psf in enumerate(self.psf.sub_psfs):
                    psfnorm[i] = psf.psfnorm
                loss = self.loss_func(forward_images, self.rois[:, ind[0]:ind[1]], self.loss_func_single, variables, mu, self.loss_weight, psfnorm)
            else:
                loss = self.loss_func(forward_images, self.rois[:, ind[0]:ind[1]], self.loss_func_single, variables, mu, self.loss_weight)
        else:
            if hasattr(self.psf, 'psfnorm'):
                loss = self.loss_func(forward_images, self.rois[ind[0]:ind[1]], variables, mu, self.loss_weight, self.psf.psfnorm)
            else:
                loss = self.loss_func(forward_images, self.rois[ind[0]:ind[1]], variables, mu, self.loss_weight)
        return loss

    def learn_psf(
        self,
        variables: Optional[List[np.ndarray]] = None,
        start_time: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], float]:
        """
        Defines the procedure of the psf learning. Just asks the psf to calculate initial
        values (if not provided), runs the optimization and uses the psf object to do postprocessing.
        """
        if variables is None:
            variables, start_time = self.psf.calc_initials(self.data, start_time=start_time)

        _, rois, _, _ = self.data.get_image_data()

        try:
            self.rois = np.stack(rois)
        except ValueError:
            raise RuntimeError("At this point each channel must have same number of rois and allow np.stack.")

        pbar = tqdm(total=self.optimizer.maxiter + 50, desc='3/6: learning', bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt}, {postfix[0]}{postfix[2][loss]:>4.5f}, {postfix[1]}{postfix[2][time]:>4.2f}s", postfix=["current loss: ", "total time: ", dict(loss=0, time=start_time)])

        variables = self.optimizer.minimize(self.__objective, variables, self.psf.varinfo, pbar)
        toc = pbar.postfix[-1]['time']
        pbar.close()
        self.psf.ind = [0, variables[0].shape[0]]
        self.forward_images = self.psf.calc_forward_images(variables).numpy()

        variables = self.psf.postprocess(variables)

        return variables, toc

    def relearn(
        self,
        initres: List[np.ndarray],
        localizer: Localizer,
        channeltype: str,
        threshold: np.ndarray,
        start_time: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], Optional[float]]:
        """Re-learn PSF after rejecting outliers based on a metric threshold."""
        metric = localizer.reject_metric
        minI = localizer.minI
        mask = self.__getMask(metric, minI, threshold)
        delete_id = np.where(~mask)
        print('outlier id:', str(delete_id[0]))

        if not ((delete_id[0].size > 0) & (delete_id[0].size < mask.size)):
            return initres, start_time
        else:
            if channeltype == 'single':
                _, rois, centers, file_idxs = self.data.get_image_data()
                cor = centers[mask, :]
                fid = file_idxs[mask]
                self.data.rois = rois[mask]
                self.data.centers = cor
                self.data.file_idxs = fid
                _, rois, _, _ = self.data.get_image_data()
                print(f"rois shape channel : {rois.shape}")
                var = initres[-1]
                var[0] = initres[-1][0][mask]
                var[1] = initres[-1][1][mask]
                var[2] = initres[-1][2][mask]
                var[-1] = initres[-1][-1][mask]
                res, toc = self.learn_psf(var, start_time=start_time)
                return res, toc

            else:
                _, rois, centers, file_idxs = self.data.get_image_data()
                for i in range(len(self.data.channels)):
                    self.data.channels[i].rois = rois[i][mask]
                    self.data.channels[i].centers = centers[i][mask, :]
                    self.data.channels[i].file_idxs = file_idxs[i][mask]
                _, rois, centers, _ = self.data.get_image_data()
                num_channels = len(rois)

                cor_ref = np.concatenate((centers[0][:, -2:], np.ones((centers[0].shape[0], 1))), axis=1)
                self.psf.cor_ref_channel = np.stack([cor_ref] * (num_channels - 1)).astype(np.float32)
                self.psf.cor_other_channels = (np.stack(centers[1:])[..., -2:]).astype(np.float32)
                for i in range(len(rois)):
                    print(f"rois shape channel {i}: {rois[i].shape}")
                var = initres[-1]
                var[0] = initres[-1][0][mask]
                var[1] = initres[-1][1][:, mask]
                var[2] = initres[-1][2][:, mask]
                var[-2] = initres[-1][-2][:, mask]
                if channeltype == '4pi':
                    var[3] = initres[-1][3][:, mask]
                    if self.psf.psftype != PSFVolumeBased4pi:
                        var[-4] = initres[-1][-4][:, mask]

                res, toc = self.learn_psf(var, start_time=start_time)
        return res, toc

    def relearn_smlm(
        self,
        initres: List[np.ndarray],
        localizer: Localizer,
        channeltype: str,
        threshold: np.ndarray,
        start_time: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], Optional[float]]:
        """Re-learn PSF for SMLM data after rejecting outliers based on fit quality and position."""
        pos = initres[-1][0]
        intensity = np.squeeze(initres[-1][2])
        xp = pos[:, -1]
        yp = pos[:, -2]
        zp = pos[:, 0]
        psf_data = self.rois
        psf_fit = self.forward_images
        mydiff = psf_fit - psf_data
        mse1 = np.mean(np.square(mydiff), axis=(-2, -1)) / np.mean(psf_data, axis=(-2, -1))
        if channeltype == 'multi' or channeltype == '4pi':
            intensity = np.min(intensity, axis=0, keepdims=False)
            mse1 = np.sum(mse1, axis=0, keepdims=False)
        a = threshold[0]
        if self.psf.options.insitu.backgroundROI:
            mask = (xp > np.quantile(xp, 1 - a)) & (xp < np.quantile(xp, a)) & (yp > np.quantile(yp, 1 - a)) & (yp < np.quantile(yp, a)) & (zp < np.quantile(zp, a))
        else:
            mask = (xp > np.quantile(xp, 1 - a)) & (xp < np.quantile(xp, a)) & (yp > np.quantile(yp, 1 - a)) & (yp < np.quantile(yp, a)) & (zp > np.quantile(zp, 1 - a)) & (zp < np.quantile(zp, a))

        mask = mask & (mse1 < np.quantile(mse1, threshold[1]))
        mask = mask & (intensity > 0)
        delete_id = np.where(~mask)
        print('outlier percentage:', 1 - np.sum(mask) / mask.size)

        if (delete_id[0].size > 0) & (delete_id[0].size < mask.size):
            if channeltype == 'single':
                _, rois, centers, frames = self.data.get_image_data()
                cor = centers[mask, :]
                fid = frames[mask]
                self.data.rois = rois[mask]
                self.data.centers = cor
                self.data.frames = fid
                _, rois, _, _ = self.data.get_image_data()
                print(f"rois shape channel : {rois.shape}")
                var = initres[-1]
                var[0] = initres[-1][0][mask]
                var[1] = initres[-1][1][mask]
                var[2] = initres[-1][2][mask]
                zw = self.psf.zweight[mask]
                if self.psf.options.insitu.backgroundROI:
                    bgroi = self.psf.options.insitu.backgroundROI
                    maskcor = (cor[:, -1] > bgroi[2]) & (cor[:, -1] < bgroi[3]) & (cor[:, -2] > bgroi[0]) & (cor[:, -2] < bgroi[1])
                    try:
                        zmin = np.quantile(var[0][maskcor, 0], 0.05)
                    except Exception:
                        zmin = np.quantile(var[0][:, 0], 0.002)
                    maskz = var[0][:, 0] < zmin
                    zw[maskz] = 0.0
                    var[0][maskz, 0] = 0.0

                self.psf.zweight = zw
                res, toc = self.learn_psf(var, start_time=start_time)
            else:
                _, rois, centers, frames = self.data.get_image_data()
                for i in range(len(self.data.channels)):
                    self.data.channels[i].rois = rois[i][mask]
                    self.data.channels[i].centers = centers[i][mask, :]
                    self.data.channels[i].frames = frames[i][mask]
                _, rois, centers, _ = self.data.get_image_data()
                num_channels = len(rois)

                cor_ref = np.concatenate((centers[0], np.ones((centers[0].shape[0], 1))), axis=1)
                self.psf.cor_ref_channel = np.stack([cor_ref] * (num_channels - 1)).astype(np.float32)
                self.psf.cor_other_channels = np.stack(centers[1:]).astype(np.float32)

                for i in range(len(rois)):
                    print(f"rois shape channel {i}: {rois[i].shape}")
                var = initres[-1]
                var[0] = initres[-1][0][mask]
                var[1] = initres[-1][1][:, mask]
                var[2] = initres[-1][2][:, mask]
                if channeltype == '4pi':
                    var[3] = initres[-1][3][:, mask]

                res, toc = self.learn_psf(var, start_time=start_time)
        else:
            res = initres
            toc = start_time
        return res, toc

    def __getMask(self, metric, minI, threshold):
        mask = metric[0] > -1
        for i, val in enumerate(metric):
            mask = (val < threshold[i]) & mask
        mask = (minI > 0) & mask
        return mask