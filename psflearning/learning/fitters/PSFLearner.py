from __future__ import annotations

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from psflearning.learning.psfs.PSFZernikeBased import ZernikePSFResult, ZernikePSFVariables

from ..loclib import LocalizationResult
from .FitterInterface import FitterInterface
from ..data_representation.PreprocessedImageDataSingleChannel import PreprocessedImageDataSingleChannel
from ..psfs.PSFInterface import LearnablePSFParameters, PSFInterface
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
        self.optimizer = optimizer

        self.mu: float = 1
        self.rate: float = 1.1
        self.loss_weight = loss_weight
        return

    def __objective(
        self,
        variables: LearnablePSFParameters,
        mu: float,
        ind: Optional[List[int]] = None,
    ) -> Any:
        """
        Define the objective that should be optimized.
        Basically asks the psf to calculate forward_images and combines those with the data
        and the loss function to calculate the loss.
        """
        if ind is None:
            ind = [0, variables.n_beads]
        forward_images = self.psf.calc_forward_images(variables)
        loss = self.loss_func(forward_images, self.data.rois[ind[0]:ind[1]], variables, mu, self.loss_weight)
        return loss

    def learn_psf(
        self,
        variables: ZernikePSFVariables,
        start_time: Optional[float] = None,
    ) -> Tuple[ZernikePSFResult, np.ndarray, float]:
        """
        Defines the procedure of the psf learning. Just asks the psf to calculate initial
        values (if not provided), runs the optimization and uses the psf object to do postprocessing.
        """

        pbar = tqdm(total=self.optimizer.maxiter + 50, desc='3/6: learning', bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt}, {postfix[0]}{postfix[2][loss]:>4.5f}, {postfix[1]}{postfix[2][time]:>4.2f}s", postfix=["current loss: ", "total time: ", dict(loss=0, time=start_time)])

        variables = self.optimizer.minimize(self.__objective, variables, pbar)
        toc = pbar.postfix[-1]['time']
        pbar.close()

        # self.psf.ind = [0, variables.n_beads]
        forward_images = self.psf.calc_forward_images(variables).numpy()

        psfResult = self.psf.postprocess(variables)

        return psfResult, forward_images, toc

    def remove_outliers_single(
        self,
        res: ZernikePSFResult,
        locres: LocalizationResult,
        threshold: list,
    ) -> Optional[ZernikePSFVariables]:
        """Remove outlier ROIs for single-channel data based on rejection metrics.

        Filters the internal data object and learning variables in-place.
        Returns the filtered variables if any outliers were removed,
        or *None* if no outliers were found (or all would be removed).
        """
        metric, minI= create_reject_metric(
            res,
            locres,
            psf_modeled,
            self.data)
        mask = get_mask(metric, minI, threshold)
        delete_id = np.where(~mask)
        print('outlier id:', str(delete_id[0]))

        if not ((delete_id[0].size > 0) & (delete_id[0].size < mask.size)):
            return None

        _, rois, centers, file_idxs = self.data.get_image_data()
        self.data.rois = rois[mask]
        self.data.centers = centers[mask, :]
        self.data.file_idxs = file_idxs[mask]
        _, rois, _, _ = self.data.get_image_data()
        print(f"rois shape channel : {rois.shape}")

        return res.filter_by_mask(mask)


def get_mask(metric, minI, threshold):
    mask = metric[0] > -1
    for i, val in enumerate(metric):
        mask = (val < threshold[i]) & mask
    mask = (minI > 0) & mask
    return mask

def create_reject_metric(
    res: ZernikePSFResult,
    locres: LocalizationResult,
    psf_modeled: np.ndarray,
    psf_data: np.ndarray
) -> Tuple[List[np.ndarray], Any]:

    intensity = np.abs(np.squeeze(res.intensities, axis=(-1, -2)))
    if res.intensities.dtype == 'complex64':
        intensityR = intensity
    else:
        intensityR = np.real(np.squeeze(res.intensities, axis=(-1, -2)))
    mydiff = psf_modeled[:, 1:-1] - psf_data[:, 1:-1]
    mse1 = np.mean(np.square(mydiff), axis=(-3, -2, -1)) / np.mean(psf_data, axis=(-3, -2, -1))

    if len(intensity.shape) < 2:
        avgI = intensity
        minI = intensityR
    else:
        avgI = np.median(intensity, axis=1)
        minI = np.min(intensityR, axis=1)

    if psf_data.shape[0] == 1:
        intRatio = np.array([1.0])
        mseRatio = np.array([1.0])
    else:
        intRatio = np.square(avgI - np.median(avgI)) / np.median(avgI) / avgI
        mseRatio = mse1 / np.median(mse1)
    msezRatio = locres.mse_z_ratio
    return [msezRatio, mseRatio, intRatio], minI
