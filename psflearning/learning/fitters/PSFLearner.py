from __future__ import annotations

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from psflearning.learning.psfs.PSFZernikeBased import ZernikePSFResult, ZernikePSFVariables

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
        self.loss_func_single = loss_func_single
        self.optimizer = optimizer

        self.rois: np.ndarray = np.array([])
        self.forward_images: np.ndarray = np.array([])
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
        self.psf.ind = ind
        forward_images = self.psf.calc_forward_images(variables)
        loss = self.loss_func(forward_images, self.rois[ind[0]:ind[1]], variables, mu, self.loss_weight)
        return loss

    def learn_psf(
        self,
        variables: ZernikePSFVariables,
        start_time: Optional[float] = None,
    ) -> Tuple[ZernikePSFResult, float]:
        """
        Defines the procedure of the psf learning. Just asks the psf to calculate initial
        values (if not provided), runs the optimization and uses the psf object to do postprocessing.
        """

        _, rois, _, _ = self.data.get_image_data()

        try:
            self.rois = np.stack(rois)
        except ValueError:
            raise RuntimeError("At this point each channel must have same number of rois and allow np.stack.")

        pbar = tqdm(total=self.optimizer.maxiter + 50, desc='3/6: learning', bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt}, {postfix[0]}{postfix[2][loss]:>4.5f}, {postfix[1]}{postfix[2][time]:>4.2f}s", postfix=["current loss: ", "total time: ", dict(loss=0, time=start_time)])

        variables = self.optimizer.minimize(self.__objective, variables, pbar)
        toc = pbar.postfix[-1]['time']
        pbar.close()

        self.psf.ind = [0, variables.n_beads]
        self.forward_images = self.psf.calc_forward_images(variables).numpy()

        psfResult = self.psf.postprocess(variables)

        return psfResult, toc

    def remove_outliers_single(
        self,
        res: ZernikePSFResult,
        localizer: Localizer,
        threshold: list,
    ) -> Optional[ZernikePSFVariables]:
        """Remove outlier ROIs for single-channel data based on rejection metrics.

        Filters the internal data object and learning variables in-place.
        Returns the filtered variables if any outliers were removed,
        or *None* if no outliers were found (or all would be removed).
        """
        metric = localizer.reject_metric
        minI = localizer.minI
        mask = self._get_mask(metric, minI, threshold)
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


    def _get_mask(self, metric, minI, threshold):
        mask = metric[0] > -1
        for i, val in enumerate(metric):
            mask = (val < threshold[i]) & mask
        mask = (minI > 0) & mask
        return mask

