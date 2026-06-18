from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf

from psflearning.io.param import RejThresholdParams
from psflearning.learning.psfs.PSFZernikeBased import ZernikePSFResult, ZernikePSFVariables
from psflearning.learning.psfs.PSFZernikeBase import PSFContext

from ...progress import ProgressReporter
from .FitterInterface import FitterInterface
from ..data_representation.ImageData import ImageData
from ..psfs.IPSFModel import LearnablePSFParameters, IPSFModel
from ..optimizers import OptimizerABC
from typing import Any, Callable, List, Optional, Tuple


class PSFLearner(FitterInterface):
    """
    Combines optimizer and loss function to define the PSF learning process.
    Holds only learning-related state (optimizer, loss, penalty params).
    Data, PSF model, and context are passed explicitly to each method call.
    """
    def __init__(
        self,
        optimizer: OptimizerABC,
        loss_func: Callable,
        loss_func_single: Optional[Callable] = None,
        loss_weight: Optional[Any] = None,
    ) -> None:

        self.loss_func = loss_func
        self.optimizer = optimizer

        self.mu: float = 1
        self.rate: float = 1.1
        self.loss_weight = loss_weight
        return

    def _make_objective(
        self,
        psf: IPSFModel,
        measured_roi_images: np.ndarray,
        context: PSFContext,
        data=None,
    ) -> Callable:
        """Create an objective closure that captures *psf*, *context*, and *measured_roi_images*.

        The returned callable has the signature expected by the optimizers:
        ``objective(variables, mu, ind)``.
        """
        def objective(
            variables: LearnablePSFParameters,
            mu: float = 1.0,
            ind: Optional[List[int]] = None,
        ) -> Any:
            if ind is None:
                ind = [0, measured_roi_images.shape[0]]
            forward_images = psf.calc_forward_images(variables, context, data=data)
            loss = self.loss_func(forward_images, measured_roi_images[ind[0]:ind[1]], variables, mu, self.loss_weight)
            return loss
        return objective

    def learn_psf(
        self,
        data: ImageData,
        psf: IPSFModel,
        variables: ZernikePSFVariables,
        context: PSFContext,
        reporter: ProgressReporter,
    ) -> Tuple[ZernikePSFResult, np.ndarray]:
        """
        Run the PSF learning optimization.

        Parameters
        ----------
        data : ImageData
            Image data (ROIs are read from here).
        psf : IPSFModel
            PSF model used to compute forward images and postprocessing.
        variables : ZernikePSFVariables
            Initial learnable variables.
        context : PSFContext
            PSF context carrying all operational state.
        reporter : ProgressReporter
            Progress reporter.

        Returns
        -------
        tuple of (ZernikePSFResult, np.ndarray)
            ``(fit_result, forward_images)``
        """
        objective = self._make_objective(psf, data.measured_roi_images, context, data=data)

        reporter.begin_stage("3/6: learning", total=self.optimizer.maxiter + 50, show_loss=True)
        variables = self.optimizer.minimize(objective, variables, reporter)
        reporter.close()

        forward_images = psf.calc_forward_images(variables, context, data=data).numpy()

        fit_result = psf.postprocess(data, variables, context)

        return fit_result, forward_images

def filter_by_mask(
    data: ImageData,
    variables: ZernikePSFVariables,
    mask: np.ndarray,
) -> Optional[Tuple[ImageData, ZernikePSFVariables]]:
    """Remove outlier ROIs based on rejection metrics.

    Returns (filtered_data, filtered_variables) if outliers were removed,
    or *None* if no outliers were found (or all would be removed).
    """

    delete_id = np.where(~mask)
    logging.info('outlier id: %s', str(delete_id[0]))

    if not ((delete_id[0].size > 0) & (delete_id[0].size < mask.size)):
        return None

    filtered_data = data.with_mask(mask)
    logging.debug("rois shape channel : %s", filtered_data.measured_roi_images.shape)

    return filtered_data, variables.filter_by_mask(mask)



# difference between a preprocessed bead and that bead modeled using the psf (forward image)
def get_MSE_difference_ratio(
    modeled_forward_images: np.ndarray,
    measured_bead_images: np.ndarray
) -> np.ndarray:
    mydiff = modeled_forward_images[:, 1:-1] - measured_bead_images[:, 1:-1]
    mse = np.mean(np.square(mydiff), axis=(-3, -2, -1)) / np.mean(measured_bead_images, axis=(-3, -2, -1))

    if measured_bead_images.shape[0] == 1:
        mseRatio = np.array([1.0])
    else:
        mseRatio = mse / np.median(mse)
    return mseRatio


def get_minimum_intensity(
    intensities: np.ndarray
)-> np.ndarray:
    absolute_intensities = np.abs(np.squeeze(intensities, axis=(-1, -2)))
    if intensities.dtype == 'complex64':
        intensityR = absolute_intensities
    else:
        intensityR = np.real(np.squeeze(intensities, axis=(-1, -2)))

    if len(absolute_intensities.shape) < 2:
        minI = intensityR
    else:
        minI = np.min(intensityR, axis=1)

    return minI

def get_intensity_difference_ratio(
    intensities: np.ndarray
)-> np.ndarray:

    absolute_intensities = np.abs(np.squeeze(intensities, axis=(-1, -2)))
    if len(absolute_intensities.shape) < 2:
        average_intensity_per_bead = absolute_intensities
    else:
        average_intensity_per_bead = np.median(absolute_intensities, axis=1)

    if absolute_intensities.shape[0] == 1:
        intRatio = np.array([1.0])
    else:
        median_intensity_of_all_beads = np.median(average_intensity_per_bead)
        intRatio = np.square(average_intensity_per_bead - median_intensity_of_all_beads) / (median_intensity_of_all_beads * average_intensity_per_bead)
    return intRatio
