from __future__ import annotations

from abc import ABCMeta, abstractmethod
import pickle

import numpy as np

class FitterInterface:
    """
    Interface for PSF fitting procedures.

    Classes implementing this interface define the fitting procedure. They
    combine an optimizer and a loss function to learn PSF parameters.
    Data and PSF models are passed explicitly to method calls, not stored
    as instance state.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def learn_psf(self, data, psf, variables, start_time=None):
        """
        Run the PSF learning optimization.

        Parameters
        ----------
        data : PreprocessedImageData
            Image data with extracted ROIs.
        psf : PSFInterface
            PSF model for forward image computation.
        variables : LearnablePSFParameters
            Initial learnable variables.
        start_time : float, optional
            Start-time stamp for progress reporting.

        Returns
        -------
        tuple
            ``(psfResult, forward_images, toc)``
        """
        raise NotImplementedError("You need to implement a 'learn_psf' method in your fitter class.")

    def save(self, filename: str) -> None:
        """
        Save object to file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> FitterInterface:
        """
        Load object from file.
        """
        with open(filename, "rb") as f:
            self = pickle.load(f)
        return self
