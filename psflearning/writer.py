"""
Handles all output operations: serialising PSF fitting results to HDF5.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Optional, Union

import h5py as h5
import json
import numpy as np
import tifffile
from abc import ABC, abstractmethod

from .learning.loclib import LocalizationResult
from .learning.psfs.PSFZernikeBased import ZernikePSFResult
from .learning.psfs.IPSFModel import IPSFModel, PupilField
from .learning.data_representation.PreprocessedImageDataInterface import PreprocessedImageDataInterface
from .io.param import RunParameters
from .progress import ProgressReporter, SilentReporter


@dataclass
class ROIsResult:
    """ROI data for storage."""

    roi_centers: np.ndarray
    source_file_indices: np.ndarray
    measured_roi_images: np.ndarray
    modeled_roi_images: np.ndarray
    full_image_size: tuple

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def write_tiff(psf: np.ndarray, filepath: str) -> None:
    tifffile.imwrite(filepath, psf.astype(np.float32))

class Writer(ABC):

    @abstractmethod
    def save_result(
        self,
        param: RunParameters,
        pupil_field: PupilField,
        dataobj: PreprocessedImageDataInterface,
        learning_result: ZernikePSFResult,
        forward_images: np.ndarray,
        loc_result: Optional[LocalizationResult] = None,
        reporter: Optional[ProgressReporter] = None,
    ) -> str:
        pass


class DefaultWriter(Writer):

    def save_result(
        self,
        param: RunParameters,
        pupil_field: PupilField,
        dataobj: PreprocessedImageDataInterface,
        learning_result: ZernikePSFResult,
        forward_images: np.ndarray,
        loc_result: Optional[LocalizationResult] = None,
        reporter: Optional[ProgressReporter] = None,
    ) -> str:

        if reporter is None:
            reporter = SilentReporter()

        reporter.begin_stage("6/6: saving results")
        print("Zernike magnitude:", learning_result.zernike_magnitude, "Zernike phase:", learning_result.zernike_phase, "\n")
        os.makedirs(param.io.output_path, exist_ok=True)
        savename = os.path.join(param.io.output_path, param.model.psf_type)
        write_tiff(learning_result.psf_model_image, savename + ".tif")
        reporter.close()

        return "succcess"



class H5Writer(Writer):
    """Unified interface for all write operations in the PSF-learning
    pipeline."""

    # ── Full save pipeline ───────────────────────────────────────────────

    def save_result(
        self,
        param: RunParameters,
        pupil_field: PupilField,
        dataobj: PreprocessedImageDataInterface,
        learning_result: ZernikePSFResult,
        forward_images: np.ndarray,
        loc_result: Optional[LocalizationResult] = None,
        reporter: Optional[ProgressReporter] = None,
    ) -> str:
        """Save fitting results, localisation results, and ROI data to a file.

        Parameters
        ----------
        param : RunParameters
            Experiment parameters.
        pupil_field : PupilField
            Precomputed optical quantities (provides zernike_polynomial_basis, apodization).
        dataobj : PreprocessedImageData
            Data object with extracted ROIs (provides roi_centers, roi_centers_all).
        learning_result : ZernikePSFResult
            Fitting output as returned by :func:`fitting.learn_psf`.
        loc_result : LocalizationResult
            Localisation output.
        forward_images : np.ndarray, optional
            Forward images from the learning step. Used for ``modeled_roi_images`` in
            the saved ROIs result.
        reporter : ProgressReporter, optional
            Progress reporter for display.

        Returns
        -------
        str
            Path to the written HDF5 file.
        """
        if reporter is None:
            reporter = SilentReporter()

        reporter.begin_stage("6/6: saving results")

        savename = param.io.output_path + "_" + param.model.psf_type
        result_dict = _build_result_dict(learning_result, pupil_field, dataobj)

        img, _, centers, file_idxs = dataobj.get_image_data()
        img = np.stack(img)
        rois = ROIsResult(
            roi_centers=np.stack(centers),
            source_file_indices=np.stack(file_idxs),
            measured_roi_images=dataobj.measured_roi_images,
            modeled_roi_images=forward_images,
            full_image_size=img.shape,
        )

        resfile = savename + ".h5"
        self.write_to_file(param, resfile, result_dict, loc_result, rois)


        reporter.update(1)
        reporter.close()
        return resfile

    # ── HDF5 I/O ────────────────────────────────────────────────────────

    def write_to_file(
        self,
        param: RunParameters,
        filename: str,
        res: dict,
        loc_result: Optional[LocalizationResult],
        rois: ROIsResult,
    ) -> None:
        """Write result dicts to an HDF5 file.

        Parameters
        ----------
        param : RunParameters
            Experiment parameters (serialised as a JSON attribute).
        filename : str
            Output path.
        res : dict
            PSF fitting result dict.
        locres : LocalizationResult
            Localization result.
        rois : ROIsResult
            ROI data.
        """
        param_dict = param.to_dict()
        with h5.File(filename, "w") as f:
            f.attrs["params"] = json.dumps(param_dict)
            if loc_result:
                self._write_group(f.create_group("locres"), loc_result.to_dict())
            self._write_group(f.create_group("res"), res)
            self._write_group(f.create_group("rois"), rois.to_dict())

    # ── Internal helpers ─────────────────────────────────────────────────

    def _write_group(self, group: h5.Group, data: dict) -> None:
        """Recursively write a dict into an HDF5 group."""
        for k, v in data.items():
            if isinstance(v, dict):
                sub = group.create_group(k)
                for ki, vi in v.items():
                    sub[ki] = vi
            else:
                group[k] = v


# ── Module-level helpers ─────────────────────────────────────────────────


def _build_result_dict(
    learning_result: ZernikePSFResult,
    pupil_field: PupilField,
    dataobj: PreprocessedImageDataInterface,
) -> dict:
    """Build the result dict for HDF5 storage from a ZernikePSFResult.

    Combines the fitting result with additional data from the pupil field
    and data object.
    """
    return {
        "fitted_positions": learning_result.positions,
        "fitted_backgrounds": np.squeeze(learning_result.backgrounds),
        "fitted_intensities": np.squeeze(learning_result.intensities),
        "psf_model_image_with_bead": learning_result.psf_model_image_with_bead,
        "psf_model_image": learning_result.psf_model_image,
        "pupil": learning_result.pupil,
        "zernike_coefficients": np.array([
            np.squeeze(learning_result.zernike_magnitude),
            np.squeeze(learning_result.zernike_phase),
        ]),
        "gaussian_blur_sigma": np.squeeze(learning_result.sigma) / np.pi,
        "drift_rate": learning_result.drift_xy,
        "model_image_offset": np.min(learning_result.psf_model_image),
        "zernike_polynomial_basis": pupil_field.zernike_polynomial_basis,
        "apodization": pupil_field.apodization,
        "all_roi_centers": dataobj.roi_centers_all,
        "selected_roi_centers": dataobj.roi_centers,
    }
