from __future__ import annotations

import pickle
from typing import Any

import numpy as np


class ImageData:
    """
    ROI data and metadata produced by PreprocessingPipeline.

    Raw images are NOT stored here — they're just numpy arrays passed separately.
    This class holds only the extracted ROI data and the metadata needed
    by fitters and PSF models.
    """

    def __init__(
        self,
        measured_roi_images: np.ndarray,
        roi_centers: np.ndarray,
        roi_centers_all: np.ndarray,
        source_file_indices: np.ndarray,
        pixelsize_x: float,
        pixelsize_y: float,
        pixelsize_z: float,
        bead_radius: float,
        image_size: tuple,
        skew_const: Any,
    ) -> None:
        self.measured_roi_images = measured_roi_images
        self.roi_centers = roi_centers
        self.roi_centers_all = roi_centers_all
        self.source_file_indices = source_file_indices
        self.pixelsize_x = pixelsize_x
        self.pixelsize_y = pixelsize_y
        self.pixelsize_z = pixelsize_z
        self.bead_radius = bead_radius
        self.image_size = image_size
        self.skew_const = skew_const

    def with_mask(self, mask: np.ndarray) -> ImageData:
        """Return a new ImageData with only the ROIs where mask is True."""
        return ImageData(
            measured_roi_images=self.measured_roi_images[mask],
            roi_centers=self.roi_centers[mask, :],
            roi_centers_all=self.roi_centers_all,
            source_file_indices=self.source_file_indices[mask],
            pixelsize_x=self.pixelsize_x,
            pixelsize_y=self.pixelsize_y,
            pixelsize_z=self.pixelsize_z,
            bead_radius=self.bead_radius,
            image_size=self.image_size,
            skew_const=self.skew_const,
        )

    def save(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> ImageData:
        with open(filename, "rb") as f:
            return pickle.load(f)
