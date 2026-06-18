from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy as sp

from .. import imagetools as nip
from .ImageData import ImageData


# could implement a interface if images of different datatypes are ever used
class PreprocessingPipeline:
    """
    Stateless pipeline that transforms raw images into an ImageData object.

    Each method is a pure step. The main entry point is process().
    """

    NUM_DIMS = 4
    DIM_NAMES = "images, z, y, x"

    @staticmethod
    def validate_images(images: Any) -> np.ndarray:
        try:
            arr = np.array(images, dtype=np.float32)
        except Exception:
            raise ValueError("Was not able to convert input to numpy array.\nCheck that dimensions are the same for all channels and for all images.")
        if arr.ndim != PreprocessingPipeline.NUM_DIMS:
            raise ValueError(f"Input needs to have {PreprocessingPipeline.NUM_DIMS} dimensions: {PreprocessingPipeline.DIM_NAMES}.")
        return arr

    @staticmethod
    def compute_skew_roi_shape(roi_size: Any, skew_const: Any, nz: int) -> list:
        if not skew_const:
            return list(roi_size)
        roisize_x = np.int32(1 + roi_size[-1] + nz * np.abs(skew_const[-1]))
        roisize_y = np.int32(1 + roi_size[-2] + nz * np.abs(skew_const[-2]))
        if len(roi_size) == 3:
            return [roi_size[0], roisize_y, roisize_x]
        return [roisize_x, roisize_x]

    @staticmethod
    def remove_close_rois(rois: np.ndarray, centers: np.ndarray, min_dist: float) -> tuple[np.ndarray, np.ndarray]:
        dist_matrix = sp.spatial.distance_matrix(centers, centers)
        keep_matrix_idxs = np.where((0 == dist_matrix) | (dist_matrix > min_dist))
        unique, counts = np.unique(keep_matrix_idxs[0], return_counts=True)
        keep_idxs = unique[counts == centers.shape[0]]
        return rois[keep_idxs], centers[keep_idxs]

    @staticmethod
    def find_rois(
        images: np.ndarray,
        roi_size: Any,
        gaus_sigma: float,
        min_border_dist: Any,
        max_threshold: float,
        max_kernel: Any,
        FOV: Any = None,
        min_center_dist: float | None = None,
        max_bead_number: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cut out rois around local maxima.

        Returns (measured_roi_images, roi_centers, source_file_indices).
        """
        func_2Dimage = lambda ims: np.max(ims, axis=-3)

        all_rois = []
        all_centers = []
        file_idxs = []

        for file_idx, image in enumerate(images):
            if len(roi_size) > 2:
                im2 = image
            else:
                im2 = func_2Dimage(image)
            rois, centers = nip.extractMultiPeaks(
                im2, ROIsize=roi_size, sigma=gaus_sigma,
                borderDist=min_border_dist, threshold_rel=max_threshold,
                alternateImg=image, kernel=max_kernel,
            )

            if rois is not None:
                if min_center_dist is None:
                    min_center_dist = np.hypot(roi_size[-2], roi_size[-1])
                rois, centers = PreprocessingPipeline.remove_close_rois(rois, centers, min_center_dist)
                if FOV is not None:
                    fov = np.array(FOV)
                    coord_r = (centers[:, -1] - fov[1]) ** 2 + (centers[:, -2] - fov[0]) ** 2
                    inFov = coord_r < (fov[2] ** 2)
                    rois = rois[inFov]
                    centers = centers[inFov]

                all_rois.append(rois)
                all_centers.append(centers)
                file_idxs += [file_idx] * rois.shape[0]
            if max_bead_number:
                if len(file_idxs) > max_bead_number:
                    break

        if not all_rois:
            raise RuntimeError('no bead is found')

        L = np.min((max_bead_number, len(file_idxs)))
        measured_roi_images = np.concatenate(all_rois)[0:L].astype(np.float32)
        roi_centers = np.concatenate(all_centers)[0:L].astype(np.int32)
        source_file_indices = np.array(file_idxs)[0:L].astype(np.int32)
        return measured_roi_images, roi_centers, source_file_indices

    @staticmethod
    def cut_new_rois(
        images: np.ndarray,
        centers: np.ndarray,
        file_idxs: np.ndarray,
        roi_size: Any,
        skew_const: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cut new rois from images with specified centers.

        Returns (measured_roi_images, roi_centers, source_file_indices).
        """
        nz = roi_size[0] if len(roi_size) == 3 else images.shape[-3]
        roi_shape = PreprocessingPipeline.compute_skew_roi_shape(roi_size, skew_const, nz)

        new_rois = []
        for i, file_idx in enumerate(file_idxs):
            new_rois.append(nip.multiROIExtract(images[file_idx], [centers[i]], roi_shape))

        return (
            np.concatenate(new_rois).astype(np.float32),
            centers.astype(np.int32),
            file_idxs.astype(np.int32),
        )

    @staticmethod
    def pad_rois_z(rois: np.ndarray, bead_radius: float, pixelsize_z: float) -> np.ndarray:
        value = np.empty((), dtype=object)
        value[()] = (0, 0)
        padsize = np.full((len(rois.shape),), value, dtype=object)
        padsize[-3] = (np.int32(bead_radius // pixelsize_z), np.int32(bead_radius // pixelsize_z))
        padded = np.pad(rois, tuple(padsize), mode='edge')
        logging.debug("padded rois shape channel : %s", padded.shape)
        return padded

    @staticmethod
    def deskew_rois(rois: np.ndarray, roi_size: Any, skew_const: Any) -> np.ndarray:
        Nz = rois.shape[-3]
        roisize_x = rois.shape[-1]
        roisize_y = rois.shape[-2]
        bxsz = roi_size
        rois1 = np.zeros(rois.shape[0:-2] + (bxsz[-2], bxsz[-1]), dtype=np.float32)
        for i in range(0, Nz):
            ccx = np.int32(np.round(roisize_x // 2 - skew_const[-1] * Nz / 2 + i * skew_const[-1]))
            ccy = np.int32(np.round(roisize_y // 2 - skew_const[-2] * Nz / 2 + i * skew_const[-2]))
            tmp = rois[..., i, ccy - bxsz[-2] // 2:ccy + bxsz[-2] // 2 + bxsz[-2] % 2, ccx - bxsz[-1] // 2:ccx + bxsz[-1] // 2 + bxsz[-1] % 2]
            rois1[..., i, :, :] = tmp
        logging.debug("deskewed rois shape channel : %s", rois1.shape)
        return rois1

    @staticmethod
    def process(
        images: Any,
        roi_size: Any,
        gaus_sigma: float,
        min_border_dist: Any,
        max_threshold: float,
        max_kernel: Any,
        pixelsize_x: float,
        pixelsize_z: float,
        bead_radius: float,
        min_center_dist: float | None = None,
        FOV: Any = None,
        padPSF: bool = True,
        plot: bool = True,
        pixelsize_y: float | None = None,
        skew_const: Any = None,
        max_bead_number: int | None = None,
    ) -> ImageData:
        """Run the full preprocessing pipeline and return an ImageData."""
        validated_images = PreprocessingPipeline.validate_images(images)

        pixelsize_y = pixelsize_y if pixelsize_y is not None else pixelsize_x

        nz = roi_size[0] if len(roi_size) == 3 else validated_images.shape[-3]
        roi_shape = PreprocessingPipeline.compute_skew_roi_shape(roi_size, skew_const, nz)
        if skew_const:
            min_border_dist = list(np.array(roi_shape) // 2 + 1)

        measured_roi_images, roi_centers, source_file_indices = PreprocessingPipeline.find_rois(
            validated_images, roi_shape if skew_const else roi_size,
            gaus_sigma, min_border_dist, max_threshold, max_kernel,
            FOV, min_center_dist, max_bead_number,
        )

        roi_centers_all = roi_centers.copy()
        logging.debug("rois shape channel : %s", measured_roi_images.shape)

        offset = np.min((np.quantile(measured_roi_images, 1e-3), 0))
        measured_roi_images = measured_roi_images - offset

        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=[6, 6])
            plt.plot(roi_centers[:, -1], roi_centers[:, -2], 'o', markersize=8, markerfacecolor='none')
            plt.show()

        if padPSF:
            measured_roi_images = PreprocessingPipeline.pad_rois_z(measured_roi_images, bead_radius, pixelsize_z)

        if skew_const:
            measured_roi_images = PreprocessingPipeline.deskew_rois(measured_roi_images, roi_size, skew_const)

        return ImageData(
            measured_roi_images=measured_roi_images,
            roi_centers=roi_centers,
            roi_centers_all=roi_centers_all,
            source_file_indices=source_file_indices,
            pixelsize_x=pixelsize_x,
            pixelsize_y=pixelsize_y,
            pixelsize_z=pixelsize_z,
            bead_radius=bead_radius,
            image_size=validated_images.shape,
            skew_const=skew_const,
        )
