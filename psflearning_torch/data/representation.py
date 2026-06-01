import numpy as np
from scipy.ndimage import gaussian_filter

from .preprocessing import extractMultiPeaks, extractMultiPeaks_smlm


class PreprocessedImageDataSingleChannel:
    def __init__(self, images, is4pi=None):
        if is4pi is None or is4pi is False:
            self.is4pi = False
            self.num_dims = 4
            self.func_2Dimage = lambda ims: np.max(ims, axis=-3)
        else:
            self.is4pi = True
            self.num_dims = 5
            self.func_2Dimage = lambda ims: np.max(ims[0], axis=0)

        self.images = np.array(images, dtype=np.float32)
        if self.images.ndim != self.num_dims:
            raise ValueError(f"Input needs to have {self.num_dims} dimensions.")
        self.rois = []
        self.centers = []
        self.centers_all = []
        self.file_idxs = []
        self.rois_available = False
        self.min_border_dist = None
        self.skew_const = None
        self.zT = None
        self.pixelsize_x = None
        self.pixelsize_y = None
        self.pixelsize_z = None
        self.bead_radius = None
        self.image_size = None

    def find_rois(self, roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel, FOV=None, min_center_dist=None, max_bead_number=None):
        self.min_border_dist = min_border_dist
        all_rois = []
        all_centers = []
        file_idxs = []

        for file_idx, image in enumerate(self.images):
            if len(roi_size) > 2:
                im2 = image
            else:
                im2 = self.func_2Dimage(image)
            rois, centers = extractMultiPeaks(
                im2, ROIsize=roi_size, sigma=gaus_sigma,
                borderDist=min_border_dist, threshold_rel=max_threshold,
                alternateImg=image, kernel=max_kernel,
            )
            if rois is not None:
                if min_center_dist is None:
                    min_center_dist = np.hypot(roi_size[-2], roi_size[-1])
                rois, centers = self.remove_close_rois(rois, centers, min_center_dist)
                if FOV is not None:
                    fov = np.array(FOV)
                    coord_r = (centers[:, -1] - fov[1]) ** 2 + (centers[:, -2] - fov[0]) ** 2
                    inFov = coord_r < (fov[2] ** 2)
                    rois = rois[inFov]
                    centers = centers[inFov]
                all_rois.append(rois)
                all_centers.append(centers)
                file_idxs += [file_idx] * rois.shape[0]
            if max_bead_number and len(file_idxs) > max_bead_number:
                break

        if all_rois:
            self.rois = np.concatenate(all_rois, axis=0)
            self.centers = np.concatenate(all_centers, axis=0)
            self.centers_all = self.centers.copy()
            self.file_idxs = np.array(file_idxs, dtype=np.int32)
            if self.rois.shape[0] > 0:
                self.rois_available = True
        return self.rois, self.centers

    def remove_close_rois(self, rois, centers, min_dist):
        from scipy.spatial import distance as dist_mod

        if centers.shape[0] <= 1:
            return rois, centers
        d = dist_mod.cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        keep = np.all(d >= min_dist, axis=1)
        return rois[keep], centers[keep]

    def cut_new_rois(self, centers, file_idxs, roi_size, min_border_dist=None):
        from .preprocessing import multiROIExtract

        if min_border_dist is None:
            min_border_dist = self.min_border_dist
        all_rois = []
        for file_idx in range(len(self.images)):
            mask = file_idxs == file_idx
            if np.any(mask):
                c = centers[mask]
                rois = multiROIExtract(self.images[file_idx], c, ROIsize=roi_size)
                all_rois.append(rois)
        if all_rois:
            self.rois = np.concatenate(all_rois, axis=0).astype(np.float32)
        self.centers = centers
        self.file_idxs = file_idxs

    def get_image_data(self):
        if not self.rois_available:
            raise RuntimeError("ROIs not available. Call find_rois() first.")
        return self.images, self.rois, self.centers, self.file_idxs

    def process(
        self, roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel,
        pixelsize_x, pixelsize_y, pixelsize_z, bead_radius, modulation_period=None,
        plot=False, padPSF=False, isVolume=False, skew_const=None, max_bead_number=None,
        FOV=None, min_center_dist=None,
    ):
        self.pixelsize_x = pixelsize_x
        self.pixelsize_y = pixelsize_y
        self.pixelsize_z = pixelsize_z
        self.bead_radius = bead_radius
        self.image_size = self.images.shape[-3:]

        self.find_rois(
            roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel,
            FOV=FOV, min_center_dist=min_center_dist, max_bead_number=max_bead_number,
        )

        if not self.rois_available:
            return

        offset = min(np.quantile(self.rois, 1e-3), 0)
        self.rois = self.rois - offset
        self.offset = offset

        if padPSF and bead_radius > 0:
            pad_size = int(bead_radius / pixelsize_z)
            self.rois = np.pad(self.rois, ((0, 0), (pad_size, pad_size), (0, 0), (0, 0)), mode="edge")

        if modulation_period is not None and modulation_period > 0:
            self.zT = modulation_period / pixelsize_z

        self.skew_const = skew_const

    def deskew_roi(self, roi_size):
        for i in range(self.rois.shape[0]):
            for z in range(self.rois.shape[-3]):
                shift = int(round(self.skew_const[0] * (z - self.rois.shape[-3] // 2)))
                self.rois[i, z] = np.roll(self.rois[i, z], shift, axis=-1)
            cy, cx = self.rois.shape[-2] // 2, self.rois.shape[-1] // 2
            ry, rx = roi_size[-2] // 2, roi_size[-1] // 2
            self.rois[i] = self.rois[i, :, cy - ry : cy + ry, cx - rx : cx + rx]


class PreprocessedImageDataMultiChannel:
    def __init__(self, images, single_channel_dtype=PreprocessedImageDataSingleChannel, is4pi=None):
        self.channels = []
        self.numofchannel = images.shape[0]
        self.shiftxy = None
        for i in range(images.shape[0]):
            self.channels.append(single_channel_dtype(images[i], is4pi=is4pi))

    def find_rois(self, **kwargs):
        for ch in self.channels:
            ch.find_rois(**kwargs)

    def cut_new_rois(self, channel, centers, file_idxs, **kwargs):
        self.channels[channel].cut_new_rois(centers, file_idxs, **kwargs)

    def get_channel(self, channel):
        return self.channels[channel]

    def get_image_data(self):
        results = [ch.get_image_data() for ch in self.channels]
        return list(map(list, zip(*results)))

    def pair_coordinates(self, delete_id=None):
        ref = self.channels[0]
        for i in range(1, len(self.channels)):
            target = self.channels[i]
            if ref.centers.shape[0] == 0 or target.centers.shape[0] == 0:
                continue
            from scipy.spatial import distance as dist_mod

            d = dist_mod.cdist(ref.centers[:, -2:], target.centers[:, -2:])
            min_idx = np.argmin(d, axis=1)
            paired = d[np.arange(len(min_idx)), min_idx] <= 5
            if delete_id is not None:
                paired[delete_id] = False
            ref.centers = ref.centers[paired]
            ref.rois = ref.rois[paired]
            ref.file_idxs = ref.file_idxs[paired]
            target.centers = target.centers[min_idx[paired]]
            target.rois = target.rois[min_idx[paired]]
            target.file_idxs = target.file_idxs[min_idx[paired]]
        self.centers = [ch.centers for ch in self.channels]

    def find_channel_shift_cor(self, plot=True):
        ref = self.channels[0]
        shift = np.array([0.0, 0.0])
        for i in range(1, len(self.channels)):
            target = self.channels[i]
            for _ in range(5):
                d = target.centers[:, -2:] - ref.centers[:, -2:] + shift
                q = np.quantile(np.abs(d), 0.985, axis=0)
                mask = np.all(np.abs(d) < q, axis=1)
                shift = np.median(d[mask], axis=0)
        self.shiftxy = shift
        return shift

    def process(
        self, roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel,
        pixelsize_x, pixelsize_y, pixelsize_z, bead_radius, modulation_period=None,
        plot=False, padPSF=False, isVolume=False, skew_const=None, max_bead_number=None,
        FOV=None, min_center_dist=None,
    ):
        for ch in self.channels:
            ch.find_rois(
                roi_size, gaus_sigma, min_border_dist, max_threshold, max_kernel,
                FOV=FOV, min_center_dist=min_center_dist, max_bead_number=max_bead_number,
            )

        if self.shiftxy is None:
            self.find_channel_shift_cor(plot=plot)
        self.pair_coordinates()

        for ch in self.channels:
            offset = min(np.quantile(ch.rois, 1e-3), 0)
            ch.rois = ch.rois - offset
            ch.pixelsize_x = pixelsize_x
            ch.pixelsize_y = pixelsize_y
            ch.pixelsize_z = pixelsize_z
            ch.bead_radius = bead_radius
            ch.image_size = ch.images.shape[-3:]
            if modulation_period is not None and modulation_period > 0:
                ch.zT = modulation_period / pixelsize_z
            ch.skew_const = skew_const
            if padPSF and bead_radius > 0:
                pad_size = int(bead_radius / pixelsize_z)
                ch.rois = np.pad(ch.rois, ((0, 0), (pad_size, pad_size), (0, 0), (0, 0)), mode="edge")
