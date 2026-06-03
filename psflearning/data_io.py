"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu

Data loading and preprocessing: loading raw image stacks from disk and
preparing :class:`PreprocessedImageData` objects ready for PSF fitting.
"""

import numpy as np

from .dataloader import get_loader


# ── Data loading ───────────────────────────────────────────────────────

def load_data(param, frange=None):
    """Load raw image stacks according to *param*.

    Parameters
    ----------
    param : OmegaConf
        Experiment parameters (file paths, channel type, PSF type, …).
    frange : tuple of (int, int), optional
        Slice ``filelist[frange[0]:frange[1]]`` to restrict the files loaded.

    Returns
    -------
    numpy.ndarray
        Image array with axes arranged for the requested *channeltype* and
        *PSFtype*.
    """
    filelist = param.filelist

    loader = get_loader(param)
    if not filelist:
        filelist = loader.getfilelist()
    if frange:
        filelist = filelist[frange[0]:frange[1]]

    images_all = loader.load(filelist)
    images = _rearrange_axes(images_all, param)
    images = _reshape_insitu(images, param)
    images = _swap_xy(images, param)
    images = _flip_if_reverse(images, param)

    print(images.shape)
    return images


# ── Data preprocessing ─────────────────────────────────────────────────

def prep_data(param, images):
    """Detect beads / localisations in *images* and return a
    :class:`PreprocessedImageData` object.

    Parameters
    ----------
    param : OmegaConf
        Experiment parameters (ROI, pixel sizes, FOV, …).
    images : numpy.ndarray
        Image array as returned by :func:`load_data`.

    Returns
    -------
    PreprocessedImageData
        Data object with extracted ROIs ready for PSF fitting.
    """
    roi_size = param.roi.roi_size
    fov = list(param.FOV.values())
    skew_const = param.LLS.skew_const
    is_volume = param.PSFtype == "voxel"

    images = _crop_fov(images, fov)

    dataobj = _create_dataobj(images, param)

    fov_param = None if fov[2] == 0 else fov
    skew_param = None if (skew_const[0] == 0.0 and skew_const[1] == 0.0) else skew_const

    dataobj.process(
        roi_size=roi_size,
        gaus_sigma=param.roi.gauss_sigma,
        min_border_dist=list(np.array(roi_size) // 2 + 1),
        min_center_dist=np.max(roi_size),
        FOV=fov_param,
        max_threshold=param.roi.peak_height,
        max_kernel=param.roi.max_kernel,
        pixelsize_x=param.pixel_size.x,
        pixelsize_y=param.pixel_size.y,
        pixelsize_z=param.pixel_size.z,
        bead_radius=param.roi.bead_radius,
        modulation_period=param.fpi.modulation_period,
        plot=param.plotall,
        padPSF=False,
        isVolume=is_volume,
        skew_const=skew_param,
        max_bead_number=param.roi.max_bead_number,
    )
    return dataobj


# ── Internal helpers ───────────────────────────────────────────────────

def _rearrange_axes(images_all, param):
    """Transpose *images_all* so the leading axis matches *channeltype*."""
    channeltype = param.channeltype
    psf_type = param.PSFtype

    if channeltype == "4pi":
        if "insitu" in psf_type:
            return np.transpose(images_all, (1, 0, 2, 3, 4))
        if param.varname:
            return np.transpose(images_all, (1, 0, 2, 3, 4, 5))
        return np.transpose(images_all, (1, 0, 3, 2, 4, 5))

    if channeltype == "multi":
        images = np.transpose(images_all, (1, 0, 2, 3, 4))
        images = _reorder_ref_channel(images, param)
        return images

    # single channel – no transpose
    return images_all


def _reorder_ref_channel(images, param):
    """Swap the first channel with *ref_channel* so the reference comes first."""
    ref = param.ref_channel
    n_channels = images.shape[0]

    defocus = [
        param.option.multi.defocus_offset + i * param.option.multi.defocus_delay
        for i in range(n_channels)
    ]
    defocus[0], defocus[ref] = defocus[ref], defocus[0]
    param.option.multi.defocus = defocus

    idx = list(range(n_channels))
    idx[0], idx[ref] = idx[ref], idx[0]
    return images[idx]


def _reshape_insitu(images, param):
    """Flatten the z-position dimension for insitu (SMLM) data."""
    if "insitu" not in param.PSFtype:
        return images

    channeltype = param.channeltype
    if channeltype == "single":
        return images.reshape(-1, images.shape[-2], images.shape[-1])
    # multi / 4pi – keep the channel axis
    return images.reshape(images.shape[0], -1, images.shape[-2], images.shape[-1])


def _swap_xy(images, param):
    """Optionally swap the x and y axes."""
    if not param.swapxy:
        return images
    tmp = np.zeros(
        images.shape[:-2] + (images.shape[-1], images.shape[-2]),
        dtype=np.float32,
    )
    tmp[0:] = np.swapaxes(images[0:], -1, -2)
    return tmp


def _flip_if_reverse(images, param):
    """Flip the z-axis when the stage moves in reverse for bead data."""
    if param.stage_mov_dir == "reverse" and param.datatype == "bead":
        return np.flip(images, axis=-3)
    return images


def _crop_fov(images, fov):
    """Slice *images* along the z-axis according to *fov*."""
    zstart = fov[-3]
    zend = images.shape[-3] + fov[-2]
    zstep = fov[-1]
    zind = range(zstart, zend, zstep)
    ims = np.swapaxes(images, 0, -3)
    ims = ims[zind]
    return np.swapaxes(ims, 0, -3)

# from .learning import (
#     PreprocessedImageDataSingleChannel,
#     PreprocessedImageDataMultiChannel,
#     PreprocessedImageDataSingleChannel_smlm,
#     PreprocessedImageDataMultiChannel_smlm,
# )
#
# def _create_dataobj(images, param):
#     """Instantiate the correct :class:`PreprocessedImageData` subclass."""
#     channeltype = param.channeltype
#     is_insitu = "insitu" in param.PSFtype
#
#     if channeltype == "single":
#         cls = PreprocessedImageDataSingleChannel_smlm if is_insitu else PreprocessedImageDataSingleChannel
#         return cls(images)
#
#     if channeltype == "4pi":
#         if is_insitu:
#             return PreprocessedImageDataMultiChannel_smlm(
#                 images, PreprocessedImageDataSingleChannel_smlm, is4pi=True
#             )
#         return PreprocessedImageDataMultiChannel(
#             images, PreprocessedImageDataSingleChannel, is4pi=True
#         )
#
#     # multi
#     if is_insitu:
#         return PreprocessedImageDataMultiChannel_smlm(
#             images, PreprocessedImageDataSingleChannel_smlm
#         )
#     return PreprocessedImageDataMultiChannel(
#         images, PreprocessedImageDataSingleChannel
#     )
