import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, label, measurements
from scipy.spatial import distance
from scipy import cluster


def extractMultiPeaks(
    im, ROIsize, sigma=None, threshold_rel=None, alternateImg=None,
    kernel=(3, 3, 3), borderDist=None, FOV=None,
):
    if sigma is not None and np.linalg.norm(sigma) > 0:
        im2 = gaussian_filter(im, sigma)
    else:
        im2 = im
    coordinates = localMax(im2, threshold_rel=threshold_rel, kernel=kernel)
    coordinates = np.array(coordinates)
    if coordinates.size > 0:
        if borderDist is not None:
            borderDist = np.array(borderDist)
            inBorder = np.all(coordinates - borderDist >= 0, axis=1) & np.all(
                im.shape - coordinates - borderDist >= 0, axis=1
            )
            coordinates = coordinates[inBorder, :]
        if FOV is not None:
            fov = np.array(FOV)
            coord_r = (coordinates[:, -1] - fov[1]) ** 2 + (coordinates[:, -2] - fov[0]) ** 2
            inFov = coord_r < (fov[2] ** 2)
            coordinates = coordinates[inFov, :]
    if alternateImg is not None:
        im = alternateImg
    centers = np.round(coordinates).astype(np.int32)
    if len(ROIsize) < centers.shape[-1]:
        centers = centers[:, -len(ROIsize) :]
    if coordinates.size > 0:
        ROIs = multiROIExtract(im, centers, ROIsize=ROIsize)
    else:
        ROIs = None
    return ROIs, centers


def extractMultiPeaks_smlm(
    im, ROIsize, sigma=None, threshold_rel=None, alternateImg=None,
    kernel=(3, 3, 3), borderDist=None, min_dist=None, FOV=None,
):
    if sigma is not None and np.linalg.norm(sigma) > 0:
        im2 = gaussian_filter(im, list(np.array(sigma) * 0.75)) - gaussian_filter(im, sigma)
    else:
        im2 = im
    coordinates = localMax(im2, threshold_rel=threshold_rel, kernel=kernel)
    coordinates = np.array(coordinates)
    if coordinates.size > 0:
        if borderDist is not None:
            borderDist = np.array(borderDist)
            inBorder = np.all(coordinates - borderDist >= 0, axis=1) & np.all(
                im.shape - coordinates - borderDist >= 0, axis=1
            )
            coordinates = coordinates[inBorder, :]
        if FOV is not None:
            fov = np.array(FOV)
            coord_r = (coordinates[:, -1] - fov[1]) ** 2 + (coordinates[:, -2] - fov[0]) ** 2
            inFov = coord_r < (fov[2] ** 2)
            coordinates = coordinates[inFov, :]
    if alternateImg is not None:
        im = alternateImg
    centers = np.round(coordinates).astype(np.int32)
    if len(ROIsize) < centers.shape[-1]:
        centers = centers[:, -len(ROIsize) :]
    if coordinates.size > 0:
        ROIs = multiROIExtract_smlm(im, centers, ROIsize=ROIsize)
    else:
        ROIs = None
    return ROIs, centers


def localMax(img, threshold_rel=None, kernel=(3, 3, 3)):
    imgMax = maximum_filter(img, size=kernel)
    imgMax = (imgMax == img) * img
    mask = imgMax == img
    if threshold_rel is not None:
        thresh = np.quantile(img[mask], 1 - 1e-4) * threshold_rel
        labels, num_labels = label(imgMax > thresh)
    else:
        labels, num_labels = label(imgMax)
    coords = measurements.center_of_mass(img, labels=labels, index=np.arange(1, num_labels + 1))
    return coords


def multiROIExtract(im, centers, ROIsize):
    listOfROIs = []
    for centerpos in centers:
        if len(centerpos) > im.ndim:
            centerpos = centerpos[-im.ndim :]
        if len(ROIsize) < len(centerpos):
            centerpos = centerpos[-len(ROIsize) :]
        myROI = _extract(im, ROIsize=ROIsize, centerpos=centerpos)
        listOfROIs.append(myROI)
    return np.stack(listOfROIs)


def multiROIExtract_smlm(im, centers, ROIsize):
    listOfROIs = []
    for centerpos in centers:
        myROI = im[_ROIcoords(centerpos, ROIsize, im.ndim)]
        listOfROIs.append(myROI)
    return np.stack(listOfROIs)


def _extract(img, ROIsize=None, centerpos=None, PadValue=0.0):
    mysize = img.shape
    if ROIsize is None:
        ROIsize = mysize
    else:
        ROIsize = _expanddimvec(ROIsize, len(mysize), mysize)
    mycenter = [sd // 2 for sd in mysize]
    if centerpos is None:
        centerpos = mycenter
    else:
        centerpos = _coordsToPos(_expanddimvec(centerpos, img.ndim, othersizes=mycenter), mysize)
    res = img[_ROIcoords(centerpos, ROIsize, img.ndim)]
    pads = [
        (max(0, ROIsize[d] // 2 - centerpos[d]), max(0, centerpos[d] + ROIsize[d] - mysize[d] - ROIsize[d] // 2))
        for d in range(img.ndim)
    ]
    resF = np.pad(res, tuple(pads), "constant", constant_values=PadValue)
    return resF


def _expanddimvec(shape, ndims, othersizes=None, trailing=False, value=1):
    import numbers

    if shape is None:
        return None
    if isinstance(shape, numbers.Number):
        shape = (shape,)
    else:
        shape = tuple(shape)
    missingdims = ndims - len(shape)
    if missingdims > 0:
        if othersizes is None:
            if trailing:
                return shape + (missingdims) * (value,)
            else:
                return (missingdims) * (value,) + shape
        else:
            if trailing:
                return shape + tuple(othersizes[-missingdims:])
            else:
                return tuple(othersizes[:missingdims]) + shape
    else:
        return shape[-ndims:]


def _coordsToPos(coords, ashape):
    return [coords[d] + (coords[d] < 0) * ashape[d] for d in range(len(coords))]


def _ROIcoords(center, asize, ndim=None):
    if ndim is None:
        ndim = len(center)
    slices = []
    if ndim > len(center):
        slices = (ndim - len(center)) * (slice(None),)
    for d in range(ndim - len(center), ndim):
        asp = asize[d]
        astart = center[d] - asp // 2
        astop = astart + asp
        slices.append(slice(max(astart, 0), max(astop, 0)))
    return tuple(slices)
