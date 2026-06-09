"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Sheng Liu
"""
#%%
from abc import ABC, abstractmethod
import h5py as h5
import czifile as czi
import numpy as np
from skimage import io
# append the path of the parent directory as long as it's not a real package
import glob
import json
import logging
import warnings
from PIL import Image
from typing import Any, List


class DataLoader(ABC):
    """Abstract base class for data loaders.

    Subclasses must implement the ``load`` method which takes a file list
    and returns a numpy array of stacked image data.
    """

    def __init__(self, param: Any = None) -> None:
        self.param = param

    def get_file_list(self) -> List[str]:
        """Return a list of file paths matching the parameter configuration.

        Uses ``param.datapath``, ``param.keyword``, ``param.format``, and
        optionally ``param.subfolder`` to build a glob pattern and collect
        matching files.
        """
        param = self.param
        if not param.subfolder:
            filelist = glob.glob(param.datapath+'/*'+param.keyword+'*'+param.format)
        else:
            filelist = []
            folderlist = glob.glob(param.datapath+'/*'+param.subfolder+'*/')
            for f in folderlist:
                filelist.append(glob.glob(f+'/*'+param.keyword+'*'+param.format)[0])

        return sorted(filelist)


    def getFileList(self) -> List[str]:
        """Deprecated alias for :meth:`get_file_list`."""
        warnings.warn(
            "getFileList is deprecated, use get_file_list instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_file_list()

    @abstractmethod
    def load(self, filelist: List[str]) -> np.ndarray:
        """Load data from *filelist* and return a numpy array."""
        ...

    def split_channel(self, dat: np.ndarray) -> np.ndarray:
        """Split image data into channels based on dual/multi channel configuration.

        For dual-channel setups the image is split either vertically (up-down)
        or horizontally (left-right) and optionally mirrored.  For multi-channel
        setups the image is divided into tiles of size
        ``param.multi.channel_size``.
        """
        param = self.param
        if param.dual.channel_arrange:
            if param.dual.channel_arrange == 'up-down':
                cc = dat.shape[-2]//2
                if param.dual.mirrortype == 'up-down':
                    dat = np.stack([dat[:,:-cc],np.flip(dat[:,cc:],axis=-2)])
                elif param.dual.mirrortype == 'left-right':
                    dat = np.stack([dat[:,:-cc],np.flip(dat[:,cc:],axis=-1)])
                else:
                    dat = np.stack([dat[:,:-cc],dat[:,cc:]])
            else:
                cc = dat.shape[-1]//2
                if param.dual.mirrortype == 'up-down':
                    dat = np.stack([dat[...,:-cc],np.flip(dat[...,cc:],axis=-2)])
                elif param.dual.mirrortype == 'left-right':
                    dat = np.stack([dat[...,:-cc],np.flip(dat[...,cc:],axis=-1)])
                else:
                    dat = np.stack([dat[...,:-cc],dat[...,cc:]])
        if param.multi.channel_size:
            roisz = param.multi.channel_size
            xdiv = list(range(0,dat.shape[-1],roisz[-1]))
            ydiv = list(range(0,dat.shape[-2],roisz[-2]))
            im = []
            for yd in ydiv[:-1]:
                for xd in xdiv[:-1]:
                    im.append(dat[...,yd:yd+roisz[-2],xd:xd+roisz[-1]])

            dat = np.stack(im)

        return dat

    def splitChannel(self, dat: np.ndarray) -> np.ndarray:
        """Deprecated alias for :meth:`split_channel`."""
        warnings.warn(
            "splitChannel is deprecated, use split_channel instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.split_channel(dat)


class TiffDataLoader(DataLoader):
    """Loader for TIFF/TIFF files."""

    def load(self, filelist: List[str]) -> np.ndarray:
        """Load TIFF images from *filelist* and return a stacked numpy array.

        For SMLM data, frames within ``param.insitu.frame_range`` are read
        sequentially.  Channels are split if ``param.channeltype`` is
        ``'multi'``.
        """
        param = self.param
        imageraw = []
        for filename in filelist:
            logging.info("Loading: %s", filename)
            if param.datatype == 'smlm':
                dat = []
                fID = Image.open(filename)

                for ii in range(param.insitu.frame_range[0],param.insitu.frame_range[1]):
                    fID.seek(ii)
                    dat.append(np.asarray(fID))
                dat = np.stack(dat).astype(np.float32)
            else:
                dat = np.squeeze(io.imread(filename).astype(np.float32))
            if param.channeltype == 'multi':
                dat = self.split_channel(dat)

            dat = (dat-param.ccd_offset)*param.gain
            imageraw.append(dat)
        imagesall = np.stack(imageraw)

        return imagesall


class MatDataLoader(DataLoader):
    """Loader for MATLAB .mat files."""

    def load(self, filelist: List[str]) -> np.ndarray:
        """Load MATLAB .mat data from *filelist* and return a stacked numpy array.

        Dataset keys ``'metadata'`` and ``'#refs#'`` are automatically
        excluded.  If ``param.varname`` is set it is used as the dataset
        name; otherwise all remaining keys are loaded.
        """
        param = self.param
        imageraw = []
        for filename in filelist:
            logging.info("Loading: %s", filename)
            fdata = h5.File(filename,'r')
            if param.varname:
                name = [param.varname]
            else:
                name = list(fdata.keys())
            try:
                name.remove('metadata')
            except ValueError:
                pass
            try:
                name.remove('#refs#')
            except ValueError:
                pass

            if param.channeltype == 'single':
                dat = np.squeeze(np.array(fdata.get(name[0])).astype(np.float32))
            else:
                if len(name)>1:
                    dat = []
                    for ch in name:
                        datai = np.squeeze(np.array(fdata.get(ch)).astype(np.float32))
                        dat.append(datai)
                    dat = np.squeeze(np.stack(dat))
                else:
                    dat = np.squeeze(np.array(fdata.get(name[0])).astype(np.float32))
                    dat = self.split_channel(dat)

            dat = (dat-param.ccd_offset)*param.gain
            imageraw.append(dat)
        imagesall = np.stack(imageraw)

        return imagesall


class H5DataLoader(DataLoader):
    """Loader for HDF5 .h5 files (currently only for smlm data)."""

    def load(self, filelist: List[str]) -> np.ndarray:
        """Load HDF5 data from *filelist* and return a stacked numpy array.

        Navigates through single-key groups until a multi-key group is found,
        then loads the first dataset.  Falls back to a nested path if the
        direct path does not resolve to a dataset.
        """
        param = self.param
        imageraw = []

        for filename in filelist:
            f = h5.File(filename,'r')
            k = list(f.keys())
            gname = ''
            while len(k)==1:
                gname += k[0]+'/'
                k = list(f[gname].keys())
            datalist = list(f[gname].keys())
            try:
                dat = np.squeeze(np.array(f.get(gname+datalist[0])).astype(np.float32))
            except (TypeError, ValueError, AttributeError):
                dat = np.squeeze(np.array(f.get(gname+datalist[0]+'/'+datalist[0])).astype(np.float32))
            dat = dat[param.insitu.frame_range[0]:param.insitu.frame_range[1]]
            dat = (dat-param.ccd_offset)*param.gain
            imageraw.append(dat)
        imagesall = np.stack(imageraw)

        return imagesall


class CziDataLoader(DataLoader):
    """Loader for CZI files."""

    def load(self, filelist: List[str]) -> np.ndarray:
        """Load CZI image data from *filelist* and return a stacked numpy array."""
        param = self.param
        imageraw = []
        for filename in filelist:
            dat = np.squeeze(czi.imread(filename).astype(np.float32))
            dat = (dat-param.ccd_offset)*param.gain
            imageraw.append(dat)
        imagesall = np.stack(imageraw)

        return imagesall


# Format string -> loader class mapping
_FORMAT_MAP = {
    '.mat': MatDataLoader,
    '.h5': H5DataLoader,
    '.tif': TiffDataLoader,
    '.tiff': TiffDataLoader,
    '.czi': CziDataLoader,
}


def get_loader(param: Any) -> DataLoader:
    """Factory function that returns the appropriate ``DataLoader`` subclass
    based on ``param.format``.

    Parameters
    ----------
    param : object
        Configuration object with at least a ``format`` attribute.

    Returns
    -------
    DataLoader
        An instance of the matching loader subclass.

    Raises
    ------
    TypeError
        If *param.format* is not a supported format.
    """
    fmt = param.format
    cls = _FORMAT_MAP.get(fmt)
    if cls is None:
        supported = ', '.join(_FORMAT_MAP.keys())
        raise TypeError(
            f'Unsupported data format: {fmt}. Supported formats: {supported}'
        )
    return cls(param)


# Backward-compatible alias
dataloader = get_loader
