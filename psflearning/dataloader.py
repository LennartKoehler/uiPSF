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
import glob
import logging
import warnings
from typing import Any, List
import logging


class DataLoader(ABC):
    """Abstract base class for data loaders.

    Subclasses must implement the ``load`` method which takes a file list
    and returns a numpy array of stacked image data.
    """

    def __init__(self, param: Any = None) -> None:
        self.param = param

    def get_file_list(self) -> List[str]:
        """Return a list of file paths matching the parameter configuration.

        Uses ``param.io.data_path``, ``param.io.keyword``, ``param.io.format``, and
        optionally ``param.io.subfolder`` to build a glob pattern and collect
        matching files.
        """
        param = self.param
        if not param.io.subfolder:
            filelist = glob.glob(param.io.data_path+'/*'+param.io.keyword+'*'+param.io.format)
        else:
            filelist = []
            folderlist = glob.glob(param.io.data_path+'/*'+param.io.subfolder+'*/')
            for f in folderlist:
                filelist.append(glob.glob(f+'/*'+param.io.keyword+'*'+param.io.format)[0])

        if not filelist:
            raise FileNotFoundError(
                f"No files matching {param.io.data_path}/*{param.io.keyword}*{param.io.format}"
            )
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


class TiffDataLoader(DataLoader):
    """Loader for TIFF/TIFF files."""

    def load(self, filelist: List[str]) -> np.ndarray:
        """Load TIFF images from *filelist* and return a stacked numpy array."""
        param = self.param
        imageraw = []
        for filename in filelist:
            logging.info("Loading: %s", filename)
            dat = np.squeeze(io.imread(filename).astype(np.float32))
            dat = (dat-param.data.camera_offset)*param.data.camera_gain
            imageraw.append(dat)
        imagesall = np.stack(imageraw)

        return imagesall


class MatDataLoader(DataLoader):
    """Loader for MATLAB .mat files."""

    def load(self, filelist: List[str]) -> np.ndarray:
        """Load MATLAB .mat data from *filelist* and return a stacked numpy array.

        Dataset keys ``'metadata'`` and ``'#refs#'`` are automatically
        excluded.          If ``param.io.variable_name`` is set it is used as the dataset
        name; otherwise all remaining keys are loaded.
        """
        param = self.param
        imageraw = []
        for filename in filelist:
            logging.info("Loading: %s", filename)
            fdata = h5.File(filename,'r')
            if param.io.variable_name:
                name = [param.io.variable_name]
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

            dat = np.squeeze(np.array(fdata.get(name[0])).astype(np.float32))
            dat = (dat-param.data.camera_offset)*param.data.camera_gain
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
            dat = (dat-param.data.camera_offset)*param.data.camera_gain
            imageraw.append(dat)
        imagesall = np.stack(imageraw)

        return imagesall


_FORMAT_MAP = {
    '.mat': MatDataLoader,
    '.tif': TiffDataLoader,
    '.tiff': TiffDataLoader,
    '.czi': CziDataLoader,
}


def get_loader(param: Any) -> DataLoader:
    """Factory function that returns the appropriate ``DataLoader`` subclass
    based on ``param.io.format``.

    Parameters
    ----------
    param : object
        Configuration object with at least a ``io.format`` attribute.

    Returns
    -------
    DataLoader
        An instance of the matching loader subclass.

    Raises
    ------
    TypeError
        If *param.io.format* is not a supported format.
    """
    fmt = param.io.format
    cls = _FORMAT_MAP.get(fmt)
    if cls is None:
        supported = ', '.join(_FORMAT_MAP.keys())
        raise TypeError(
            f'Unsupported data format: {fmt}. Supported formats: {supported}'
        )
    return cls(param)


dataloader = get_loader
