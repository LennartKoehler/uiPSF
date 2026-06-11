from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf, DictConfig

import hdfdict
from dotted_dict import DottedDict
import h5py


_SHARED_RENAMES = {
    "pos": "fitted_positions",
    "bg": "fitted_backgrounds",
    "intensity": "fitted_intensities",
    "I_model_bead": "psf_model_image_with_bead",
    "I_model": "psf_model_image",
    "zernike_coeff": "zernike_coefficients",
    "sigma": "gaussian_blur_sigma",
    "offset": "model_image_offset",
    "zernike_polynomial": "zernike_polynomial_basis",
    "P": "mle_parameters",
    "CRLB": "cramer_rao_bounds",
    "LL": "log_likelihoods",
    "loc": "localized_positions",
    "psf_data": "measured_roi_images",
    "psf_fit": "modeled_roi_images",
    "fileID": "source_file_indices",
}

_GROUP_SPECIFIC_RENAMES = {
    "rois": {"cor": "roi_centers", "image_size": "full_image_size"},
    "res": {"cor": "selected_roi_centers", "cor_all": "all_roi_centers"},
}


def _remap_recursive(d: dict, group: str | None = None) -> dict:
    out = {}
    renames = dict(_SHARED_RENAMES)
    if group in _GROUP_SPECIFIC_RENAMES:
        renames.update(_GROUP_SPECIFIC_RENAMES[group])
    for k, v in d.items():
        new_key = renames.get(k, k)
        child_group = k if group is None else None
        if isinstance(v, dict):
            out[new_key] = _remap_recursive(v, child_group)
        else:
            out[new_key] = v
    return out


def load(path: str | Path) -> tuple[DottedDict, DictConfig]:
    """Load data and parameters from an HDF5 file.

    Old key names are automatically remapped to their current equivalents
    so that callers always see the new field names regardless of when the
    file was written.

    Args:
        path: Path to the HDF5 file.

    Returns:
        A tuple containing the loaded data as a DottedDict and the
        parameters as a DictConfig.
    """
    with h5py.File(path, "r") as f:
        raw = hdfdict.load(f, lazy=False)
        remapped = _remap_recursive(raw)
        res = DottedDict(remapped)
        params = OmegaConf.create(f.attrs["params"])  # type: ignore[reportCallIssue]
    return res, params
