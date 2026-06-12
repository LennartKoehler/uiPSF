import os
from pathlib import Path

import pytest

from psflearning.io.param import (
    RunParameters,
    RoiParams,
    PixelSizeParams,
    FOVParams,
    ModelConfig,
    PSFModelParams,
    LossWeightParams,
    RejThresholdParams,
    LLSParams,
    IOParams,
    DataParams,
    SelectionParams,
    RuntimeParams,
    RefractiveIndices,
    load_params,
    load,
    _redefine,
)
from omegaconf import OmegaConf


_PKG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _config_path(name: str) -> str:
    return os.path.join(_PKG_DIR, "config", name)


class TestLoadParamsBaseOnly:
    def test_returns_run_parameters(self):
        param = load_params()
        assert isinstance(param, RunParameters)

    def test_io_defaults(self):
        param = load_params()
        assert isinstance(param.io, IOParams)
        assert param.io.format == ".tif"
        assert param.io.datapath == ""
        assert param.io.varname == ""

    def test_data_defaults(self):
        param = load_params()
        assert isinstance(param.data, DataParams)
        assert param.data.gain == 0.2
        assert param.data.ccd_offset == 398.6
        assert param.data.stage_mov_dir == "normal"
        assert param.data.swapxy is False

    def test_selection_defaults(self):
        param = load_params()
        assert isinstance(param.selection, SelectionParams)
        assert param.selection.roi.roi_size == [21, 21]
        assert param.selection.roi.gauss_sigma == [2, 2]
        assert param.selection.roi.peak_height == 0.2
        assert param.selection.FOV.z_step == 1

    def test_model_defaults(self):
        param = load_params()
        assert isinstance(param.model, ModelConfig)
        assert param.model.PSFtype == "zernike"

    def test_data_optics_defaults(self):
        param = load_params()
        assert param.data.emission_wavelength == 0.68
        assert param.data.NA == 1.43
        assert param.data.RI.imm == 1.516
        assert isinstance(param.data.RI, RefractiveIndices)

    def test_model_psf_defaults(self):
        param = load_params()
        assert isinstance(param.model.psf, PSFModelParams)
        assert param.model.psf.pupilsize == 64
        assert param.model.psf.n_max == 8
        assert param.model.psf.blur_sigma == 0.5
        assert param.model.psf.bin == 2

    def test_runtime_defaults(self):
        param = load_params()
        assert isinstance(param.runtime, RuntimeParams)
        assert param.runtime.usecuda is True
        assert param.runtime.relearn is True
        assert param.runtime.plotall is False
        assert param.runtime.batch_size == 1600
        assert param.runtime.iteration == 200

    def test_pixel_size_defaults(self):
        param = load_params()
        assert param.data.pixel_size.x == 0.127
        assert param.data.pixel_size.y == 0.116
        assert param.data.pixel_size.z == 0.05

    def test_loss_weight_defaults(self):
        param = load_params()
        assert param.model.loss_weight.mse1 == 1
        assert param.model.loss_weight.edge == 0.01

    def test_rej_threshold_defaults(self):
        param = load_params()
        assert param.model.rej_threshold.bias_z == 0.99
        assert param.model.rej_threshold.mse == 0.8
        assert param.model.rej_threshold.photon == 1.5


class TestLoadParamsWithPsftype:
    def test_psftype_overrides_base(self):
        param = load_params(psftype="zernike")
        assert param.model.PSFtype == "zernike_vector"

    def test_psftype_nested_override(self):
        param = load_params(psftype="zernike")
        assert param.model.loss_weight.smooth == 0
        assert param.model.loss_weight.gxy_min == 0.1
        assert param.model.rej_threshold.bias_z == 3
        assert param.model.psf.var_photon is True

    def test_psftype_preserves_unoverridden_base(self):
        param = load_params(psftype="zernike")
        assert param.data.gain == 0.2
        assert param.data.ccd_offset == 398.6
        assert param.model.loss_weight.mse1 == 1


class TestLoadParamsWithSysfile:
    def test_sysfile_overrides_base(self):
        param = load_params(sysfile="M2")
        assert param.io.format == ".tif"
        assert param.data.pixel_size.x == 0.127
        assert param.data.pixel_size.y == 0.116
        assert param.data.swapxy is True

    def test_sysfile_overrides_roi(self):
        param = load_params(sysfile="M2")
        assert param.selection.roi.roi_size == [25, 25]

    def test_sysfile_overrides_imaging(self):
        param = load_params(sysfile="M2")
        assert param.data.emission_wavelength == 0.6
        assert param.data.NA == 1.43

    def test_lls_sysfile(self):
        param = load_params(sysfile="LLS")
        assert param.data.pixel_size.x == 0.104
        assert param.selection.roi.roi_size == [37, 27, 27]
        assert param.data.LLS.skew_const == [0, -0.7845]


class TestLoadParamsLayerOrder:
    def test_psftype_overrides_sysfile(self):
        param = load_params(psftype="zernike", sysfile="M2")
        assert param.model.PSFtype == "zernike_vector"
        assert param.data.swapxy is True

    def test_sysfile_overrides_psftype_for_shared_keys(self):
        param = load_params(psftype="zernike", sysfile="M2")
        assert param.data.emission_wavelength == 0.6

    def test_full_stack(self):
        param = load_params(psftype="zernike", sysfile="M2")
        assert isinstance(param, RunParameters)
        assert param.model.PSFtype == "zernike_vector"
        assert param.selection.roi.roi_size == [25, 25]
        assert param.data.swapxy is True
        assert param.model.psf.var_photon is True


class TestLoadParamsWithUserfile:
    def test_userfile_sparse_override(self, tmp_path):
        user_yaml = tmp_path / "my_config.yaml"
        user_yaml.write_text(
            "Params:\n  data:\n    gain: 0.99\n    NA: 1.50\n"
        )
        param = load_params(userfile=str(user_yaml.with_suffix("")))
        assert param.data.gain == 0.99
        assert param.data.NA == 1.50

    def test_userfile_preserves_unoverridden(self, tmp_path):
        user_yaml = tmp_path / "my_config.yaml"
        user_yaml.write_text("Params:\n  data:\n    gain: 0.99\n")
        param = load_params(userfile=str(user_yaml.with_suffix("")))
        assert param.data.gain == 0.99
        assert param.data.ccd_offset == 398.6
        assert param.data.NA == 1.43

    def test_userfile_deeply_nested(self, tmp_path):
        user_yaml = tmp_path / "my_config.yaml"
        user_yaml.write_text(
            "Params:\n  data:\n    RI:\n      imm: 1.400\n"
        )
        param = load_params(userfile=str(user_yaml.with_suffix("")))
        assert param.data.RI.imm == 1.400
        assert param.data.RI.med == 1.335

    def test_userfile_with_psftype_and_sysfile(self, tmp_path):
        user_yaml = tmp_path / "my_config.yaml"
        user_yaml.write_text("Params:\n  data:\n    gain: 0.55\n")
        param = load_params(
            userfile=str(user_yaml.with_suffix("")),
            psftype="zernike",
            sysfile="M2",
        )
        assert param.data.gain == 0.55
        assert param.model.PSFtype == "zernike_vector"
        assert param.data.swapxy is True


class TestLoadRaw:
    def test_load_yaml(self, tmp_path):
        p = tmp_path / "test.yaml"
        OmegaConf.save(OmegaConf.create({"a": 42}), p)
        cfg = load(p)
        assert cfg.a == 42


class TestRedefine:
    def test_leaf_override(self):
        base = OmegaConf.create({"a": 1, "b": 2})
        override = OmegaConf.create({"a": 10})
        result = _redefine(base, override)
        assert result.a == 10
        assert result.b == 2

    def test_nested_override(self):
        base = OmegaConf.create({"x": {"a": 1, "b": 2}})
        override = OmegaConf.create({"x": {"a": 99}})
        result = _redefine(base, override)
        assert result.x.a == 99
        assert result.x.b == 2

    def test_new_key_added(self):
        base = OmegaConf.create({"a": 1})
        override = OmegaConf.create({"b": 2})
        result = _redefine(base, override)
        assert result.a == 1
        assert result.b == 2
