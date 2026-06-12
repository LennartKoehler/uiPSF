import os
from pathlib import Path

import pytest

from psflearning.io.param import RunParameters, load_params, save_params, load
from omegaconf import OmegaConf


class TestSaveParamsBasic:
    def test_creates_file(self, tmp_path):
        param = load_params()
        out = tmp_path / "out.yaml"
        save_params(param, out)
        assert out.exists()

    def test_file_has_params_key(self, tmp_path):
        param = load_params()
        out = tmp_path / "out.yaml"
        save_params(param, out)
        cfg = OmegaConf.load(out)
        assert "Params" in cfg

    def test_str_path(self, tmp_path):
        param = load_params()
        out = str(tmp_path / "out.yaml")
        save_params(param, out)
        assert os.path.exists(out)


class TestSaveParamsRoundTrip:
    def test_roundtrip_base(self, tmp_path):
        original = load_params()
        out = tmp_path / "saved.yaml"
        save_params(original, out)

        reloaded_cfg = load(out)
        assert reloaded_cfg.Params.data.camera_gain == 0.2
        assert reloaded_cfg.Params.model.psf_type == "zernike"

    def test_roundtrip_modified_params(self, tmp_path):
        param = load_params()
        param.data.camera_gain = 0.5
        param.data.numerical_aperture = 1.49
        param.selection.roi.roi_size = [31, 31]
        param.runtime.use_cuda = False

        out = tmp_path / "saved.yaml"
        save_params(param, out)

        cfg = load(out)
        assert cfg.Params.data.camera_gain == 0.5
        assert cfg.Params.data.numerical_aperture == 1.49
        assert cfg.Params.selection.roi.roi_size == [31, 31]
        assert cfg.Params.runtime.use_cuda is False

    def test_roundtrip_preserves_nested_structure(self, tmp_path):
        param = load_params()
        param.data.refractive_indices.immersion = 1.400
        param.model.loss_weight.smooth = 5
        param.model.rej_threshold.photon = 2.0

        out = tmp_path / "saved.yaml"
        save_params(param, out)

        cfg = load(out)
        assert cfg.Params.data.refractive_indices.immersion == 1.400
        assert cfg.Params.model.loss_weight.smooth == 5
        assert cfg.Params.model.rej_threshold.photon == 2.0

    def test_roundtrip_list_fields(self, tmp_path):
        param = load_params()
        param.selection.roi.roi_size = [31, 31, 31]
        param.selection.roi.gauss_sigma = [4, 4, 4]
        param.data.lattice_light_sheet.skew_translation_per_slice = [1, -1]
        param.io.filelist = ["a.tif", "b.tif"]

        out = tmp_path / "saved.yaml"
        save_params(param, out)

        cfg = load(out)
        assert cfg.Params.selection.roi.roi_size == [31, 31, 31]
        assert cfg.Params.selection.roi.gauss_sigma == [4, 4, 4]
        assert cfg.Params.data.lattice_light_sheet.skew_translation_per_slice == [1, -1]
        assert list(cfg.Params.io.filelist) == ["a.tif", "b.tif"]


class TestSaveParamsAsUserConfig:
    def test_saved_file_usable_as_userfile(self, tmp_path):
        param = load_params()
        param.data.camera_gain = 0.77
        param.data.numerical_aperture = 1.60

        out = tmp_path / "user_override.yaml"
        save_params(param, out)

        reloaded = load_params(userfile=str(out.with_suffix("")))
        assert reloaded.data.camera_gain == 0.77
        assert reloaded.data.numerical_aperture == 1.60

    def test_saved_merged_config_reloadable(self, tmp_path):
        param = load_params(psftype="zernike", sysfile="M2")
        param.data.camera_gain = 0.33

        out = tmp_path / "merged.yaml"
        save_params(param, out)

        reloaded_cfg = load(out)
        assert reloaded_cfg.Params.data.camera_gain == 0.33
        assert reloaded_cfg.Params.model.psf_type == "zernike_vector"
        assert reloaded_cfg.Params.data.swap_xy_dimensions is True


class TestToDict:
    def test_to_dict_returns_plain_dict(self):
        param = load_params()
        d = param.to_dict()
        assert isinstance(d, dict)
        assert "io" in d
        assert "data" in d
        assert "selection" in d
        assert "model" in d
        assert "runtime" in d

    def test_to_dict_nested(self):
        param = load_params()
        d = param.to_dict()
        assert isinstance(d["model"], dict)
        assert isinstance(d["data"], dict)
        assert d["data"]["numerical_aperture"] == 1.43

    def test_to_dict_reflects_modifications(self):
        param = load_params()
        param.data.camera_gain = 0.99
        d = param.to_dict()
        assert d["data"]["camera_gain"] == 0.99
