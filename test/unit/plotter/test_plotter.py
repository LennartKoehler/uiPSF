import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import pytest

from psflearning import io
from psflearning.plotter import Plotter, save_figs

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_FILE = PROJECT_ROOT / "example_data_for_uiPSF" / "1ch_40nm_bead" / "psfmodel_zernike_vector_single.h5"
OUTPUT_DIR = PROJECT_ROOT / "test_output" / "plotter"

def _load_result():
    if not RESULT_FILE.exists():
        pytest.skip(
            f"Result file not found at {RESULT_FILE}. "
            "Run test_demo_beadPSF_1ch.py first to generate it."
        )
    return io.h5.load(RESULT_FILE)

@pytest.fixture(scope="module")
def data():
    f, p = _load_result()
    return f, p

@pytest.fixture(scope="module")
def plotter():
    return Plotter()

@pytest.fixture(scope="module")
def output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR

def test_plot_learned_params(plotter, data, output_dir):
    f, p = data
    fig = plotter.plot_learned_params(
        roi_centers=f.rois.roi_centers, fitted_positions=f.res.fitted_positions, intensity=f.res.fitted_intensities,
        bg=f.res.fitted_backgrounds, drift_rate=f.res.drift_rate,
    )
    paths = save_figs(fig, str(output_dir), "learned_params")
    assert all(Path(pt).exists() for pt in paths)

def test_plot_pupil(plotter, data, output_dir):
    f, p = data
    figs = plotter.plot_pupil(
        f.res.pupil,
    )
    paths = save_figs(figs, str(output_dir), "pupil")
    assert all(Path(pt).exists() for pt in paths)

def test_plot_zernike(plotter, data, output_dir):
    f, p = data
    fig_coeff, fig_pupil = plotter.plot_zernike(
        f.res.zernike_coefficients, f.res.pupil, f.res.zernike_polynomial_basis,
    )
    paths_coeff = save_figs(fig_coeff, str(output_dir), "zernike")
    paths_pupil = save_figs(fig_pupil, str(output_dir), "pupil")
    assert all(Path(pt).exists() for pt in paths_coeff + paths_pupil)

def test_plot_psf_vs_data(plotter, data, output_dir):
    f, p = data
    fig = plotter.plot_psf_vs_data(
        f.rois.measured_roi_images, f.rois.modeled_roi_images,
        pixel_size_z=p.pixel_size.z,
        index=0,
    )
    paths = save_figs(fig, str(output_dir), "psf_vs_data")
    assert all(Path(pt).exists() for pt in paths)


def test_plot_localization(plotter, data, output_dir):
    f, p = data
    try:
        fig = plotter.plot_localization(f.locres.localized_positions, p.pixel_size)
    except AttributeError:
        pytest.skip("localization data not available for this result file")
    paths = save_figs(fig, str(output_dir), "localization")
    assert all(Path(pt).exists() for pt in paths)

def test_plot_coordinates(plotter, data, output_dir):
    f, p = data
    fig = plotter.plot_coordinates(f.res.selected_roi_centers, f.res.all_roi_centers)
    paths = save_figs(fig, str(output_dir), "coordinates")
    assert all(Path(pt).exists() for pt in paths)

def test_plot_psf(plotter, data, output_dir):
    f, p = data
    figs = plotter.plot_psf(f.res.psf_model_image, pixel_size_z=p.pixel_size.z)
    paths = save_figs(figs, str(output_dir), "psf")
    assert all(Path(pt).exists() for pt in paths)

def test_generate_report(plotter, data, output_dir):
    pytest.skip("generate_report requires ZernikePSFResult + PupilField, not available from h5")
