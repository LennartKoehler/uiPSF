"""
All visualization functionality for the PSF-learning pipeline.

Every public method returns a ``matplotlib.figure.Figure`` (or a list of
Figures) so that the caller decides whether to display, save, or further
customise the output.  No method calls ``plt.show()`` directly.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Union

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from typing import Tuple
from .learning.psf_variables import LocResResult, PSFResult, ROIsResult, ReportResult


def save_figs(
    figs: Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]],
    output_dir: str,
    prefix: str,
    fmt: str = "png",
    dpi: int = 150,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(figs, matplotlib.figure.Figure):
        figs = [figs]
    saved: List[str] = []
    for i, fig in enumerate(figs):
        if len(figs) == 1:
            fname = f"{prefix}.{fmt}"
        else:
            fname = f"{prefix}_{i}.{fmt}"
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", format=fmt)
        plt.close(fig)
        saved.append(os.path.abspath(path))
    return saved


class Plotter:

    # ── Report generation ────────────────────────────────────────────────

    def generate_report(
        self,
        res: PSFResult,
        rois: ROIsResult,
        locres: LocResResult,
        p,
        output_dir: str,
        index: int = 0,
        fmt: str = "png",
        dpi: int = 150,
    ) -> ReportResult:
        channeltype = p.channeltype
        pixel_size_z = p.pixel_size.z

        zernike_paths = None
        pupil_paths = None

        fig = self.plot_psf_vs_data(
            rois.measured_roi_images, rois.modeled_roi_images,
            channeltype=channeltype, pixel_size_z=pixel_size_z,
            index=index,
        )
        psf_vs_data_paths = save_figs(fig, output_dir, "psf_vs_data", fmt, dpi)

        fig = self.plot_localization(locres.localized_positions, p.pixel_size)
        if hasattr(locres, "fourier_domain_positions") and locres.fourier_domain_positions is not None:
            fig_fd = self.plot_localization(locres.fourier_domain_positions, p.pixel_size)
            localization_paths = save_figs([fig, fig_fd], output_dir, "localization", fmt, dpi)
        else:
            localization_paths = save_figs(fig, output_dir, "localization", fmt, dpi)

        try:
            figs = self.plot_zernike(
                res.zernike_coefficients, res.pupil, res.zernike_polynomial_basis,
            )
            zernike_paths = save_figs(figs, output_dir, "zernike", fmt, dpi)
        except Exception:
            try:
                n_channel = rois.measured_roi_images.shape[0] if channeltype != "single" else 1
                figs = self.plot_pupil(
                    res.pupil, channeltype=channeltype, n_channel=n_channel,
                )
                pupil_paths = save_figs(figs, output_dir, "pupil", fmt, dpi)
            except Exception:
                logging.warning("no pupil / zernike plot available")

        fig = self.plot_learned_params(
            roi_centers=rois.roi_centers, fitted_positions=res.fitted_positions, intensity=res.fitted_intensities,
            bg=res.fitted_backgrounds, drift_rate=res.drift_rate, channeltype=channeltype,
        )
        learned_params_paths = save_figs(fig, output_dir, "learned_params", fmt, dpi)

        fig = self.plot_coordinates(res.selected_roi_centers, res.all_roi_centers)
        coordinates_paths = save_figs(fig, output_dir, "coordinates", fmt, dpi)

        return ReportResult(
            psf_vs_data=psf_vs_data_paths,
            localization=localization_paths,
            zernike=zernike_paths,
            pupil=pupil_paths,
            learned_params=learned_params_paths,
            coordinates=coordinates_paths,
        )

    # ── Learned parameters ───────────────────────────────────────────────

    @staticmethod
    def plot_learned_params(
        roi_centers: np.ndarray,
        fitted_positions: np.ndarray,
        intensity: np.ndarray,
        bg: np.ndarray,
        drift_rate: np.ndarray,
        channeltype: str = "single",
    ) -> matplotlib.figure.Figure:
        photon = intensity.transpose()
        if channeltype == "4pi":
            phi = np.angle(intensity)
            photon = np.abs(intensity.transpose())

        fig = plt.figure(figsize=[16, 8])
        spec = gridspec.GridSpec(
            ncols=4, nrows=2,
            width_ratios=[3, 3, 3, 3], wspace=0.4,
            hspace=0.3, height_ratios=[4, 4],
        )

        ax = fig.add_subplot(spec[0])
        ax.plot(fitted_positions[:, 2] - roi_centers[:, 1])
        ax.set_xlabel("bead number")
        ax.set_ylabel("x (pixel)")

        ax = fig.add_subplot(spec[1])
        ax.plot(fitted_positions[:, 1] - roi_centers[:, 0])
        ax.set_xlabel("bead number")
        ax.set_ylabel("y (pixel)")

        ax = fig.add_subplot(spec[2])
        ax.plot(fitted_positions[:, 0])
        ax.set_xlabel("bead number")
        ax.set_ylabel("z (pixel)")

        if channeltype == "4pi":
            ax = fig.add_subplot(spec[3])
            ax.plot(phi)
            ax.set_xlabel("bead number")
            ax.set_ylabel("phi (radian)")

        ax = fig.add_subplot(spec[4])
        ax.plot(photon)
        if len(photon.shape) > 1:
            ax.set_xlabel("z slice")
            ax.legend(["bead 1"])
        else:
            ax.set_xlabel("bead number")
        ax.set_ylabel("photon")

        ax = fig.add_subplot(spec[5])
        ax.plot(bg)
        ax.set_xlabel("bead number")
        ax.set_ylabel("background")

        ax = fig.add_subplot(spec[6])
        ax.plot(drift_rate)
        ax.set_xlabel("bead number")
        ax.set_ylabel("drift per z slice (pixel)")
        ax.legend(["x", "y"])

        return fig

    @staticmethod
    def plot_learned_params_insitu(
        roi_centers: np.ndarray,
        fitted_positions: np.ndarray,
        intensity: np.ndarray,
        bg: np.ndarray,
        channeltype: str = "single",
    ) -> matplotlib.figure.Figure:
        photon = intensity
        if channeltype == "4pi":
            phi = np.angle(intensity)
            photon = np.abs(intensity)

        fig = plt.figure(figsize=[16, 8])
        spec = gridspec.GridSpec(
            ncols=4, nrows=2,
            width_ratios=[3, 3, 3, 3], wspace=0.4,
            hspace=0.3, height_ratios=[4, 4],
        )

        ax = fig.add_subplot(spec[0])
        ax.plot(fitted_positions[:, 2] - roi_centers[:, 1], ".")
        ax.set_xlabel("emitter number")
        ax.set_ylabel("x (pixel)")

        ax = fig.add_subplot(spec[1])
        ax.plot(fitted_positions[:, 1] - roi_centers[:, 0], ".")
        ax.set_xlabel("emitter number")
        ax.set_ylabel("y (pixel)")

        ax = fig.add_subplot(spec[2])
        ax.plot(fitted_positions[:, 0], ".")
        ax.set_xlabel("emitter number")
        ax.set_ylabel("z (pixel)")

        if channeltype == "4pi":
            ax = fig.add_subplot(spec[3])
            ax.plot(phi, ".")
            ax.set_xlabel("emitter number")
            ax.set_ylabel("phi (radian)")
            ax = fig.add_subplot(spec[6])
            ax.plot(fitted_positions[:, 0], phi, ".")
            ax.set_xlabel("z (pixel)")
            ax.set_ylabel("phi (radian)")

        ax = fig.add_subplot(spec[4])
        ax.plot(photon, ".")
        ax.set_xlabel("emitter number")
        ax.set_ylabel("photon")

        ax = fig.add_subplot(spec[5])
        ax.plot(bg, ".")
        ax.set_xlabel("emitter number")
        ax.set_ylabel("background")

        return fig

    # ── Pupil ────────────────────────────────────────────────────────────

    @staticmethod
    def plot_pupil(
        pupil: np.ndarray,
        channeltype: str = "single",
        n_channel: int = 1,
        index: Optional[int] = None,
    ) -> Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]:
        if channeltype == "single":
            return _plot_pupil_single(pupil, index)
        return plt.figure()

    # ── Zernike ──────────────────────────────────────────────────────────

    @staticmethod
    def plot_zernike(
        zernike_coeff: np.ndarray,
        pupil: np.ndarray,
        zernike_polynomial: np.ndarray,
        index: Optional[int] = None,
    ) -> List[matplotlib.figure.Figure]:
        return _plot_zernike_single(zernike_coeff, pupil, zernike_polynomial, index)

    # ── Zernike map ──────────────────────────────────────────────────────

    @staticmethod
    def plot_zernike_map(
        zmap: np.ndarray,
        zcoeff: np.ndarray,
        pupil: np.ndarray,
        Zk: np.ndarray,
        image_size: tuple,
        index: Optional[list] = None,
    ) -> matplotlib.figure.Figure:
        return _zernike_map(image_size, index, zmap, zcoeff, pupil, Zk)

    # ── PSF vs data ──────────────────────────────────────────────────────

    @staticmethod
    def plot_psf_vs_data(
        measured_images: np.ndarray,
        modeled_images: np.ndarray,
        channeltype: str = "single",
        pixel_size_z: float = 1.0,
        index: int = 0,
    ) -> Tuple[matplotlib.figure.Figure,matplotlib.figure.Figure]:
        if channeltype == "single":
            im1 = measured_images[index]
            im2 = modeled_images[index]
            fig1 = _psf_display(im1, pixel_size_z)
            fig2 = _psf_display(im2, pixel_size_z)
            return fig1, fig2

        n_channel = measured_images.shape[0]
        figs = []
        for ch in range(n_channel):
            if channeltype == "4pi":
                im1 = measured_images[ch, index, 0]
                im2 = modeled_images[ch, index, 0]
            else:
                im1 = measured_images[ch, index]
                im2 = modeled_images[ch, index]

            fig1 = _psf_display(im1, pixel_size_z)
            fig2 = _psf_display(im2, pixel_size_z)
            figs.extend([fig1, fig2])
        return figs

    @staticmethod
    def plot_psf_vs_data_insitu(
        measured_roi_images: np.ndarray,
        I_model: np.ndarray,
        pos: np.ndarray,
        zoffset: np.ndarray,
        channeltype: str = "single",
        pixel_size_z: float = 1.0,
        n_channel: int = 1,
    ) -> Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]:
        if channeltype == "single":
            zf = pos[:, 0]
            Nz = I_model.shape[0]
            edge = np.real(zoffset) + range(0, Nz + 1)
            ind = np.digitize(zf, np.array(edge).flatten())
            averaged_roi_images = np.zeros(I_model.shape)
            for ii in range(1, Nz + 1):
                mask = ind == ii
                if sum(mask) > 0:
                    averaged_roi_images[ii - 1] = np.mean(measured_roi_images[mask], axis=0)
            fig1 = _psf_display(averaged_roi_images, pixel_size_z)
            fig2 = _psf_display(I_model, pixel_size_z)
            return [fig1, fig2]

        figs = []
        for ch in range(n_channel):
            ch_measured_roi_images = measured_roi_images[ch]
            ch_I_model = I_model[ch]
            if channeltype == "4pi":
                ch_I_model = I_model[ch]
            zf = pos[:, 0]
            Nz = ch_I_model.shape[0]
            edge = np.real(zoffset) + range(0, Nz + 1)
            ind = np.digitize(zf, np.array(edge).flatten())
            averaged_roi_images = np.zeros(ch_I_model.shape)
            for ii in range(1, Nz + 1):
                mask = ind == ii
                if sum(mask) > 0:
                    averaged_roi_images[ii - 1] = np.mean(ch_measured_roi_images[mask], axis=0)

            fig1 = _psf_display(averaged_roi_images, pixel_size_z)
            fig2 = _psf_display(ch_I_model, pixel_size_z)
            figs.extend([fig1, fig2])
        return figs

    # ── Localization ─────────────────────────────────────────────────────

    @staticmethod
    def plot_localization(
        loc,
        pixel_size,
    ) -> matplotlib.figure.Figure:
        return _plot_loc_bias(loc, pixel_size)

    # ── Transform ────────────────────────────────────────────────────────

    @staticmethod
    def plot_transform(
        ref_pos: np.ndarray,
        positions: list,
        xyshift: np.ndarray,
        imgcenter: np.ndarray,
        T: np.ndarray,
        n_channel: int = 2,
    ) -> matplotlib.figure.Figure:
        fig = plt.figure(figsize=[5 * n_channel, 10])
        spec = gridspec.GridSpec(
            ncols=n_channel, nrows=2,
            width_ratios=list(np.ones(n_channel)), wspace=0.3,
            hspace=0.2, height_ratios=[1, 1],
        )

        cor_ref = np.concatenate(
            (ref_pos[:, 1:], np.ones((ref_pos.shape[0], 1))), axis=1
        )

        for i in range(1, n_channel):
            pos = positions[i]
            if n_channel < 3:
                cor_target = (
                    np.matmul(cor_ref - imgcenter, T)[..., :-1]
                    + imgcenter[:-1]
                )
            else:
                cor_target = (
                    np.matmul(cor_ref - imgcenter, T[i - 1])[..., :-1]
                    + imgcenter[:-1]
                )

            ax = fig.add_subplot(spec[i])
            ax.plot(ref_pos[:, -1], ref_pos[:, -2], ".")
            ax.plot(
                pos[:, -1] - xyshift[i][-1],
                pos[:, -2] - xyshift[i][-2],
                "o", markersize=8, mfc="none",
            )
            ax.plot(imgcenter[1], imgcenter[0], "*")
            ax.set_xlabel("x (pixel)")
            ax.set_ylabel("y (pixel)")
            ax.set_title("channel" + str(i))

            ax1 = fig.add_subplot(spec[n_channel + i])
            ax1.plot(cor_target[:, -1], cor_target[:, -2], ".")
            ax1.plot(pos[:, -1], pos[:, -2], "o", markersize=8, mfc="none")
            ax1.plot(imgcenter[1], imgcenter[0], "*")
            ax1.set_xlabel("x (pixel)")
            ax1.set_ylabel("y (pixel)")

        ax.legend(["ref", "target", "center"])
        ax1.legend(["ref_trans", "target", "center"])

        return fig

    # ── PSF display ──────────────────────────────────────────────────────

    @staticmethod
    def plot_psf(
        psf_model_image: np.ndarray,
        pixel_size_z: float = 1.0,
    ) -> matplotlib.figure.Figure:
        return _psf_display(psf_model_image, pixel_size_z)

    # ── Coordinates ──────────────────────────────────────────────────────

    @staticmethod
    def plot_coordinates(
        selected_roi_centers: np.ndarray,
        all_roi_centers: np.ndarray,
    ) -> matplotlib.figure.Figure:
        fig = plt.figure(figsize=[5, 5])
        ax = fig.add_subplot(111)
        ax.plot(all_roi_centers[:, -1], all_roi_centers[:, -2], ".")
        ax.plot(selected_roi_centers[:, -1], selected_roi_centers[:, -2], "o", markersize=8, mfc="none")
        for j in range(selected_roi_centers.shape[0]):
            ax.annotate(str(j), (selected_roi_centers[j, -1], selected_roi_centers[j, -2]), fontsize=7)
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        ax.legend(["all", "selected"])
        return fig

    # ── Strehl ratio ─────────────────────────────────────────────────────

    @staticmethod
    def plot_strehl_ratio(
        strehl_map: np.ndarray,
    ) -> matplotlib.figure.Figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(strehl_map)
        fig.colorbar(im, ax=ax)
        ax.set_title("Strehl ratio map", fontsize=15)
        return fig

    # ── FWHM ─────────────────────────────────────────────────────────────

    @staticmethod
    def plot_fwhm(
        fwhmx: np.ndarray,
        fwhmy: np.ndarray,
        fwhmz: np.ndarray,
        psf_model_image: Optional[np.ndarray] = None,
        pixel_size_x: float = 1.0,
        pixel_size_y: float = 1.0,
        pixel_size_z: float = 1.0,
    ) -> matplotlib.figure.Figure:
        if psf_model_image is not None and psf_model_image.ndim >= 3:
            return _plot_fwhm_single(psf_model_image, fwhmx, fwhmy, fwhmz, pixel_size_x, pixel_size_y, pixel_size_z)
        return _plot_fwhm_simple(fwhmx, fwhmy, fwhmz)

    # ── Z-slices (from iterative learning) ───────────────────────────────

    @staticmethod
    def plot_z_slices(
        psf_model_image: np.ndarray, step: int = 4
    ) -> matplotlib.figure.Figure:
        Nz = psf_model_image.shape[-3]
        zind = range(0, Nz, step)
        fig = plt.figure(figsize=[3 * len(zind), 3])
        for i, idx in enumerate(zind):
            ax = fig.add_subplot(1, len(zind), i + 1)
            ax.imshow(psf_model_image[idx], cmap="twilight")
            ax.axis("off")
        return fig


# ── Module-level helpers ─────────────────────────────────────────────────


def _zernike_labels() -> list:
    return [
        r"$\mathrm{D \ astigmatism}$",
        r"$\mathrm{astigmatism}$",
        r"$\mathrm{V \ coma}$",
        r"$\mathrm{H \ coma}$",
        r"$\mathrm{spherical}$",
        r"$\mathrm{2nd \ spherical}$",
    ]




def _psf_display(
    im1: np.ndarray, pz: float
) -> matplotlib.figure.Figure:
    Nz = im1.shape[0]
    zrange = np.linspace(-Nz / 2 + 0.5, Nz / 2 - 0.5, Nz) * pz
    zind = range(0, Nz, 4)
    cc = im1.shape[-1] // 2
    N = len(zind) + 1
    fig = plt.figure(figsize=[3 * N, 3])
    for i, idx in enumerate(zind):
        ax = fig.add_subplot(1, N, i + 1)
        ax.imshow(im1[idx], cmap="twilight")
        ax.set_title(
            str(np.round(zrange[idx], 2)) + r"$\ \mu$m", fontsize=30
        )
        ax.axis("off")
    ax = fig.add_subplot(1, N, N)
    ax.imshow(im1[:, cc], cmap="twilight")
    ax.axis("off")
    ax.set_title("xz", fontsize=30)
    fig.colorbar(ax.images[0], ax=ax)
    return fig


def _plot_loc_bias(
    loc,
    pixel_size,
) -> matplotlib.figure.Figure:
    Nz = loc.z.shape[1]
    fig = plt.figure(figsize=[16, 4])
    spec = gridspec.GridSpec(
        ncols=3, nrows=1,
        width_ratios=[3, 3, 3], wspace=0.3,
        hspace=0.3, height_ratios=[1],
    )

    ax = fig.add_subplot(spec[0])
    ax.plot(loc.x.transpose() * pixel_size.x * 1e3, "k", alpha=0.1)
    ax.plot(loc.x[0] * 0.0, "r")
    ax.set_xlabel("z slice")
    ax.set_ylabel("x bias (nm)")

    ax = fig.add_subplot(spec[1])
    ax.plot(loc.y.transpose() * pixel_size.y * 1e3, "k", alpha=0.1)
    ax.plot(loc.y[0] * 0.0, "r")
    ax.set_xlabel("z slice")
    ax.set_ylabel("y bias (nm)")

    ax = fig.add_subplot(spec[2])
    bias_z = (
        (loc.z - np.linspace(0, Nz - 1, Nz))
        * pixel_size.z
        * 1e3
    )
    ax.plot(bias_z.transpose(), "k", alpha=0.1)
    ax.plot(loc.z[0] * 0.0, "r")
    ax.set_xlabel("z slice")
    ax.set_ylabel("z bias (nm)")
    ax.set_ylim(
        [
            np.maximum(np.quantile(bias_z[:, 2:-2], 0.001), -300),
            np.minimum(np.quantile(bias_z[:, 2:-2], 0.999), 300),
        ]
    )

    return fig


def _plot_pupil_single(
    pupil: np.ndarray,
    index: Optional[int] = None,
) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=[12, 5])
    if index is not None:
        pupil = pupil[index]

    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np.abs(pupil), interpolation="nearest")
    ax.set_title("pupil magnitude")
    fig.colorbar(ax.images[0], ax=ax)

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.angle(pupil), interpolation="nearest")
    ax.set_title("pupil phase")
    fig.colorbar(ax.images[0], ax=ax)

    return fig


def _plot_zernike_single(
    zernike_coeff: np.ndarray,
    pupil: np.ndarray,
    zernike_polynomial_basis: np.ndarray,
    index: Optional[int] = None,
) -> List[matplotlib.figure.Figure]:
    n_max = zernike_coeff.shape[-1] - 1
    n_max_approx = int(np.sqrt(2 * n_max))
    Nk = (n_max_approx + 1) * (n_max_approx + 2) // 2

    indz = np.array([4, 5, 6, 7, 10, 21])
    textstr = _zernike_labels()

    mask = indz < Nk
    Nzk = int(np.sum(mask))

    if index is not None:
        zcoeff = zernike_coeff[:, index]
    else:
        zcoeff = zernike_coeff

    if len(pupil.shape) > 2:
        aperture = np.float32(np.abs(pupil[0]) > 0.0)
    else:
        aperture = np.float32(np.abs(pupil) > 0.0)

    fig_coeff = plt.figure(figsize=[10, 8])
    ax = fig_coeff.add_subplot(2, 1, 1)
    ax.plot(zcoeff.transpose(), ".-")
    ax.plot(indz[mask], zcoeff[1, indz[mask]], "ko", markersize=6, mfc="none")
    ax.set_xlabel("zernike polynomial")
    ax.set_ylabel("coefficient")
    ax.legend(["pupil magnitude", "pupil phase"])

    ax1 = fig_coeff.add_subplot(2, 1, 2)
    tstr = ""
    for i in range(Nzk):
        tstr = "\n".join(
            (tstr, textstr[i] + "=%.2f" % (zcoeff[1][indz[i]],))
        )
    tstr = tstr[1:]
    bbox = dict(
        boxstyle="round", fc="blanchedalmond", ec="orange", alpha=0.5
    )
    ax1.text(
        0.03, 0.9, tstr, fontsize=12, bbox=bbox,
        transform=ax1.transAxes,
        horizontalalignment="left", verticalalignment="top",
    )
    ax1.set_axis_off()

    pupil_mag = (
        np.sum(zernike_polynomial_basis * zcoeff[0].reshape((-1, 1, 1)), axis=0) * aperture
    )
    pupil_phase = (
        np.sum(zernike_polynomial_basis[4:] * zcoeff[1][4:].reshape((-1, 1, 1)), axis=0)
        * aperture
    )

    fig_pupil = plt.figure(figsize=[12, 5])
    ax = fig_pupil.add_subplot(1, 2, 1)
    ax.imshow(pupil_mag, interpolation="nearest")
    fig_pupil.colorbar(ax.images[0], ax=ax)
    ax.set_title("pupil magnitude", fontsize=20)
    ax = fig_pupil.add_subplot(1, 2, 2)
    ax.imshow(pupil_phase, interpolation="nearest")
    fig_pupil.colorbar(ax.images[0], ax=ax)
    ax.set_title("pupil phase", fontsize=20)

    return [fig_coeff, fig_pupil]


def _zernike_map(image_size, index, zmap, zcoeff, pupil, Zk):
    if index is None:
        index = [4, 5, 6, 7, 10, 11, 12, 15, 16, 21]
        mask = np.array(index) < (zcoeff.shape[-1] - 1)
        index = np.array(index)[mask]

    fig = plt.figure(figsize=[16, 4])
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(zcoeff[0].transpose(), "k", alpha=0.1)
    ax.plot(index, zcoeff[0, 0, index], "ro")
    ax.set_xlabel("zernike polynomial")
    ax.set_ylabel("coefficient")
    ax.set_title("pupil magnitude")
    ax.legend(["bead 1"])
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(zcoeff[1].transpose(), "k", alpha=0.1)
    ax.plot(index, zcoeff[1, 0, index], "ro")
    ax.set_xlabel("zernike polynomial")
    ax.set_ylabel("coefficient")
    ax.set_title("pupil phase")
    ax.legend(["bead 1"])

    if len(pupil.shape) > 2:
        aperture = np.float32(np.abs(pupil[0]) > 0.0)
    else:
        aperture = np.float32(np.abs(pupil) > 0.0)
    imsz = np.array(image_size)

    scale = (imsz[-2:] - 1) / (np.array(zmap.shape[-2:]) - 1)

    N = len(index)
    Nx = 4
    Ny = N // Nx + 1
    fig_map = plt.figure(figsize=[4.5 * Nx, 7 * Ny])
    spec = gridspec.GridSpec(
        ncols=Nx, nrows=2 * Ny,
        width_ratios=list(np.ones(Nx)), wspace=0.1,
        hspace=0.2, height_ratios=list(np.ones(2 * Ny)),
    )

    abername = [""] * np.max([zcoeff.shape[-1], 22])
    abername[3] = "defocus"
    abername[4] = "D astigmatism"
    abername[5] = "astigmatism"
    abername[6] = "V coma"
    abername[7] = "H coma"
    abername[10] = "spherical"
    abername[11] = "2nd ast."
    abername[12] = "2nd D ast."
    abername[15] = "2nd H coma"
    abername[16] = "2nd V coma"
    abername[21] = "2nd spherical"

    for i, id_val in enumerate(index):
        j = i // Nx
        ax = fig_map.add_subplot(spec[i + j * Nx])
        ax.imshow(
            zmap[1, id_val], cmap="twilight", interpolation="nearest"
        )
        ax.axis("off")
        ax.set_title(
            "(" + str(id_val) + ") " + abername[id_val], fontsize=16
        )
        fig_map.colorbar(ax.images[0], ax=ax)
        ax = fig_map.add_subplot(spec[i + (j + 1) * Nx])
        ax.imshow(
            Zk[id_val] * aperture, cmap="viridis", interpolation="nearest"
        )
        ax.axis("off")
        fig_map.colorbar(ax.images[0], ax=ax)

    return fig_map


def _plot_fwhm_single(
    psf_model_image: np.ndarray,
    fwhmx: np.ndarray,
    fwhmy: np.ndarray,
    fwhmz: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
    pixel_size_z: float,
) -> matplotlib.figure.Figure:
    from .analysis import getfwhm

    Imaxh = np.max(psf_model_image) / 2
    Ix, xh, Iy, yh, Iz, zh = getfwhm(psf_model_image)
    fwhmx_val = np.diff(xh) * pixel_size_x * 1e3
    fwhmy_val = np.diff(yh) * pixel_size_y * 1e3
    fwhmz_val = np.diff(zh) * pixel_size_z * 1e3

    fig = plt.figure(figsize=[12, 4])
    ax = fig.add_subplot(121)
    ax.plot(Ix, "o-")
    ax.plot(xh, [Imaxh, Imaxh], "-")
    ax.plot(Iy, "o-")
    ax.plot(yh, [Imaxh, Imaxh], "-")
    ax.set_title(
        "FWHMxy: "
        + str(np.round((fwhmx_val[0] + fwhmy_val[0]) / 2, 2))
        + " nm",
        fontsize=15,
    )
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("intensity")

    ax = fig.add_subplot(122)
    ax.plot(Iz, "o-")
    ax.plot(zh, [Imaxh, Imaxh], "-")
    ax.set_title(
        "FWHMz: " + str(np.round(fwhmz_val[0], 2)) + " nm",
        fontsize=15,
    )
    ax.set_xlabel("z (pixel)")
    ax.set_ylabel("intensity")

    return fig


def _plot_fwhm_simple(
    fwhmx: np.ndarray,
    fwhmy: np.ndarray,
    fwhmz: np.ndarray,
) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=[12, 4])
    ax = fig.add_subplot(121)
    ax.set_title(
        "FWHMxy: "
        + str(
            np.round(
                (np.mean(fwhmx) + np.mean(fwhmy)) / 2, 2
            )
        )
        + " nm",
        fontsize=15,
    )
    ax = fig.add_subplot(122)
    ax.set_title(
        "FWHMz: " + str(np.round(np.mean(fwhmz), 2)) + " nm",
        fontsize=15,
    )
    return fig
