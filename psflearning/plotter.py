"""
All visualization functionality for the PSF-learning pipeline.

Every public method returns a ``matplotlib.figure.Figure`` (or a list of
Figures) so that the caller decides whether to display, save, or further
customise the output.  No method calls ``plt.show()`` directly.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from omegaconf import DictConfig


def save_figs(
    figs: Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]],
    output_dir: str,
    prefix: str,
    fmt: str = "png",
    dpi: int = 150,
) -> List[str]:
    """Save one or more Figures to *output_dir* with descriptive filenames.

    Parameters
    ----------
    figs : Figure or list of Figure
        The figure(s) to save.
    output_dir : str
        Directory to write into (created if it does not exist).
    prefix : str
        Filename prefix, e.g. ``"psf_vs_data"``.
    fmt : str
        File format extension (``"png"``, ``"pdf"``, ``"svg"``, …).
    dpi : int
        Resolution in dots per inch.

    Returns
    -------
    list of str
        Absolute paths of the saved files.
    """
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
    """Unified interface for all PSF-learning visualisations."""

    # ── Learned parameters ───────────────────────────────────────────────

    def plot_learned_params(
        self, fitter, param: DictConfig
    ) -> matplotlib.figure.Figure:
        """Plot positions, photons, background, and drift for bead data.

        Parameters
        ----------
        fitter : Fitter
            Fitter object with ``res`` and ``rois`` attributes.
        param : DictConfig
            Experiment parameters (``channeltype`` is used).

        Returns
        -------
        Figure
        """
        if param.channeltype == "single":
            cor = fitter.rois.cor
            pos = fitter.res.pos
            photon = fitter.res.intensity.transpose()
            bg = fitter.res.bg
            drift = fitter.res.drift_rate
        else:
            cor = fitter.rois.cor[0]
            pos = fitter.res.channel0.pos
            photon = fitter.res.channel0.intensity.transpose()
            bg = fitter.res.channel0.bg
            drift = fitter.res.channel0.drift_rate

        if param.channeltype == "4pi":
            phi = np.angle(fitter.res.channel0.intensity)
            photon = np.abs(fitter.res.channel0.intensity.transpose())

        fig = plt.figure(figsize=[16, 8])
        spec = gridspec.GridSpec(
            ncols=4, nrows=2,
            width_ratios=[3, 3, 3, 3], wspace=0.4,
            hspace=0.3, height_ratios=[4, 4],
        )

        ax = fig.add_subplot(spec[0])
        ax.plot(pos[:, 2] - cor[:, 1])
        ax.set_xlabel("bead number")
        ax.set_ylabel("x (pixel)")

        ax = fig.add_subplot(spec[1])
        ax.plot(pos[:, 1] - cor[:, 0])
        ax.set_xlabel("bead number")
        ax.set_ylabel("y (pixel)")

        ax = fig.add_subplot(spec[2])
        ax.plot(pos[:, 0])
        ax.set_xlabel("bead number")
        ax.set_ylabel("z (pixel)")

        if param.channeltype == "4pi":
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
        ax.plot(drift)
        ax.set_xlabel("bead number")
        ax.set_ylabel("drift per z slice (pixel)")
        ax.legend(["x", "y"])

        return fig

    def plot_learned_params_insitu(
        self, fitter, param: DictConfig
    ) -> matplotlib.figure.Figure:
        """Plot positions, photons, and background for insitu (SMLM) data.

        Parameters
        ----------
        fitter : Fitter
            Fitter object with ``res`` and ``rois`` attributes.
        param : DictConfig
            Experiment parameters (``channeltype`` is used).

        Returns
        -------
        Figure
        """
        if param.channeltype == "single":
            cor = fitter.rois.cor
            pos = fitter.res.pos
            photon = fitter.res.intensity
            bg = fitter.res.bg
        else:
            cor = fitter.rois.cor[0]
            pos = fitter.res.channel0.pos
            photon = fitter.res.channel0.intensity
            bg = fitter.res.channel0.bg

        if param.channeltype == "4pi":
            phi = np.angle(fitter.res.channel0.intensity)
            photon = np.abs(fitter.res.channel0.intensity)

        fig = plt.figure(figsize=[16, 8])
        spec = gridspec.GridSpec(
            ncols=4, nrows=2,
            width_ratios=[3, 3, 3, 3], wspace=0.4,
            hspace=0.3, height_ratios=[4, 4],
        )

        ax = fig.add_subplot(spec[0])
        ax.plot(pos[:, 2] - cor[:, 1], ".")
        ax.set_xlabel("emitter number")
        ax.set_ylabel("x (pixel)")

        ax = fig.add_subplot(spec[1])
        ax.plot(pos[:, 1] - cor[:, 0], ".")
        ax.set_xlabel("emitter number")
        ax.set_ylabel("y (pixel)")

        ax = fig.add_subplot(spec[2])
        ax.plot(pos[:, 0], ".")
        ax.set_xlabel("emitter number")
        ax.set_ylabel("z (pixel)")

        if param.channeltype == "4pi":
            ax = fig.add_subplot(spec[3])
            ax.plot(phi, ".")
            ax.set_xlabel("emitter number")
            ax.set_ylabel("phi (radian)")
            ax = fig.add_subplot(spec[6])
            ax.plot(pos[:, 0], phi, ".")
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

    def plot_pupil(
        self, fitter, param: DictConfig, index: Optional[int] = None
    ) -> Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]:
        """Plot pupil magnitude and phase.

        Parameters
        ----------
        fitter : Fitter
            Fitter object with ``res`` and ``rois`` attributes.
        param : DictConfig
            Experiment parameters (``channeltype`` is used).
        index : int, optional
            Index into the pupil array (for multi-slice pupils).

        Returns
        -------
        Figure or list of Figure
        """
        if param.channeltype == "single":
            return self._plot_pupil_single(fitter, index)

        if param.channeltype == "multi":
            return self._plot_pupil_multi(fitter, index)

        if param.channeltype == "4pi":
            return self._plot_pupil_4pi(fitter)

        return plt.figure()

    @staticmethod
    def _plot_pupil_single(
        fitter, index: Optional[int] = None
    ) -> matplotlib.figure.Figure:
        fig = plt.figure(figsize=[12, 5])
        if index is None:
            pupil = fitter.res.pupil
        else:
            pupil = fitter.res.pupil[index]

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(np.abs(pupil), interpolation="nearest")
        ax.set_title("pupil magnitude")
        fig.colorbar(ax.images[0], ax=ax)

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(np.angle(pupil), interpolation="nearest")
        ax.set_title("pupil phase")
        fig.colorbar(ax.images[0], ax=ax)

        return fig

    @staticmethod
    def _plot_pupil_multi(
        fitter, index: Optional[int] = None
    ) -> List[matplotlib.figure.Figure]:
        n_channel = fitter.rois.psf_data.shape[0]
        fig_mag = plt.figure(figsize=[5 * n_channel, 4])
        fig_phase = plt.figure(figsize=[5 * n_channel, 4])

        for i in range(n_channel):
            if index is None:
                pupil = fitter.res["channel" + str(i)].pupil
            else:
                pupil = fitter.res["channel" + str(i)].pupil[index]

            ax = fig_mag.add_subplot(1, n_channel, i + 1)
            pupil_mag = np.abs(pupil)
            h = ax.imshow(pupil_mag, interpolation="nearest")
            ax.axis("off")
            ax.set_title("pupil magnitude " + str(i))
            fig_mag.colorbar(h, ax=ax)

            ax1 = fig_phase.add_subplot(1, n_channel, i + 1)
            pupil_phase = np.angle(pupil)
            h1 = ax1.imshow(pupil_phase, interpolation="nearest")
            ax1.axis("off")
            ax1.set_title("pupil phase " + str(i))
            fig_phase.colorbar(h1, ax=ax1)

        return [fig_mag, fig_phase]

    @staticmethod
    def _plot_pupil_4pi(
        fitter,
    ) -> List[matplotlib.figure.Figure]:
        n_channel = fitter.rois.psf_data.shape[0]
        fig_mag = plt.figure(figsize=[20, 8])
        for i in range(n_channel):
            ax = fig_mag.add_subplot(2, 4, i + 1)
            pupil_mag = np.abs(fitter.res["channel" + str(i)].pupil1)
            ax.imshow(pupil_mag, interpolation="nearest")
            ax.axis("off")
            ax.set_title("top pupil magnitude " + str(i))
            fig_mag.colorbar(ax.images[0], ax=ax)
            ax = fig_mag.add_subplot(2, 4, i + 5)
            pupil_mag = np.abs(fitter.res["channel" + str(i)].pupil2)
            ax.imshow(pupil_mag, interpolation="nearest")
            ax.axis("off")
            ax.set_title("bottom pupil magnitude " + str(i))
            fig_mag.colorbar(ax.images[0], ax=ax)

        fig_phase = plt.figure(figsize=[20, 8])
        for i in range(n_channel):
            ax = fig_phase.add_subplot(2, 4, i + 1)
            pupil_phase = np.angle(
                fitter.res["channel" + str(i)].pupil1
            )
            ax.imshow(pupil_phase, interpolation="nearest")
            ax.axis("off")
            ax.set_title("top pupil phase " + str(i))
            fig_phase.colorbar(ax.images[0], ax=ax)
            ax = fig_phase.add_subplot(2, 4, i + 5)
            pupil_phase = np.angle(
                fitter.res["channel" + str(i)].pupil2
            )
            ax.imshow(pupil_phase, interpolation="nearest")
            ax.axis("off")
            ax.set_title("bottom pupil phase " + str(i))
            fig_phase.colorbar(ax.images[0], ax=ax)

        return [fig_mag, fig_phase]

    # ── Zernike ──────────────────────────────────────────────────────────

    def plot_zernike(
        self, fitter, param: DictConfig, index: Optional[int] = None
    ) -> Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]:
        """Plot Zernike coefficients and reconstructed pupil.

        Parameters
        ----------
        fitter : Fitter
            Fitter object with ``res`` and ``rois`` attributes.
        param : DictConfig
            Experiment parameters (``channeltype`` is used).
        index : int, optional
            Index into the Zernike coefficient array.

        Returns
        -------
        Figure or list of Figure
        """
        if param.channeltype == "single":
            return self._plot_zernike_single(fitter, index)

        if param.channeltype == "multi":
            return self._plot_zernike_multi(fitter, index)

        if param.channeltype == "4pi":
            return self._plot_zernike_4pi(fitter)

        return plt.figure()

    @staticmethod
    def _plot_zernike_single(
        fitter, index: Optional[int] = None
    ) -> List[matplotlib.figure.Figure]:
        n_max = fitter.res.zernike_coeff.shape[-1] - 1
        n_max_approx = int(np.sqrt(2 * n_max))
        Nk = (n_max_approx + 1) * (n_max_approx + 2) // 2

        indz = np.array([4, 5, 6, 7, 10, 21])
        textstr = _zernike_labels()

        mask = indz < Nk
        Nzk = int(np.sum(mask))

        if index is None:
            zcoeff = fitter.res.zernike_coeff
        else:
            zcoeff = fitter.res.zernike_coeff[:, index]

        if len(fitter.res.pupil.shape) > 2:
            aperture = np.float32(np.abs(fitter.res.pupil[0]) > 0.0)
        else:
            aperture = np.float32(np.abs(fitter.res.pupil) > 0.0)

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

        Zk = fitter.res.zernike_polynomial
        pupil_mag = (
            np.sum(Zk * zcoeff[0].reshape((-1, 1, 1)), axis=0) * aperture
        )
        pupil_phase = (
            np.sum(Zk[4:] * zcoeff[1][4:].reshape((-1, 1, 1)), axis=0)
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

    @staticmethod
    def _plot_zernike_multi(
        fitter, index: Optional[int] = None
    ) -> List[matplotlib.figure.Figure]:
        n_channel = fitter.rois.psf_data.shape[0]
        indz = np.array([4, 5, 6, 7, 10, 21])
        textstr = _zernike_labels()

        fig_coeff = plt.figure(figsize=[12, 6])
        ax1 = fig_coeff.add_subplot(2, 2, 1)
        ax2 = fig_coeff.add_subplot(2, 2, 2)
        ax5 = fig_coeff.add_subplot(2, 2, 3)

        fig_pupil_mag = plt.figure(figsize=[5 * n_channel, 4])
        fig_pupil_phase = plt.figure(figsize=[5 * n_channel, 4])
        Zk = fitter.res.channel0.zernike_polynomial

        n_max = fitter.res.channel0.zernike_coeff.shape[-1] - 1
        n_max_approx = int(np.sqrt(2 * n_max))
        Nk = (n_max_approx + 1) * (n_max_approx + 2) // 2
        mask = indz < Nk
        Nzk = int(np.sum(mask))

        for i in range(n_channel):
            if index is None:
                zcoeff = fitter.res["channel" + str(i)].zernike_coeff
            else:
                zcoeff = fitter.res["channel" + str(i)].zernike_coeff[
                    :, index
                ]

            if len(fitter.res["channel" + str(i)].pupil.shape) > 2:
                aperture = np.float32(
                    np.abs(
                        fitter.res["channel" + str(i)].pupil[0]
                    )
                    > 0.0
                )
            else:
                aperture = np.float32(
                    np.abs(fitter.res["channel" + str(i)].pupil) > 0.0
                )

            line, = ax1.plot(zcoeff[0], ".-")
            ax2.plot(zcoeff[1], ".-")
            ax1.set_xlabel("zernike polynomial")
            ax1.set_ylabel("coefficient")
            ax1.set_title("pupil magnitude")
            ax2.set_title("pupil phase")
            line.set_label("channel " + str(i))
            ax1.legend()
            if i == n_channel - 1:
                ax1.plot(
                    indz[mask], zcoeff[0][indz[mask]],
                    "ko", markersize=6, mfc="none",
                )
                ax2.plot(
                    indz[mask], zcoeff[1][indz[mask]],
                    "ko", markersize=6, mfc="none",
                )

            for k in range(Nzk):
                textstr[k] = "\n".join(
                    (
                        textstr[k],
                        r"$\mathrm{ch}%d=%.2f$"
                        % (i, zcoeff[1][indz[k]],),
                    )
                )

            ax3 = fig_pupil_mag.add_subplot(1, n_channel, i + 1)
            pupil_mag = (
                np.sum(
                    Zk * zcoeff[0].reshape((-1, 1, 1)), axis=0
                )
                * aperture
            )
            h = ax3.imshow(pupil_mag, interpolation="nearest")
            ax3.axis("off")
            ax3.set_title("pupil magnitude " + str(i), fontsize=20)
            fig_pupil_mag.colorbar(h, ax=ax3)

            ax4 = fig_pupil_phase.add_subplot(1, n_channel, i + 1)
            pupil_phase = (
                np.sum(
                    Zk[4:] * zcoeff[1][4:].reshape((-1, 1, 1)), axis=0
                )
                * aperture
            )
            h1 = ax4.imshow(pupil_phase, interpolation="nearest")
            ax4.axis("off")
            ax4.set_title("pupil phase " + str(i), fontsize=20)
            fig_pupil_phase.colorbar(h1, ax=ax4)

        bbox = dict(
            boxstyle="round", fc="blanchedalmond", ec="orange", alpha=0.5
        )
        for k in range(len(textstr)):
            ax5.text(
                0.01 + k * 0.35, 0.9, textstr[k], fontsize=12,
                bbox=bbox, transform=ax5.transAxes,
                horizontalalignment="left", verticalalignment="top",
            )
        ax5.set_axis_off()

        return [fig_coeff, fig_pupil_mag, fig_pupil_phase]

    @staticmethod
    def _plot_zernike_4pi(
        fitter,
    ) -> List[matplotlib.figure.Figure]:
        n_channel = fitter.rois.psf_data.shape[0]
        indz = np.array([4, 5, 6, 7, 10, 21])
        textstr = _zernike_labels()

        n_max = fitter.res.channel0.zernike_coeff_phase.shape[-1] - 1
        n_max_approx = int(np.sqrt(2 * n_max))
        Nk = (n_max_approx + 1) * (n_max_approx + 2) // 2
        mask = indz < Nk
        Nzk = int(np.sum(mask))

        for k in range(Nzk):
            textstr[k] = "\n".join(
                (
                    textstr[k],
                    r"$\mathrm{upper}=%.2f$"
                    % (fitter.res.channel0.zernike_coeff_phase[0][indz[k]],),
                    r"$\mathrm{lower}=%.2f$"
                    % (fitter.res.channel0.zernike_coeff_phase[1][indz[k]],),
                )
            )

        fig_coeff = plt.figure(figsize=[12, 10])
        ax1 = fig_coeff.add_subplot(3, 2, 1)
        ax2 = fig_coeff.add_subplot(3, 2, 2)
        ax3 = fig_coeff.add_subplot(3, 2, 3)
        ax4 = fig_coeff.add_subplot(3, 2, 4)
        ax5 = fig_coeff.add_subplot(3, 2, 5)

        for i in range(n_channel):
            zcoeff_mag = fitter.res[
                "channel" + str(i)
            ].zernike_coeff_mag
            zcoeff_phase = fitter.res[
                "channel" + str(i)
            ].zernike_coeff_phase
            line, = ax1.plot(zcoeff_mag[0], ".-")
            ax2.plot(zcoeff_phase[0], ".-")
            ax2.set_ylim((-0.6, 0.6))
            ax3.plot(zcoeff_mag[1], ".-")
            ax4.plot(zcoeff_phase[1], ".-")
            ax4.set_ylim((-0.6, 0.6))
            ax3.set_xlabel("zernike polynomial")
            ax3.set_ylabel("coefficient")
            ax1.set_title("upper pupil magnitude")
            ax2.set_title("upper pupil phase")
            ax3.set_title("lower pupil magnitude")
            ax4.set_title("lower pupil phase")
            line.set_label("channel " + str(i))
            ax1.legend()
            if i == n_channel - 1:
                ax1.plot(
                    indz[mask], zcoeff_mag[0][indz[mask]],
                    "ko", markersize=6, mfc="none",
                )
                ax2.plot(
                    indz[mask], zcoeff_phase[0][indz[mask]],
                    "ko", markersize=6, mfc="none",
                )
                ax3.plot(
                    indz[mask], zcoeff_mag[1][indz[mask]],
                    "ko", markersize=6, mfc="none",
                )
                ax4.plot(
                    indz[mask], zcoeff_phase[1][indz[mask]],
                    "ko", markersize=6, mfc="none",
                )

        bbox = dict(
            boxstyle="round", fc="blanchedalmond", ec="orange", alpha=0.5
        )
        for k in range(len(textstr)):
            ax5.text(
                0.01 + k * 0.35, 0.9, textstr[k], fontsize=12,
                bbox=bbox, transform=ax5.transAxes,
                horizontalalignment="left", verticalalignment="top",
            )
        ax5.set_axis_off()

        aperture = np.float32(
            np.abs(fitter.res.channel0.pupil1) > 0.0
        )
        Zk = fitter.res.channel0.zernike_polynomial

        fig_mag = plt.figure(figsize=[20, 8])
        for i in range(n_channel):
            ax = fig_mag.add_subplot(2, 4, i + 1)
            pupil_mag = (
                np.sum(
                    Zk
                    * fitter.res[
                        "channel" + str(i)
                    ].zernike_coeff_mag[0].reshape((-1, 1, 1)),
                    axis=0,
                )
                * aperture
            )
            ax.imshow(pupil_mag, interpolation="nearest")
            ax.axis("off")
            ax.set_title("upper pupil magnitude " + str(i), fontsize=20)
            fig_mag.colorbar(ax.images[0], ax=ax)
            ax = fig_mag.add_subplot(2, 4, i + 5)
            pupil_mag = (
                np.sum(
                    Zk
                    * fitter.res[
                        "channel" + str(i)
                    ].zernike_coeff_mag[1].reshape((-1, 1, 1)),
                    axis=0,
                )
                * aperture
            )
            ax.imshow(pupil_mag, interpolation="nearest")
            ax.axis("off")
            ax.set_title("lower pupil magnitude " + str(i), fontsize=20)
            fig_mag.colorbar(ax.images[0], ax=ax)

        fig_phase = plt.figure(figsize=[20, 8])
        for i in range(n_channel):
            ax = fig_phase.add_subplot(2, 4, i + 1)
            pupil_phase = (
                np.sum(
                    Zk[4:]
                    * fitter.res[
                        "channel" + str(i)
                    ].zernike_coeff_phase[0][4:].reshape((-1, 1, 1)),
                    axis=0,
                )
                * aperture
            )
            ax.imshow(pupil_phase, interpolation="nearest")
            ax.axis("off")
            ax.set_title("upper pupil phase " + str(i), fontsize=20)
            fig_phase.colorbar(ax.images[0], ax=ax)
            ax = fig_phase.add_subplot(2, 4, i + 5)
            pupil_phase = (
                np.sum(
                    Zk[4:]
                    * fitter.res[
                        "channel" + str(i)
                    ].zernike_coeff_phase[1][4:].reshape((-1, 1, 1)),
                    axis=0,
                )
                * aperture
            )
            ax.imshow(pupil_phase, interpolation="nearest")
            ax.axis("off")
            ax.set_title("lower pupil phase " + str(i), fontsize=20)
            fig_phase.colorbar(ax.images[0], ax=ax)

        return [fig_coeff, fig_mag, fig_phase]

    # ── Zernike map ──────────────────────────────────────────────────────

    def plot_zernike_map(
        self, fitter, param: DictConfig, index: Optional[list] = None
    ) -> Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]:
        """Plot spatially-resolved Zernike coefficient maps.

        Parameters
        ----------
        fitter : Fitter
            Fitter object with ``res`` and ``rois`` attributes.
        param : DictConfig
            Experiment parameters.
        index : list of int, optional
            Zernike indices to display.

        Returns
        -------
        Figure or list of Figure
        """
        if param.channeltype == "single":
            return self._zernike_map_single(fitter, index)

        if param.channeltype in ("multi", "4pi"):
            n_channel = fitter.rois.psf_data.shape[0]
            figs = []
            for i in range(n_channel):
                zmap = fitter.res["channel" + str(i)].zernike_map
                zcoeff = fitter.res["channel" + str(i)].zernike_coeff
                pupil = fitter.res["channel" + str(i)].pupil
                Zk = fitter.res["channel" + str(i)].zernike_polynomial
                figs.append(_zernike_map(fitter, index, zmap, zcoeff, pupil, Zk))
            return figs

        return plt.figure()

    @staticmethod
    def _zernike_map_single(
        fitter, index: Optional[list] = None
    ) -> matplotlib.figure.Figure:
        zmap = fitter.res.zernike_map
        zcoeff = fitter.res.zernike_coeff
        pupil = fitter.res.pupil
        Zk = fitter.res.zernike_polynomial
        return _zernike_map(fitter, index, zmap, zcoeff, pupil, Zk)

    # ── PSF vs data ──────────────────────────────────────────────────────

    def plot_psf_vs_data(
        self, fitter, param: DictConfig, index: int
    ) -> matplotlib.figure.Figure:
        """Compare measured PSF data with the fitted model for bead data.

        Parameters
        ----------
        fitter : Fitter
            Fitter object with ``res`` and ``rois`` attributes.
        param : DictConfig
            Experiment parameters.
        index : int
            Bead index to display.

        Returns
        -------
        Figure
        """
        psf_data = fitter.rois.psf_data
        psf_fit = fitter.rois.psf_fit

        if param.channeltype == "single":
            im1 = psf_data[index]
            im2 = psf_fit[index]
            fig = _psf_compare(im1, im2, param.pixel_size.z)
        else:
            n_channel = psf_data.shape[0]
            figs = []
            for ch in range(n_channel):
                if param.channeltype == "4pi":
                    im1 = psf_data[ch, index, 0]
                    im2 = psf_fit[ch, index, 0]
                else:
                    im1 = psf_data[ch, index]
                    im2 = psf_fit[ch, index]
                figs.append(
                    _psf_compare(im1, im2, param.pixel_size.z)
                )
            fig = figs[0]

        try:
            cor = fitter.res.cor
        except AttributeError:
            cor = fitter.res.channel1.cor

        imsz = fitter.rois.image_size
        fig_pos = plt.figure(figsize=[4, 4])
        ax = fig_pos.add_subplot(111)
        ax.plot(cor[index, -1], cor[index, -2], "ro")
        ax.set_xlim(0, imsz[-1])
        ax.set_ylim(0, imsz[-2])
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        ax.legend(["bead" + str(index)])

        return fig_pos

    def plot_psf_vs_data_insitu(
        self, fitter, param: DictConfig
    ) -> Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]:
        """Compare measured PSF data with the fitted model for insitu data.

        Parameters
        ----------
        fitter : Fitter
            Fitter object with ``res`` and ``rois`` attributes.
        param : DictConfig
            Experiment parameters.

        Returns
        -------
        Figure or list of Figure
        """
        if param.channeltype == "single":
            rois = fitter.rois.psf_data
            I_model = fitter.res.I_model
            zf = fitter.res.pos[:, 0]
            Nz = I_model.shape[0]
            edge = np.real(fitter.res.zoffset) + range(0, Nz + 1)
            ind = np.digitize(zf, np.array(edge).flatten())
            rois_avg = np.zeros(I_model.shape)
            for ii in range(1, Nz + 1):
                mask = ind == ii
                if sum(mask) > 0:
                    rois_avg[ii - 1] = np.mean(rois[mask], axis=0)
            return _psf_compare(rois_avg, I_model, param.pixel_size.z)

        n_channel = fitter.rois.psf_data.shape[0]
        zoffset = fitter.res.channel0.zoffset
        figs = []
        for ch in range(n_channel):
            rois = fitter.rois.psf_data[ch]
            I_model = fitter.res["channel" + str(ch)].I_model
            if param.channeltype == "4pi":
                I_model = fitter.res["channel" + str(ch)].psf_model
            zf = fitter.res.channel0.pos[:, 0]
            Nz = I_model.shape[0]
            edge = np.real(zoffset) + range(0, Nz + 1)
            ind = np.digitize(zf, np.array(edge).flatten())
            rois_avg = np.zeros(I_model.shape)
            for ii in range(1, Nz + 1):
                mask = ind == ii
                if sum(mask) > 0:
                    rois_avg[ii - 1] = np.mean(rois[mask], axis=0)
            figs.append(
                _psf_compare(rois_avg, I_model, param.pixel_size.z)
            )
        return figs

    # ── Localization ─────────────────────────────────────────────────────

    def plot_localization(
        self, fitter, param: DictConfig
    ) -> Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]:
        """Plot localisation bias curves.

        Parameters
        ----------
        fitter : Fitter
            Fitter object with ``locres`` attribute.
        param : DictConfig
            Experiment parameters.

        Returns
        -------
        Figure or list of Figure
        """
        loc = fitter.locres.loc
        fig = _plot_loc_bias(loc, param)
        if hasattr(fitter.locres, "loc_FD"):
            loc = fitter.locres.loc_FD
            fig_fd = _plot_loc_bias(loc, param)
            return [fig, fig_fd]
        return fig

    # ── Transform ────────────────────────────────────────────────────────

    def plot_transform(
        self, fitter
    ) -> matplotlib.figure.Figure:
        """Plot the channel-to-channel coordinate transform.

        Parameters
        ----------
        fitter : Fitter
            Fitter object with ``res`` and ``rois`` attributes.

        Returns
        -------
        Figure
        """
        n_channel = fitter.rois.psf_data.shape[0]
        ref_pos = fitter.res.channel0.pos
        dxy = fitter.res.xyshift

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
            pos = fitter.res["channel" + str(i)].pos
            if n_channel < 3:
                cor_target = (
                    np.matmul(cor_ref - fitter.res.imgcenter, fitter.res.T)[
                        ..., :-1
                    ]
                    + fitter.res.imgcenter[:-1]
                )
            else:
                cor_target = (
                    np.matmul(
                        cor_ref - fitter.res.imgcenter,
                        fitter.res.T[i - 1],
                    )[..., :-1]
                    + fitter.res.imgcenter[:-1]
                )

            ax = fig.add_subplot(spec[i])
            ax.plot(ref_pos[:, -1], ref_pos[:, -2], ".")
            ax.plot(
                pos[:, -1] - dxy[i][-1],
                pos[:, -2] - dxy[i][-2],
                "o", markersize=8, mfc="none",
            )
            ax.plot(fitter.res.imgcenter[1], fitter.res.imgcenter[0], "*")
            ax.set_xlabel("x (pixel)")
            ax.set_ylabel("y (pixel)")
            ax.set_title("channel" + str(i))

            ax1 = fig.add_subplot(spec[n_channel + i])
            ax1.plot(cor_target[:, -1], cor_target[:, -2], ".")
            ax1.plot(pos[:, -1], pos[:, -2], "o", markersize=8, mfc="none")
            ax1.plot(
                fitter.res.imgcenter[1], fitter.res.imgcenter[0], "*"
            )
            ax1.set_xlabel("x (pixel)")
            ax1.set_ylabel("y (pixel)")

        ax.legend(["ref", "target", "center"])
        ax1.legend(["ref_trans", "target", "center"])

        return fig

    # ── PSF display ──────────────────────────────────────────────────────

    def plot_psf(
        self, fitter, param: DictConfig
    ) -> Union[matplotlib.figure.Figure, List[matplotlib.figure.Figure]]:
        """Display the learned PSF model.

        Parameters
        ----------
        fitter : Fitter
            Fitter object with ``res`` and ``rois`` attributes.
        param : DictConfig
            Experiment parameters.

        Returns
        -------
        Figure or list of Figure
        """
        if param.channeltype == "single":
            im1 = fitter.res.I_model
            return _psf_display(im1, param.pixel_size.z)

        n_channel = fitter.rois.psf_data.shape[0]
        figs = []
        for ch in range(n_channel):
            if param.channeltype == "4pi":
                im1 = fitter.res["channel" + str(ch)].psf_model
            else:
                im1 = fitter.res["channel" + str(ch)].I_model
            figs.append(_psf_display(im1, param.pixel_size.z))
        return figs

    # ── Coordinates ──────────────────────────────────────────────────────

    def plot_coordinates(
        self, fitter, param: DictConfig
    ) -> matplotlib.figure.Figure:
        """Plot bead / emitter coordinates on the FOV.

        Parameters
        ----------
        fitter : Fitter
            Fitter object with ``res`` and ``rois`` attributes.
        param : DictConfig
            Experiment parameters.

        Returns
        -------
        Figure
        """
        if param.channeltype == "single":
            fig = plt.figure(figsize=[5, 5])
            cor = fitter.res.cor
            cor_all = fitter.res.cor_all
            ax = fig.add_subplot(111)
            ax.plot(cor_all[:, -1], cor_all[:, -2], ".")
            ax.plot(cor[:, -1], cor[:, -2], "o", markersize=8, mfc="none")
            ax.set_xlabel("x (pixel)")
            ax.set_ylabel("y (pixel)")
            ax.legend(["all", "selected"])
        else:
            n_channel = fitter.rois.psf_data.shape[0]
            fig = plt.figure(figsize=[5 * n_channel, 5])
            spec = gridspec.GridSpec(
                ncols=n_channel, nrows=1,
                width_ratios=list(np.ones(n_channel)), wspace=0.3,
                hspace=0.2, height_ratios=[1],
            )
            for i in range(n_channel):
                cor = fitter.res["channel" + str(i)].cor
                cor_all = fitter.res["channel" + str(i)].cor_all
                ax = fig.add_subplot(spec[i])
                ax.plot(cor_all[:, -1], cor_all[:, -2], ".")
                ax.plot(
                    cor[:, -1], cor[:, -2],
                    "o", markersize=8, mfc="none",
                )
                ax.set_xlabel("x (pixel)")
                ax.set_ylabel("y (pixel)")
                ax.set_title("channel" + str(i))
            ax.legend(["all", "selected"])

        return fig

    # ── Report generation ────────────────────────────────────────────────

    def generate_report(
        self,
        f,
        p,
        output_dir: str,
        index: int = 0,
        fmt: str = "png",
        dpi: int = 150,
    ) -> Dict[str, List[str]]:
        """Generate all standard plots and save them to *output_dir*.

        Parameters
        ----------
        f : Fitter
            Fitter / result object (has ``res``, ``rois``, ``locres``).
        p : DictConfig
            Experiment parameters.
        output_dir : str
            Directory where plot images are written.
        index : int
            Bead / ROI index for PSF-vs-data plots.
        fmt : str
            Image format (``"png"``, ``"pdf"``, …).
        dpi : int
            Resolution.

        Returns
        -------
        dict
            Mapping of plot name → list of saved file paths.
        """
        saved: Dict[str, List[str]] = {}

        fig = self.plot_psf_vs_data(f, p, index=index)
        saved["psf_vs_data"] = save_figs(fig, output_dir, "psf_vs_data", fmt, dpi)

        fig = self.plot_localization(f, p)
        saved["localization"] = save_figs(fig, output_dir, "localization", fmt, dpi)

        try:
            figs = self.plot_zernike(f, p)
            saved["zernike"] = save_figs(figs, output_dir, "zernike", fmt, dpi)
        except Exception:
            try:
                figs = self.plot_pupil(f, p)
                saved["pupil"] = save_figs(figs, output_dir, "pupil", fmt, dpi)
            except Exception:
                print("no pupil / zernike plot available")

        fig = self.plot_learned_params(f, p)
        saved["learned_params"] = save_figs(fig, output_dir, "learned_params", fmt, dpi)

        fig = self.plot_coordinates(f, p)
        saved["coordinates"] = save_figs(fig, output_dir, "coordinates", fmt, dpi)

        return saved

    # ── Strehl ratio ─────────────────────────────────────────────────────

    @staticmethod
    def plot_strehl_ratio(
        strehl_map: np.ndarray,
    ) -> matplotlib.figure.Figure:
        """Plot a Strehl-ratio map (for field-dependent PSFs).

        Parameters
        ----------
        strehl_map : numpy.ndarray
            2-D Strehl ratio map.

        Returns
        -------
        Figure
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(strehl_map)
        fig.colorbar(im, ax=ax)
        ax.set_title("Strehl ratio map", fontsize=15)
        return fig

    # ── FWHM ─────────────────────────────────────────────────────────────

    def plot_fwhm(
        self,
        param: DictConfig,
        fwhmx: np.ndarray,
        fwhmy: np.ndarray,
        fwhmz: np.ndarray,
        I_model: Optional[np.ndarray] = None,
    ) -> matplotlib.figure.Figure:
        """Plot FWHM curves.

        Parameters
        ----------
        param : DictConfig
            Experiment parameters.
        fwhmx, fwhmy, fwhmz : numpy.ndarray
            FWHM values along x, y, z.
        I_model : numpy.ndarray, optional
            PSF model (used for single-channel intensity plots).

        Returns
        -------
        Figure
        """
        if I_model is not None and I_model.ndim >= 3:
            return self._plot_fwhm_single(param, I_model, fwhmx, fwhmy, fwhmz)
        return self._plot_fwhm_simple(param, fwhmx, fwhmy, fwhmz)

    @staticmethod
    def _plot_fwhm_single(
        param: DictConfig,
        I_model: np.ndarray,
        fwhmx: np.ndarray,
        fwhmy: np.ndarray,
        fwhmz: np.ndarray,
    ) -> matplotlib.figure.Figure:
        from .analysis import getfwhm

        Imaxh = np.max(I_model) / 2
        Ix, xh, Iy, yh, Iz, zh = getfwhm(I_model)
        fwhmx_val = np.diff(xh) * param.pixel_size.x * 1e3
        fwhmy_val = np.diff(yh) * param.pixel_size.y * 1e3
        fwhmz_val = np.diff(zh) * param.pixel_size.z * 1e3

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

    @staticmethod
    def _plot_fwhm_simple(
        param: DictConfig,
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

    # ── Z-slices (from iterative learning) ───────────────────────────────

    @staticmethod
    def plot_z_slices(
        I_model: np.ndarray, step: int = 4
    ) -> matplotlib.figure.Figure:
        """Display z-slices of *I_model* (every *step*-th slice).

        Parameters
        ----------
        I_model : numpy.ndarray
            3-D PSF model volume.
        step : int
            Step size between displayed slices.

        Returns
        -------
        Figure
        """
        Nz = I_model.shape[-3]
        zind = range(0, Nz, step)
        fig = plt.figure(figsize=[3 * len(zind), 3])
        for i, idx in enumerate(zind):
            ax = fig.add_subplot(1, len(zind), i + 1)
            ax.imshow(I_model[idx], cmap="twilight")
            ax.axis("off")
        return fig


# ── Module-level helpers ─────────────────────────────────────────────────


def _zernike_labels() -> list:
    """Return human-readable labels for common Zernike polynomials."""
    return [
        r"$\mathrm{D \ astigmatism}$",
        r"$\mathrm{astigmatism}$",
        r"$\mathrm{V \ coma}$",
        r"$\mathrm{H \ coma}$",
        r"$\mathrm{spherical}$",
        r"$\mathrm{2nd \ spherical}$",
    ]


def _psf_compare(
    im1: np.ndarray, im2: np.ndarray, pz: float
) -> matplotlib.figure.Figure:
    """Side-by-side comparison of data vs model PSF z-slices."""
    Nz = im1.shape[0]
    zrange = np.linspace(-Nz / 2 + 0.5, Nz / 2 - 0.5, Nz) * pz
    zind = range(0, Nz, 4)
    cc = im1.shape[-1] // 2
    N = len(zind) + 1
    fig = plt.figure(figsize=[3 * N, 6])
    for i, idx in enumerate(zind):
        ax = fig.add_subplot(2, N, i + 1)
        ax.imshow(im1[idx], cmap="twilight")
        ax.axis("off")
        ax.set_title(
            str(np.round(zrange[idx], 2)) + r"$\ \mu$m", fontsize=30
        )
        ax = fig.add_subplot(2, N, i + 1 + N)
        ax.imshow(im2[idx], cmap="twilight")
        ax.axis("off")
    ax = fig.add_subplot(2, N, N)
    ax.imshow(im1[:, cc], cmap="twilight")
    ax.axis("off")
    ax.set_title("xz", fontsize=30)
    fig.colorbar(ax.images[0], ax=ax)
    ax = fig.add_subplot(2, N, 2 * N)
    ax.imshow(im2[:, cc], cmap="twilight")
    ax.axis("off")
    fig.colorbar(ax.images[0], ax=ax)
    return fig


def _psf_display(
    im1: np.ndarray, pz: float
) -> matplotlib.figure.Figure:
    """Display z-slices of a PSF model."""
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
    loc, param: DictConfig
) -> matplotlib.figure.Figure:
    """Plot localisation bias in x, y, and z."""
    Nz = loc.z.shape[1]
    fig = plt.figure(figsize=[16, 4])
    spec = gridspec.GridSpec(
        ncols=3, nrows=1,
        width_ratios=[3, 3, 3], wspace=0.3,
        hspace=0.3, height_ratios=[1],
    )

    ax = fig.add_subplot(spec[0])
    ax.plot(loc.x.transpose() * param.pixel_size.x * 1e3, "k", alpha=0.1)
    ax.plot(loc.x[0] * 0.0, "r")
    ax.set_xlabel("z slice")
    ax.set_ylabel("x bias (nm)")

    ax = fig.add_subplot(spec[1])
    ax.plot(loc.y.transpose() * param.pixel_size.y * 1e3, "k", alpha=0.1)
    ax.plot(loc.y[0] * 0.0, "r")
    ax.set_xlabel("z slice")
    ax.set_ylabel("y bias (nm)")

    ax = fig.add_subplot(spec[2])
    bias_z = (
        (loc.z - np.linspace(0, Nz - 1, Nz))
        * param.pixel_size.z
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


def _zernike_map(fitter, index, zmap, zcoeff, pupil, Zk):
    """Build a zernike-map figure for one channel."""
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
    imsz = np.array(fitter.rois.image_size)

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
