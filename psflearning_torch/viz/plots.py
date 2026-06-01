import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def showlearnedparam(f, p):
    res = f.res if hasattr(f, "res") else None
    if res is None:
        return
    pos = res[0]
    bg = np.squeeze(res[1])
    intensity = np.squeeze(res[2])
    fig = plt.figure(figsize=[12, 8])
    ax = fig.add_subplot(2, 2, 1)
    plt.plot(pos[:, -2], "o-", label="y")
    plt.plot(pos[:, -1], "o-", label="x")
    plt.legend()
    plt.title("Position offsets")
    ax = fig.add_subplot(2, 2, 2)
    plt.plot(pos[:, 0], "o-")
    plt.title("Z position")
    ax = fig.add_subplot(2, 2, 3)
    plt.plot(intensity, "o-")
    plt.title("Intensity")
    ax = fig.add_subplot(2, 2, 4)
    plt.plot(bg, "o-")
    plt.title("Background")
    plt.tight_layout()
    plt.show()


def showpsfvsdata(f, p, index=0):
    res = f.res if hasattr(f, "res") else None
    if res is None:
        return
    I_model = res[4] if len(res) > 4 else res[3]
    psf_data = f.rois[index] if f.rois is not None else None
    if psf_data is None:
        return
    psfcompare(psf_data, I_model, p.pixel_size.z)


def psfcompare(im1, im2, pz):
    fig = plt.figure(figsize=[12, 6])
    n_slices = min(im1.shape[0], 8)
    zind = np.linspace(0, im1.shape[0] - 1, n_slices, dtype=int)
    for i, zid in enumerate(zind):
        ax = fig.add_subplot(2, n_slices, i + 1)
        plt.imshow(im1[zid], cmap="hot")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Data")
        ax = fig.add_subplot(2, n_slices, n_slices + i + 1)
        plt.imshow(im2[zid] if zid < im2.shape[0] else np.zeros_like(im1[0]), cmap="hot")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Model")
    plt.tight_layout()
    plt.show()


def showzernike(f, p, index=None):
    res = f.res if hasattr(f, "res") else None
    if res is None:
        return
    if len(res) > 6 and hasattr(res[6], "shape"):
        zcoeff = np.squeeze(res[6])
        fig = plt.figure(figsize=[10, 4])
        ax = fig.add_subplot(1, 2, 1)
        plt.plot(zcoeff[0], "o-", label="mag")
        plt.plot(zcoeff[1], "o-", label="phase")
        plt.legend()
        plt.title("Zernike coefficients")
    plt.show()


def showpsf(f, p):
    res = f.res if hasattr(f, "res") else None
    if res is None:
        return
    I_model = res[4] if len(res) > 4 else res[3]
    if isinstance(I_model, np.ndarray):
        psfdisp(I_model, p.pixel_size.z)


def psfdisp(im1, pz):
    fig = plt.figure(figsize=[12, 3])
    n_slices = min(im1.shape[0], 8)
    zind = np.linspace(0, im1.shape[0] - 1, n_slices, dtype=int)
    for i, zid in enumerate(zind):
        ax = fig.add_subplot(1, n_slices + 1, i + 1)
        plt.imshow(im1[zid], cmap="hot")
        plt.axis("off")
    ax = fig.add_subplot(1, n_slices + 1, n_slices + 1)
    mid_y = im1.shape[-2] // 2
    plt.imshow(im1[:, mid_y, :], cmap="hot", aspect="auto")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
