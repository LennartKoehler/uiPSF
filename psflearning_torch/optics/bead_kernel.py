import numpy as np
import scipy.special as spf


def generate_bead_kernel(
    pixelsize_z: float,
    pixelsize_x: float,
    pixelsize_y: float,
    bead_radius: float,
    roi_shape: tuple,
    is_volume: bool = False,
    bin_val: int = 1,
) -> np.ndarray:
    if is_volume:
        Nz = roi_shape[-3]
        bin_val = 1
    else:
        Nz = roi_shape[-3] + int(bead_radius / pixelsize_z) * 2 + 4

    Lx = roi_shape[-1] * bin_val
    px_x = pixelsize_x / bin_val
    px_y = pixelsize_y / bin_val

    xrange = np.linspace(-Lx / 2 + 0.5, Lx / 2 - 0.5, Lx) + 1e-6
    zrange = np.linspace(-Nz / 2 + 0.5, Nz / 2 - 0.5, Nz)
    xx, yy, zz = np.meshgrid(xrange, xrange, zrange)
    xx = np.swapaxes(xx, 0, 2)
    yy = np.swapaxes(yy, 0, 2)
    zz = np.swapaxes(zz, 0, 2)

    pkx = 1 / Lx / px_x
    pky = 1 / Lx / px_y
    pkz = 1 / Nz / pixelsize_z

    if bead_radius > 0:
        Zk0 = np.sqrt((xx * pkx) ** 2 + (yy * pky) ** 2 + (zz * pkz) ** 2) * bead_radius
        mu = 1.5
        kernel = spf.jv(mu, 2 * np.pi * Zk0) / (Zk0**mu) * bead_radius**3
        kernel = kernel / np.max(kernel)
        kernel = np.float32(kernel)
    else:
        kernel = np.ones((Nz, Lx, Lx), dtype=np.float32)

    return kernel
