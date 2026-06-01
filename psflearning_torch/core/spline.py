import numpy as np
import scipy as sp
from scipy import ndimage


def psf2cspline_np(psf: np.ndarray) -> np.ndarray:
    A = np.zeros((64, 64))
    for i in range(1, 5):
        dx = (i - 1) / 3
        for j in range(1, 5):
            dy = (j - 1) / 3
            for k in range(1, 5):
                dz = (k - 1) / 3
                for l in range(1, 5):
                    for m in range(1, 5):
                        for n in range(1, 5):
                            A[
                                (i - 1) * 16 + (j - 1) * 4 + k - 1,
                                (l - 1) * 16 + (m - 1) * 4 + n - 1,
                            ] = dx ** (l - 1) * dy ** (m - 1) * dz ** (n - 1)

    psf_up = ndimage.zoom(psf, 3.0, mode="grid-constant", grid_mode=True)[
        1:-1, 1:-1, 1:-1
    ]
    A = np.float32(A)
    coeff = _calsplinecoeff(A, psf, psf_up)
    return coeff


def _calsplinecoeff(
    A: np.ndarray, psf: np.ndarray, psf_up: np.ndarray
) -> np.ndarray:
    coeff = np.zeros((64, psf.shape[0] - 1, psf.shape[1] - 1, psf.shape[2] - 1))
    for i in range(coeff.shape[1]):
        for j in range(coeff.shape[2]):
            for k in range(coeff.shape[3]):
                temp = psf_up[
                    i * 3 : 3 * (i + 1) + 1,
                    j * 3 : 3 * (j + 1) + 1,
                    k * 3 : 3 * (k + 1) + 1,
                ]
                x = sp.linalg.solve(A, temp.flatten())
                coeff[:, i, j, k] = x
    return coeff
