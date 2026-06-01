import numpy as np


def psf2IAB(ROIs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    G = np.zeros(ROIs.shape, dtype=np.complex64)
    G[:, 0] = (
        ROIs[:, 0] * np.exp(-2 * np.pi / 3 * 1j)
        + ROIs[:, 1]
        + ROIs[:, 2] * np.exp(2 * np.pi / 3 * 1j)
    )
    G[:, 1] = np.sum(ROIs, axis=1)
    G[:, 2] = (
        ROIs[:, 0] * np.exp(2 * np.pi / 3 * 1j)
        + ROIs[:, 1]
        + ROIs[:, 2] * np.exp(-2 * np.pi / 3 * 1j)
    )

    I = np.real(G[:, 1]) / 3
    A = G[:, 0] / 3
    B = G[:, 2] / 3

    a = np.squeeze(np.sum(np.real(A[0]), axis=(-1, -2)))
    b = np.squeeze(np.sum(np.imag(A[0]), axis=(-1, -2)))
    y1 = np.squeeze(np.sum((ROIs[:, 2] - ROIs[:, 0]) / np.sqrt(3), axis=(-1, -2)))
    y2 = np.squeeze(
        np.sum(ROIs[:, 1] - np.sum(ROIs, axis=1) / 3, axis=(-1, -2))
    )
    q = 1j * (a * y1 - b * y2) + (a * y2 + b * y1)
    if len(q.shape) > 1:
        phi = np.median(np.angle(q), axis=1)
    else:
        phi = np.median(np.angle(q))

    return I, A, B, phi
