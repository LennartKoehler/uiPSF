import numpy as np
from math import factorial


def nl2noll(n: int, l: int) -> int:
    """Convert Zernike (n, l) indices to Noll single-index j.

    Uses the Noll ordering convention (Noll, 1976) where j=1 is piston,
    j=2 is tip, j=3 is tilt, etc. The azimuthal order l follows the
    sign convention: even j → l ≤ 0 (sin terms), odd j → l ≥ 0 (cos terms).

    Parameters
    ----------
    n : int
        Radial degree (n ≥ 0).
    l : int
        Azimuthal frequency (|l| ≤ n, same parity as n).

    Returns
    -------
    int
        Noll index j (1-based).
    """
    azimuthal_order = abs(l)
    j = n * (n + 1) // 2 + 1 + max(0, azimuthal_order - 1)
    if ((l > 0) and (n % 4 >= 2)) or ((l < 0) and (n % 4 <= 1)):
        j += 1
    return np.int32(j)


def noll2nl(j: int) -> tuple[int, int]:
    """Convert Noll single-index j to Zernike (n, l) indices.

    Inverse of nl2noll. See Noll (1976) for the ordering convention.

    Parameters
    ----------
    j : int
        Noll index (1-based, j ≥ 1).

    Returns
    -------
    tuple[int, int]
        (radial degree n, azimuthal frequency l).
    """
    n = int(np.ceil((-3 + np.sqrt(1 + 8 * j)) / 2))
    l = j - n * (n + 1) // 2 - 1
    if n % 2 != l % 2:
        l += 1
    if j % 2 == 1:
        l = -l
    return np.int32(n), np.int32(l)


def radialpoly(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Compute the radial component of a Zernike polynomial.

    Evaluates the radial polynomial R_n^m(rho) defined as:

        R_n^m(rho) = sum_{k=0}^{(n-m)/2} (-1)^k (n-k)! /
                     [k! ((n+m)/2 - k)! ((n-m)/2 - k)!] * rho^{n-2k}

    The result includes the normalization factor so that the full
    Zernike polynomial Z_j = N * R_n^m * {cos/sin}(m*phi) is
    orthonormal over the unit disk (integral of |Z_j|^2 = pi).

    Parameters
    ----------
    n : int
        Radial degree.
    m : int
        Azimuthal frequency (m ≥ 0, must have same parity as n).
    rho : np.ndarray
        Radial coordinate (normalized, rho=1 at pupil edge).

    Returns
    -------
    np.ndarray
        Normalized radial polynomial evaluated at rho.
    """
    if m == 0:
        norm_factor = np.sqrt(n + 1)
    else:
        norm_factor = np.sqrt(2 * n + 2)

    result = np.zeros(rho.shape)
    for k in range(0, (n - m) // 2 + 1):
        coeff = (
            norm_factor
            * ((-1) ** k)
            * factorial(n - k)
            / factorial(k)
            / factorial((n + m) // 2 - k)
            / factorial((n - m) // 2 - k)
        )
        result += coeff * rho ** (n - 2 * k)
    return result


def genZern1(n_max: int, xsz: int) -> np.ndarray:
    """Generate a complete set of orthonormal Zernike polynomials on a grid.

    Computes all Zernike polynomials up to radial degree n_max on an
    xsz × xsz grid. The polynomials are ordered by the Noll convention
    (j = 1, 2, ..., Nk) and are orthonormal over the unit disk.

    Z_j(rho, phi) = R_n^|l|(rho) * cos(m*phi)   if l ≥ 0
                    R_n^|l|(rho) * sin(m*phi)   if l < 0

    Parameters
    ----------
    n_max : int
        Maximum radial degree of Zernike polynomials to generate.
    xsz : int
        Grid size (number of pixels along each axis).

    Returns
    -------
    np.ndarray
        Array of shape (Nk, xsz, xsz) where Nk = (n_max+1)*(n_max+2)/2
        is the total number of Zernike modes. Z[j] contains the j-th
        Noll-ordered Zernike polynomial.
    """
    Nk = (n_max + 1) * (n_max + 2) // 2
    Z = np.ones((Nk, xsz, xsz))

    pixel_to_normalized = 2 / xsz
    pixel_coords = np.linspace(-xsz / 2 + 0.5, xsz / 2 - 0.5, xsz)
    xx, yy = np.meshgrid(pixel_coords, pixel_coords)

    rho = np.lib.scimath.sqrt((xx * pixel_to_normalized) ** 2 + (yy * pixel_to_normalized) ** 2)
    phi = np.arctan2(yy, xx)

    for j in range(Nk):
        n, l = noll2nl(j + 1)
        m = abs(l)
        radial = radialpoly(n, m, rho)
        if l < 0:
            Z[j] = radial * np.sin(phi * m)
        else:
            Z[j] = radial * np.cos(phi * m)
    return Z
