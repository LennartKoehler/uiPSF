import torch
import numpy as np


def fft3d(tfin: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.fftshift(tfin, dim=(-3, -2, -1)), dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    )


def ifft3d(tfin: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(tfin, dim=(-3, -2, -1)), dim=(-3, -2, -1)
        ),
        dim=(-3, -2, -1),
    )


def fft2d(tfin: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.fftshift(tfin, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )


def ifft2d(tfin: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(tfin, dim=(-2, -1)), dim=(-2, -1)
        ),
        dim=(-2, -1),
    )


def prechirpz(
    kpixelsize: float,
    pixelsize_x: float,
    pixelsize_y: float,
    N: int,
    M: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    krange = np.linspace(-N / 2 + 0.5, N / 2 - 0.5, N, dtype=np.float32)
    xxK, yyK = np.meshgrid(krange, krange)
    xrange = np.linspace(-M / 2 + 0.5, M / 2 - 0.5, M, dtype=np.float32)
    xxR, yyR = np.meshgrid(xrange, xrange)

    a = 1j * np.pi * kpixelsize
    A = np.exp(a * (pixelsize_x * xxK * xxK + pixelsize_y * yyK * yyK))
    C = np.exp(a * (pixelsize_x * xxR * xxR + pixelsize_y * yyR * yyR))

    brange = np.linspace(-(N + M) / 2 + 1, (N + M) / 2 - 1, N + M - 1, dtype=np.float32)
    xxB, yyB = np.meshgrid(brange, brange)
    B = np.exp(-a * (pixelsize_x * xxB * xxB + pixelsize_y * yyB * yyB))
    Bh = np.fft.fft2(B)

    return A, Bh, C


def cztfunc(datain: torch.Tensor, param: tuple) -> torch.Tensor:
    A, Bh, C = param
    if isinstance(A, torch.Tensor):
        A_t = A.to(datain.device, datain.dtype)
        C_t = C.to(datain.device, datain.dtype)
        Bh_t = Bh.to(datain.device, datain.dtype)
    else:
        A_t = torch.from_numpy(A).to(datain.device, datain.dtype)
        C_t = torch.from_numpy(C).to(datain.device, datain.dtype)
        Bh_t = torch.from_numpy(Bh).to(datain.device, datain.dtype)

    N = A_t.shape[0]
    L = Bh_t.shape[0]
    M = C_t.shape[0]

    Apad = torch.cat(
        (A_t * datain / N, torch.zeros(*datain.shape[:-1], L - N, dtype=datain.dtype, device=datain.device)),
        dim=-1,
    )
    Apad = torch.cat(
        (Apad, torch.zeros(*Apad.shape[:-2], L - N, Apad.shape[-1], dtype=datain.dtype, device=datain.device)),
        dim=-2,
    )
    Ah = torch.fft.fftn(Apad, dim=(-2, -1))
    cztout = torch.fft.ifftn(Ah * Bh_t / L, dim=(-2, -1))
    dataout = C_t * cztout[..., -M:, -M:]
    return dataout
