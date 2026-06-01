import torch
import numpy as np


def _poisson_log_likelihood(model, data):
    ll = model - data - data * torch.log(model) + data * torch.log(data)
    return torch.mean(ll[torch.isfinite(ll)])


def _neg_penalty(x):
    return torch.sum(torch.square(torch.clamp(x, max=0)))


def mse_real_zernike(model, data, variables=None, mu=None, w=None):
    mydiff = model - data
    bg = variables[1]
    intensity = variables[2]
    gxymean = torch.mean(torch.abs(torch.from_numpy(variables[-1].astype(np.float32)))) if not isinstance(variables[-1], torch.Tensor) else torch.mean(torch.abs(variables[-1]))

    bg_t = torch.from_numpy(bg.astype(np.float32)) if not isinstance(bg, torch.Tensor) else bg
    intensity_t = torch.from_numpy(intensity.astype(np.float32)) if not isinstance(intensity, torch.Tensor) else intensity

    bgmin = _neg_penalty(bg_t)
    intensitymin = _neg_penalty(intensity_t)

    LL = _poisson_log_likelihood(model, data)
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + gxymean * w[8]
    return loss


def mse_real_zernike_FD(model, data, variables=None, mu=None, w=None):
    mydiff = model - data
    bg = variables[1]
    intensity = variables[2]
    gxymean = torch.mean(torch.abs(torch.from_numpy(variables[-1].astype(np.float32)))) if not isinstance(variables[-1], torch.Tensor) else torch.mean(torch.abs(variables[-1]))

    bg_t = torch.from_numpy(bg.astype(np.float32)) if not isinstance(bg, torch.Tensor) else bg
    intensity_t = torch.from_numpy(intensity.astype(np.float32)) if not isinstance(intensity, torch.Tensor) else intensity

    bgmin = _neg_penalty(bg_t)
    intensitymin = _neg_penalty(intensity_t)

    Zmap = variables[3]
    Zmap_t = torch.from_numpy(Zmap.astype(np.float32)) if not isinstance(Zmap, torch.Tensor) else Zmap
    dfxy = torch.sum(torch.square(torch.diff(Zmap_t, n=1, dim=-1))) + torch.sum(
        torch.square(torch.diff(Zmap_t, n=1, dim=-2))
    )

    LL = _poisson_log_likelihood(model, data)
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfxy * w[2] + gxymean * w[8]
    return loss


def mse_zernike_4pi(model, data, variables=None, mu=None, w=None):
    bg = variables[1]
    intensity = variables[2]
    alpha = variables[7]
    gxymean = torch.mean(torch.abs(torch.from_numpy(variables[-1].astype(np.float32)))) if not isinstance(variables[-1], torch.Tensor) else torch.mean(torch.abs(variables[-1]))

    bg_t = torch.from_numpy(bg.astype(np.float32)) if not isinstance(bg, torch.Tensor) else bg
    intensity_t = torch.from_numpy(intensity.astype(np.float32)) if not isinstance(intensity, torch.Tensor) else intensity
    alpha_t = torch.from_numpy(alpha.astype(np.float32)) if not isinstance(alpha, torch.Tensor) else alpha

    bgmin = _neg_penalty(bg_t)
    intensitymin = _neg_penalty(intensity_t)
    alphamin = _neg_penalty(alpha_t)

    LL = _poisson_log_likelihood(model, data)
    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + alphamin * w[4] * mu + gxymean * w[8]
    return loss


def mse_real_All(model, data, loss_func, variables=None, mu=None, w=None, psfnorm=None):
    varsize = len(variables)
    var = [None] * (varsize - 1)
    loss = 0.0
    for i in range(model.shape[0]):
        for j in range(1, varsize - 1):
            var[j] = variables[j][i]
        var[0] = variables[0]
        if psfnorm:
            loss += loss_func(model[i], data[i], var, mu, w, psfnorm[i])
        else:
            loss += loss_func(model[i], data[i], var, mu, w)
    return loss


def mse_real_4pi_All(model, data, loss_func, variables=None, mu=None, w=None, psfnorm=None):
    return mse_real_All(model, data, loss_func, variables, mu, w, psfnorm)
