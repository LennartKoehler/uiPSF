"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved     

@author: Sheng Liu, Jonas Hellgoth

Loss functions for PSF optimization. Each function receives:
    - model: Forward PSF images
    - data: Actual measured images  
    - variables: List of optimizable parameters (see psf_variables.py for structure)
    - mu: Regularization weight
    - w: Loss function weights
"""

import tensorflow as tf

import numpy as np

from .psf_variables import (
    Pupil4PiLossVariables,
    PupilLossVariables,
    PupilSMLMLossVariables,
    Zernike4PiLossVariables,
    Zernike4PiSMLMLossVariables,
    ZernikeFDLossVariables,
    ZernikeFDSMLMLossVariables,
    ZernikeLossVariables,
    ZernikeSMLMLossVariables,
)


def mse_real(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss with regularization for Zernike PSF.

    *variables* may be a :class:`ZernikeLossVariables` or a plain list
    of tensors (7 elements, see :class:`ZernikeLossVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = ZernikeLossVariables.from_list(variables)

    mydiff = model - data
    mydiff = mydiff[:, 1:-1]
    data = data[:, 1:-1]
    model = model[:, 1:-1]
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    zernike_magnitude = variables.zernike_magnitude
    zernike_phase = variables.zernike_phase
    background = variables.backgrounds
    intensity = variables.intensities

    gxymean = tf.reduce_mean(tf.abs(zernike_phase))
    s = tf.math.reduce_sum(
        tf.math.square(zernike_magnitude[0] - zernike_magnitude[1]) +
        tf.math.square(zernike_magnitude[-1] - zernike_magnitude[-2])
    )

    dfz = tf.math.square(tf.experimental.numpy.diff(zernike_magnitude, n=1, axis=-3))
    dfz = tf.reduce_sum(dfz)

    Imin = tf.reduce_sum(tf.math.square(tf.math.minimum(zernike_magnitude, 0)))
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(background, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    fsz = zernike_magnitude.shape
    Inorm = tf.reduce_mean(
        tf.math.square(
            tf.math.reduce_sum(zernike_magnitude, axis=(-1, -2)) -
            tf.math.reduce_sum(zernike_magnitude) / fsz[0]
        )
    )

    loss = (mse_norm1 * w[0] + mse_norm2 * w[1] + w[2] * dfz + s * w[3] +
            w[4] * Imin * mu + bgmin * w[5] * mu + intensitymin * w[6] * mu +
            Inorm * w[7] + gxymean * w[8])

    return loss

    
def mse_real_4pi(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss for 4Pi Zernike PSF.

    *variables* may be a :class:`Zernike4PiLossVariables` or a plain list
    of tensors (11 elements, see :class:`Zernike4PiLossVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = Zernike4PiLossVariables.from_list(variables)

    mydiff = model - data
    mydiff = mydiff[:, :, 1:-1]
    data = data[:, :, 1:-1]
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    zernike_magnitude_1 = variables.zernike_magnitude_1
    background = variables.backgrounds
    intensity = variables.intensities
    drift_xy = variables.drift_xy

    gxymean = tf.reduce_mean(tf.abs(drift_xy))

    s = tf.math.reduce_sum(
        tf.math.square(zernike_magnitude_1[0] - zernike_magnitude_1[1]) +
        tf.math.square(zernike_magnitude_1[-1] - zernike_magnitude_1[-2])
    )
    fsz = zernike_magnitude_1.shape

    zernike_magnitude_2 = variables.zernike_magnitude_2
    zernike_phase_2 = variables.zernike_phase_2

    Areal = zernike_magnitude_2
    Aimg = zernike_phase_2
    A = tf.complex(Areal, Aimg)

    dfz = (tf.math.square(tf.experimental.numpy.diff(zernike_magnitude_1, n=1, axis=-3)) +
           tf.math.square(tf.experimental.numpy.diff(Areal, n=1, axis=-3)) +
           tf.math.square(tf.experimental.numpy.diff(Aimg, n=1, axis=-3)))
    dfz = tf.reduce_sum(dfz)

    s1 = tf.math.reduce_sum(
        tf.math.square(Areal[0] - Areal[1]) + tf.math.square(Areal[-1] - Areal[-2])
    )
    s2 = tf.math.reduce_sum(
        tf.math.square(Aimg[0] - Aimg[1]) + tf.math.square(Aimg[-1] - Aimg[-2])
    )
    Imin = tf.reduce_sum(tf.math.square(tf.math.minimum(zernike_magnitude_1 - 2 * tf.math.abs(A), 0)))
    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(background, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    Inorm = tf.reduce_mean(
        tf.math.square(
            tf.math.reduce_sum(zernike_magnitude_1, axis=(-1, -2)) -
            tf.math.reduce_sum(zernike_magnitude_1) / fsz[0]
        )
    )

    loss = (mse_norm1 * w[0] + mse_norm2 * w[1] + w[2] * dfz +
            (s + s1 + s2) * w[3] + w[4] * Imin * mu + bgmin * mu * w[5] +
            intensitymin * w[6] * mu + Inorm * w[7] * mu + gxymean * w[8])

    return loss


def mse_real_4pi_All(model, data, loss_func, variables=None, mu=None, w=None, psfnorm=None):
    """
    Wrapper for 4Pi PSF loss across multiple beads.
    Iterates over each bead and accumulates loss.
    """
    assert variables is not None and w is not None and mu is not None
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


def mse_real_All(model, data, loss_func, variables=None, mu=None, w=None, psfnorm=None):
    """
    Wrapper for PSF loss across multiple beads.
    Iterates over each bead and accumulates loss.
    """
    assert variables is not None and w is not None and mu is not None
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


def mse_real_pupil(model, data, variables=None, mu=None, w=None, psfnorm=1.0):
    """
    Mean squared error loss for direct pupil function PSF.

    *variables* may be a :class:`PupilLossVariables` or a plain list
    of tensors (7 elements, see :class:`PupilLossVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = PupilLossVariables.from_list(variables)

    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    pupil_real = variables.pupil_real
    pupil_imag = variables.pupil_imag
    background = variables.backgrounds
    intensity = variables.intensities
    drift_xy = variables.drift_xy

    gxymean = tf.reduce_mean(tf.abs(drift_xy))
    Inorm = tf.math.square(tf.math.minimum(psfnorm - 0.97, 0))

    dfxy1 = (tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_imag, n=1, axis=-1))) +
             tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_imag, n=1, axis=-2))))
    dfxy2 = (tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_real, n=1, axis=-1))) +
             tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_real, n=1, axis=-2))))
    dfxy = dfxy2

    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(background, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))

    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfxy * w[2] + gxymean * w[8] + Inorm * w[7]

    return loss


def mse_pupil_4pi(model, data, variables=None, mu=None, w=None, psfnorm=[1.0, 1.0]):
    """
    Mean squared error loss for 4Pi direct pupil PSF.

    *variables* may be a :class:`Pupil4PiLossVariables` or a plain list
    of tensors (10 elements, see :class:`Pupil4PiLossVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = Pupil4PiLossVariables.from_list(variables)

    mydiff = model - data
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    pupil_real_1 = variables.pupil_real_1
    pupil_imag_1 = variables.pupil_imag_1
    pupil_real_2 = variables.pupil_real_2
    pupil_imag_2 = variables.pupil_imag_2
    background = variables.backgrounds
    intensity = variables.intensities
    alpha = variables.alpha
    wavelength = variables.wavelength
    drift_xy = variables.drift_xy

    gxymean = tf.reduce_mean(tf.abs(drift_xy))
    Inorm = (tf.math.square(tf.math.minimum(psfnorm[0] - 0.97, 0)) +
             tf.math.square(tf.math.minimum(psfnorm[1] - 0.97, 0)))

    dfxy1 = (tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_imag_1, n=1, axis=-1))) +
             tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_imag_1, n=1, axis=-2))))
    dfxy2 = (tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_real_1, n=1, axis=-1))) +
             tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_real_1, n=1, axis=-2))))
    dfxy3 = (tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_imag_2, n=1, axis=-1))) +
             tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_imag_2, n=1, axis=-2))))
    dfxy4 = (tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_real_2, n=1, axis=-1))) +
             tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_real_2, n=1, axis=-2))))
    dfxy = dfxy2 + dfxy4

    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(background, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    alphamin = tf.reduce_sum(tf.math.square(tf.math.minimum(alpha, 0)))

    loss = (LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu +
            dfxy * w[2] + alphamin * w[4] * mu + gxymean * w[8] + Inorm * w[7])

    return loss


def mse_real_zernike(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss for Zernike PSF.

    Accepts either a :class:`ZernikeLossVariables` (or
    ``ZernikePSFVariables``) object or a plain list of tensors.
    The plain list form is used by the L-BFGS-B optimizer batching loop,
    which passes raw tensors for gradient tracking.
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = ZernikeLossVariables.from_list(variables)

    if hasattr(variables, 'value'):
        background = variables.backgrounds.value
        intensity = variables.intensities.value
        zernike_magnitude = variables.zernike_magnitude.value
        sigma = variables.sigma.value
        drift_xy = variables.drift_xy.value
    else:
        background = variables.backgrounds
        intensity = variables.intensities
        zernike_magnitude = variables.zernike_magnitude
        sigma = variables.sigma
        drift_xy = variables.drift_xy

    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    gxymean = tf.reduce_mean(tf.abs(drift_xy))

    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(background, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    g1 = tf.reduce_sum(tf.square(zernike_magnitude[0][1:]))
    g2 = tf.reduce_sum(tf.square(zernike_magnitude[1]))

    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + gxymean * w[8]

    return loss


def mse_zernike_4pi(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss for 4Pi Zernike PSF.

    *variables* may be a :class:`Zernike4PiLossVariables` or a plain list
    of tensors (11 elements, see :class:`Zernike4PiLossVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = Zernike4PiLossVariables.from_list(variables)

    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables.backgrounds
    intensity = variables.intensities
    zernike_magnitude_1 = variables.zernike_magnitude_1
    zernike_magnitude_2 = variables.zernike_magnitude_2
    alpha = variables.alpha
    posd = variables.wavelength
    drift_xy = variables.drift_xy

    gxymean = tf.reduce_mean(tf.abs(drift_xy))

    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(background, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    alphamin = tf.reduce_sum(tf.math.square(tf.math.minimum(alpha, 0)))

    g1 = tf.reduce_sum(tf.abs(zernike_magnitude_1[1][1:]))
    g2 = tf.reduce_sum(tf.abs(zernike_magnitude_1[0][1:])) * 2 + tf.reduce_sum(tf.abs(zernike_magnitude_2[0][1:])) * 2
    g3 = tf.reduce_sum(tf.abs(zernike_magnitude_2[1][1:]))
    g4 = tf.reduce_sum(tf.square(posd)) * 2

    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + alphamin * w[4] * mu + gxymean * w[8]

    return loss


def mse_zernike_4pi_smlm(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss for 4Pi SMLM Zernike PSF.

    *variables* may be a :class:`Zernike4PiSMLMLossVariables` or a plain
    list of tensors (12 elements, see :class:`Zernike4PiSMLMLossVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = Zernike4PiSMLMLossVariables.from_list(variables)

    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-2, -1))
    ) * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables.backgrounds
    intensity = variables.intensities
    stage_position = variables.stage_position
    sample_height = variables.sample_height
    zernike_magnitude_1 = variables.zernike_magnitude_1
    zernike_magnitude_2 = variables.zernike_magnitude_2
    alpha = variables.alpha
    zpos = variables.positions[:, 0, ...]

    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(background, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    alphamin = tf.reduce_sum(tf.math.square(tf.math.minimum(alpha, 0)))
    zmin = (tf.reduce_mean(tf.math.square(tf.math.minimum(zpos, 0))) +
            tf.reduce_mean(tf.math.square(tf.math.minimum(stage_position, 0))) +
            tf.reduce_mean(tf.math.square(tf.math.minimum(sample_height, 0))))

    g1 = tf.reduce_sum(tf.abs(zernike_magnitude_1[1][1:]))
    g2 = tf.reduce_sum(tf.abs(zernike_magnitude_1[0][1:])) * 2 + tf.reduce_sum(tf.abs(zernike_magnitude_2[0][1:])) * 2
    g3 = tf.reduce_sum(tf.abs(zernike_magnitude_2[1][1:]))

    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + alphamin * w[4] * mu + zmin * w[4] * mu

    return loss


def mse_real_zernike_FD(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss for field-dependent Zernike PSF.

    *variables* may be a :class:`ZernikeFDLossVariables` or a plain list
    of tensors (6 elements, see :class:`ZernikeFDLossVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = ZernikeFDLossVariables.from_list(variables)

    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables.backgrounds
    intensity = variables.intensities
    drift_xy = variables.drift_xy
    zernike_map = variables.zernike_map

    gxymean = tf.reduce_mean(tf.abs(drift_xy))

    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(background, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))

    dfxy = (tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(zernike_map, n=1, axis=-1))) +
            tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(zernike_map, n=1, axis=-2))))

    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfxy * w[2] + gxymean * w[8]

    return loss


def mse_real_zernike_IMM(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss for IMM (Interference Microscopy Model) Zernike PSF.

    *variables* may be a :class:`ZernikeFDLossVariables` or a plain list
    of tensors (6 elements, see :class:`ZernikeFDLossVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = ZernikeFDLossVariables.from_list(variables)

    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables.backgrounds
    intensity = variables.intensities
    positions = variables.positions
    drift_xy = variables.drift_xy
    zernike_map = variables.zernike_map

    gxymean = tf.reduce_mean(tf.abs(drift_xy))

    bgmin = tf.reduce_sum(tf.math.square(tf.math.minimum(background, 0)))
    intensitymin = tf.reduce_sum(tf.math.square(tf.math.minimum(intensity, 0)))
    state_pos = positions[:, 1, ...]
    zmin = tf.reduce_mean(tf.math.square(tf.math.minimum(state_pos, 0)))

    dfz = tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(zernike_map, n=1, axis=-1)))

    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfz * w[2] + gxymean * w[8] + zmin * w[4] * mu

    return loss


def mse_real_zernike_FD_smlm(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss for field-dependent SMLM Zernike PSF.

    *variables* may be a :class:`ZernikeFDSMLMLossVariables` or a plain
    list of tensors (7 elements, see :class:`ZernikeFDSMLMLossVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = ZernikeFDSMLMLossVariables.from_list(variables)

    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_mean(tf.square(mydiff), axis=(-2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-2, -1))
    ) * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables.backgrounds
    intensity = variables.intensities
    stage_position = variables.stage_position
    positions = variables.positions
    zernike_map = variables.zernike_map

    zpos = positions[:, 0, ...]
    bgmin = tf.reduce_mean(tf.math.square(tf.math.minimum(background, 0)))
    zmin = (tf.reduce_mean(tf.math.square(tf.math.minimum(zpos, 0))) +
            tf.reduce_mean(tf.math.square(tf.math.minimum(stage_position, 0))))
    intensitymin = tf.reduce_mean(tf.math.square(tf.math.minimum(intensity, 0)))

    dfxy = (tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(zernike_map, n=1, axis=-1))) +
            tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(zernike_map, n=1, axis=-2))))

    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfxy * w[2] + zmin * w[4] * mu

    return loss


def mse_real_zernike_smlm(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss for SMLM Zernike PSF.

    *variables* may be a :class:`ZernikeSMLMLossVariables` or a plain
    list of tensors (8 elements, see :class:`ZernikeSMLMLossVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = ZernikeSMLMLossVariables.from_list(variables)

    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_mean(tf.square(mydiff), axis=(-2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-2, -1))
    ) * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables.backgrounds
    intensity = variables.intensities
    zernike_magnitude = variables.zernike_magnitude
    stage_position = variables.stage_position
    positions = variables.positions

    zpos = positions[:, 0, ...]
    bgmin = tf.reduce_mean(tf.math.square(tf.math.minimum(background, 0)))
    zmin = (tf.reduce_mean(tf.math.square(tf.math.minimum(zpos, 0))) +
            tf.reduce_mean(tf.math.square(tf.math.minimum(stage_position, 0))))
    intensitymin = tf.reduce_mean(tf.math.square(tf.math.minimum(intensity, 0)))

    g1 = tf.reduce_sum(tf.square(zernike_magnitude[0][1:]))
    g2 = tf.reduce_sum(tf.square(zernike_magnitude[1]))

    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + (g1 + g2) * w[2] + zmin * w[4] * mu

    return loss


def mse_real_pupil_smlm(model, data, variables=None, mu=None, w=None, psfnorm=1.0):
    """
    Mean squared error loss for SMLM direct pupil PSF.

    *variables* may be a :class:`PupilSMLMLossVariables` or a plain list
    of tensors (8 elements, see :class:`PupilSMLMLossVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = PupilSMLMLossVariables.from_list(variables)

    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_mean(tf.square(mydiff), axis=(-2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-2, -1))
    ) * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables.backgrounds
    intensity = variables.intensities
    pupil_real = variables.pupil_real
    pupil_imag = variables.pupil_imag
    stage_position = variables.stage_position
    positions = variables.positions

    zpos = positions[:, 0, ...]
    bgmin = tf.reduce_mean(tf.math.square(tf.math.minimum(background, 0)))
    zmin = (tf.reduce_mean(tf.math.square(tf.math.minimum(zpos, 0))) +
            tf.reduce_mean(tf.math.square(tf.math.minimum(stage_position, 0))))
    intensitymin = tf.reduce_mean(tf.math.square(tf.math.minimum(intensity, 0)))
    Inorm = tf.math.square(tf.math.minimum(psfnorm - 0.97, 0))

    dfxy1 = (tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_imag, n=1, axis=-1))) +
             tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_imag, n=1, axis=-2))))
    dfxy2 = (tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_real, n=1, axis=-1))) +
             tf.reduce_sum(tf.math.square(tf.experimental.numpy.diff(pupil_real, n=1, axis=-2))))
    dfxy = dfxy2

    loss = LL * w[0] + bgmin * w[5] * mu + intensitymin * w[6] * mu + dfxy * w[2] + zmin * w[4] * mu + Inorm * w[7]

    return loss