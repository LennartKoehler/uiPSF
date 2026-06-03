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


def mse_real(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss with regularization for Zernike PSF.
    
    Variables structure (7 elements):
        [0] positions: Emitter positions [n_beads, 2-4]
        [1] backgrounds: Background level [n_beads, 1, 1, 1]
        [2] intensities: Emitter intensity [n_beads, ...]
        [3] zernike_magnitude: Zernike magnitude coeffs [n_zernike, 1, 1]
        [4] zernike_phase: Zernike phase coeffs [n_zernike, 1, 1]
        [5] sigma: Gaussian blur [2]
        [6] drift_xy: Lateral drift [n_beads, 2]
    """
    assert variables is not None and w is not None and mu is not None
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

    zernike_magnitude = variables[3]
    zernike_phase = variables[4]
    background = variables[1]
    intensity = variables[2]

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
    
    Variables structure (11 elements):
        [0] positions: Emitter positions [n_beads, 2-4]
        [1] backgrounds: Background level [n_beads, 1, 1, 1]
        [2] intensities: Emitter intensity [n_beads, ...]
        [3] intensity_phase: Interference phase [n_beads, 1, 1, 1]
        [4] zernike_magnitude_1: Zernike magnitude arm 1 [n_zernike, 1, 1]
        [5] zernike_phase_1: Zernike phase arm 1 [n_zernike, 1, 1]
        [6] zernike_magnitude_2: Zernike magnitude arm 2 [n_zernike, 1, 1]
        [7] zernike_phase_2: Zernike phase arm 2 [n_zernike, 1, 1]
        [8] alpha: Interference visibility [1]
        [9] wavelength: Wavelength [1]
        [10] drift_xy: Lateral drift [n_beads, 2]
    """
    assert variables is not None and w is not None and mu is not None
    mydiff = model - data
    mydiff = mydiff[:, :, 1:-1]
    data = data[:, :, 1:-1]
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    zernike_magnitude_1 = variables[4]
    background = variables[1]
    intensity = variables[2]
    drift_xy = variables[-1]

    gxymean = tf.reduce_mean(tf.abs(drift_xy))

    s = tf.math.reduce_sum(
        tf.math.square(zernike_magnitude_1[0] - zernike_magnitude_1[1]) +
        tf.math.square(zernike_magnitude_1[-1] - zernike_magnitude_1[-2])
    )
    fsz = zernike_magnitude_1.shape

    zernike_magnitude_2 = variables[6]
    zernike_phase_2 = variables[7]

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
    
    Variables structure (7 elements):
        [0] positions: Emitter positions [n_beads, 2-4]
        [1] backgrounds: Background level [n_beads, 1, 1, 1]
        [2] intensities: Emitter intensity [n_beads, ...]
        [3] pupil_real: Real part of pupil [n_pupil, 1, 1]
        [4] pupil_imag: Imaginary part of pupil [n_pupil, 1, 1]
        [5] sigma: Gaussian blur [2]
        [6] drift_xy: Lateral drift [n_beads, 2]
    """
    assert variables is not None and w is not None and mu is not None
    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    pupil_real = variables[3]
    pupil_imag = variables[4]
    background = variables[1]
    intensity = variables[2]
    drift_xy = variables[-1]

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
    
    Variables structure (10 elements):
        [0] positions: Emitter positions [n_beads, 2-4]
        [1] backgrounds: Background level [n_beads, 1, 1, 1]
        [2] intensities: Emitter intensity [n_beads, ...]
        [3] pupil_real_1: Real pupil arm 1 [n_pupil, 1, 1]
        [4] pupil_imag_1: Imaginary pupil arm 1 [n_pupil, 1, 1]
        [5] pupil_real_2: Real pupil arm 2 [n_pupil, 1, 1]
        [6] pupil_imag_2: Imaginary pupil arm 2 [n_pupil, 1, 1]
        [7] alpha: Interference visibility [1]
        [8] wavelength: Wavelength [1]
        [9] drift_xy: Lateral drift [n_beads, 2]
    """
    assert variables is not None and w is not None and mu is not None
    mydiff = model - data
    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    pupil_real_1 = variables[3]
    pupil_imag_1 = variables[4]
    pupil_real_2 = variables[5]
    pupil_imag_2 = variables[6]
    background = variables[1]
    intensity = variables[2]
    alpha = variables[7]
    wavelength = variables[8]
    drift_xy = variables[-1]

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
    
    Accepts either a ZernikePSFVariables object or a plain list of tensors.
    The plain list form is used by the L-BFGS-B optimizer batching loop,
    which passes raw tensors for gradient tracking.
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        # Raw tensor list: use directly (do NOT wrap in LearnablePSFParameters,
        # as that would create new tf.Variables that break the gradient chain)
        background = variables[1]
        intensity = variables[2]
        zernike_magnitude = variables[3]
        sigma = variables[5]
        drift_xy = variables[-1]
    else:
        background = variables.backgrounds.value
        intensity = variables.intensities.value
        zernike_magnitude = variables.zernike_magnitude.value
        sigma = variables.sigma.value
        drift_xy = variables.drift_xy.value

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
    
    Variables structure (11 elements):
        [0] positions: Emitter positions [n_beads, 2-4]
        [1] backgrounds: Background level [n_beads, 1, 1, 1]
        [2] intensities: Emitter intensity [n_beads, ...]
        [3] intensity_phase: Interference phase [n_beads, 1, 1, 1]
        [4] zernike_magnitude_1: Zernike magnitude arm 1 [n_zernike, 1, 1]
        [5] zernike_phase_1: Zernike phase arm 1 [n_zernike, 1, 1]
        [6] zernike_magnitude_2: Zernike magnitude arm 2 [n_zernike, 1, 1]
        [7] zernike_phase_2: Zernike phase arm 2 [n_zernike, 1, 1]
        [8] alpha: Interference visibility [1]
        [9] wavelength: Wavelength [1]
        [10] drift_xy: Lateral drift [n_beads, 2]
    """
    assert variables is not None and w is not None and mu is not None
    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables[1]
    intensity = variables[2]
    intensity_phase = variables[3]
    zernike_magnitude_1 = variables[4]
    zernike_phase_1 = variables[5]
    zernike_magnitude_2 = variables[6]
    zernike_phase_2 = variables[7]
    alpha = variables[8]
    wavelength = variables[9]
    posd = variables[10]
    drift_xy = variables[-1]

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
    
    Variables structure (12 elements):
        [0] positions: Emitter positions [n_beads, 2-4]
        [1] backgrounds: Background level [n_beads, 1, 1, 1]
        [2] intensities: Emitter intensity [n_beads, ...]
        [3] intensity_phase: Interference phase [n_beads, 1, 1, 1]
        [4] stage_position: Stage position [n_beads, 1, 1, 1]
        [5] sample_height: Sample height [1]
        [6] zernike_magnitude_1: Zernike magnitude arm 1 [n_zernike, 1, 1]
        [7] zernike_phase_1: Zernike phase arm 1 [n_zernike, 1, 1]
        [8] zernike_magnitude_2: Zernike magnitude arm 2 [n_zernike, 1, 1]
        [9] zernike_phase_2: Zernike phase arm 2 [n_zernike, 1, 1]
        [10] alpha: Interference visibility [1]
        [11] drift_xy: Lateral drift [n_beads, 2]
    """
    assert variables is not None and w is not None and mu is not None
    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-2, -1))
    ) * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables[1]
    intensity = variables[2]
    intensity_phase = variables[3]
    stage_position = variables[4]
    sample_height = variables[5]
    zernike_magnitude_1 = variables[6]
    zernike_phase_1 = variables[7]
    zernike_magnitude_2 = variables[8]
    zernike_phase_2 = variables[9]
    alpha = variables[10]
    zpos = variables[0][:, 0, ...]

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
    
    Variables structure (6 elements):
        [0] positions: Emitter positions [n_beads, 2-4]
        [1] backgrounds: Background level [n_beads, 1, 1, 1]
        [2] intensities: Emitter intensity [n_beads, ...]
        [3] zernike_map: Per-bead Zernike coeffs [n_beads, n_zernike]
        [4] sigma: Gaussian blur [2]
        [5] drift_xy: Lateral drift [n_beads, 2]
    """
    assert variables is not None and w is not None and mu is not None
    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables[1]
    intensity = variables[2]
    drift_xy = variables[-1]
    zernike_map = variables[3]

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
    
    Variables structure (6 elements):
        [0] positions: Extended positions [n_beads, 4] = [z, state, y, x]
        [1] backgrounds: Background level [n_beads, 1, 1, 1]
        [2] intensities: Emitter intensity [n_beads, ...]
        [3] zernike_map: Per-bead Zernike coeffs [n_beads, n_zernike]
        [4] sigma: Gaussian blur [2]
        [5] drift_xy: Lateral drift [n_beads, 2]
    """
    assert variables is not None and w is not None and mu is not None
    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_sum(tf.square(mydiff), axis=(-3, -2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-3, -2, -1))
    ) / data.shape[-3] * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables[1]
    intensity = variables[2]
    positions = variables[0]
    drift_xy = variables[-1]
    zernike_map = variables[3]

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
    
    Variables structure (7 elements):
        [0] positions: Emitter positions [n_beads, 2-4]
        [1] backgrounds: Background level [n_beads, 1, 1, 1]
        [2] intensities: Emitter intensity [n_beads, ...]
        [3] zernike_map: Per-bead Zernike coeffs [n_beads, n_zernike]
        [4] sigma: Gaussian blur [2]
        [5] stage_position: Stage position [n_beads, 1, 1, 1]
        [6] drift_xy: Lateral drift [n_beads, 2]
    """
    assert variables is not None and w is not None and mu is not None
    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_mean(tf.square(mydiff), axis=(-2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-2, -1))
    ) * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables[1]
    intensity = variables[2]
    stage_position = variables[5]
    positions = variables[0]
    zernike_map = variables[3]

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
    
    Variables structure (8 elements):
        [0] positions: Emitter positions [n_beads, 2-4]
        [1] backgrounds: Background level [n_beads, 1, 1, 1]
        [2] intensities: Emitter intensity [n_beads, ...]
        [3] zernike_magnitude: Zernike magnitude coeffs [n_zernike, 1, 1]
        [4] zernike_phase: Zernike phase coeffs [n_zernike, 1, 1]
        [5] stage_position: Stage position [n_beads, 1, 1, 1]
        [6] sigma: Gaussian blur [2]
        [7] drift_xy: Lateral drift [n_beads, 2]
    """
    assert variables is not None and w is not None and mu is not None
    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_mean(tf.square(mydiff), axis=(-2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-2, -1))
    ) * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables[1]
    intensity = variables[2]
    zernike_magnitude = variables[3]
    stage_position = variables[5]
    positions = variables[0]

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
    
    Variables structure (8 elements):
        [0] positions: Emitter positions [n_beads, 2-4]
        [1] backgrounds: Background level [n_beads, 1, 1, 1]
        [2] intensities: Emitter intensity [n_beads, ...]
        [3] pupil_real: Real part of pupil [n_pupil, 1, 1]
        [4] pupil_imag: Imaginary part of pupil [n_pupil, 1, 1]
        [5] stage_position: Stage position [n_beads, 1, 1, 1]
        [6] sigma: Gaussian blur [2]
        [7] drift_xy: Lateral drift [n_beads, 2]
    """
    assert variables is not None and w is not None and mu is not None
    mydiff = model - data

    mse_norm1 = tf.reduce_mean(tf.square(mydiff)) / tf.reduce_mean(data)
    mse_norm2 = tf.reduce_mean(
        tf.reduce_mean(tf.square(mydiff), axis=(-2, -1)) /
        tf.math.reduce_max(tf.square(data), axis=(-2, -1))
    ) * 200

    LL = (model - data - data * tf.math.log(model) + data * tf.math.log(data))
    LL = tf.reduce_mean(LL[tf.math.is_finite(LL)])

    background = variables[1]
    intensity = variables[2]
    pupil_real = variables[3]
    pupil_imag = variables[4]
    stage_position = variables[6]
    positions = variables[0]

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