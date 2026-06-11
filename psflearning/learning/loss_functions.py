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

from psflearning.learning.psfs.PSFZernikeBased import ZernikePSFVariables


def mse_real(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss with regularization for Zernike PSF.

    *variables* may be a :class:`ZernikePSFVariables` or a plain list
    of tensors (7 elements, see :class:`ZernikePSFVariables`).
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = ZernikePSFVariables.from_list(variables)

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

    zernike_magnitude = variables.zernike_magnitude.value
    zernike_phase = variables.zernike_phase.value
    background = variables.backgrounds.value
    intensity = variables.intensities.value

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


def mse_real_zernike(model, data, variables=None, mu=None, w=None):
    """
    Mean squared error loss for Zernike PSF.

    Accepts either a :class:`ZernikePSFVariables` (or
    ``ZernikePSFVariables``) object or a plain list of tensors.
    The plain list form is used by the L-BFGS-B optimizer batching loop,
    which passes raw tensors for gradient tracking.
    """
    assert variables is not None and w is not None and mu is not None
    if isinstance(variables, list):
        variables = ZernikePSFVariables.from_list(variables)

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
