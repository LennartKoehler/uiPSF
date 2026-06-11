"""
Copyright (c) 2022      Ries Lab, EMBL, Heidelberg, Germany
All rights reserved

@author: Jonas Hellgoth, Sheng Liu
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from .psfs.IPSFModel import LearnablePSFParameters, LearnableParameter, ParameterScope
import time

from typing import Any, List
import numpy as np
import scipy as sp
import scipy.optimize as optimize
import tensorflow as tf
import sys
import tkinter as tk
from tkinter import messagebox as mbox


@dataclass
class _VariableMetadata:
    """Metadata for flattened variables, used internally by L-BFGS-B.

    Holds the shape/dtype information needed to reconstruct structured
    variables from the flat 1-D vector that scipy requires, plus the
    per-parameter scope/id needed for batched gradient computation.
    """
    shapes: list
    lengths: list
    dtypes: list
    param_scopes: list
    param_ids: list
    variables_template: LearnablePSFParameters

class OptimizerABC:
    """
    Abstract base class for optimizers. It ensures consistency and compatability between Fitters and Optimizers.
    Core function is 'minimize' which is called by the fitter. The rest is handled by the optimizer.
    Allows to use different TensorFLow optimizers and also the L-BFGS-B optimizer from scipy.
    Is basically a wraper around those optimizers to call them similarly in the Fitter.
    Defines an interface for other optimizers (basically the minimize function) that self-made optimizers must fulfill.
    """

    __metaclass__ = ABCMeta

    objective: Any

    def __init__(self, maxiter, options, kwargs) -> None:
        self.maxiter = maxiter

        self.print_step_size = np.max((np.round(self.maxiter / 10).astype(np.int32),20))
        self.print_width = len(str(self.print_step_size))

        self.history = [['step', 'time', 'loss']]

        self.weight = None
        self.opt = self.create_actual_optimizer(options, kwargs)

    @abstractmethod
    def create_actual_optimizer(self, options, kwargs):
        """
        Here the actual underlying optimizer should be created.
        """
        raise NotImplementedError("You need to implement a 'create_actual_optimizer' method in your optimizer class.")

    def minimize(self, objective, variables : LearnablePSFParameters, pbar):
        """
        Adapts the given variables in a way that minimizes the given objective.
        Returns the same variables object (mutated in-place by the optimizer).
        """
        variablesTensor = variables.toTensorList()  # List[tf.Variable]

        for step in range(self.maxiter):
            start = time.time()
            with tf.GradientTape() as tape:
                # No tape.watch() needed — tf.Variables are automatically watched
                loss = objective(variables)
            pbar.update(1)
            pbar.set_description("current loss %f" % loss)
            gradients = tape.gradient(loss, variablesTensor)
            self.opt.apply_gradients(zip(gradients, variablesTensor))

            self.update_history(step+1, time.time()-start, loss.numpy())

        return variables  # Mutated in-place, no need to reconstruct

    def objective_wrapper_for_optimizer(self, variables):
        """
        Wrapper around the actual objective. Needed since TensorFLow optimizer
        can only optimize a function that takes no arguments and returns a loss.
        """
        return self.objective(variables)

    def write_output(self, step, loss, do_anyway=False):
        """
        Writes output to the console in a nicely formatted way.
        Used to show user the progress of the optimization.
        """
        # TODO: one could caluclate an estimate how long the optimization still takes
        self.print_step_size =  np.max((np.round(self.maxiter / 10).astype(np.int32),20))
        if (step % self.print_step_size == 0) or do_anyway:
            #tf.print(f"[{step:5}/{self.maxiter}]  loss={loss:>8.2f} ")
            tf.print("step:",step,"loss:",loss)
        return

    def update_history(self, step, time, loss):
        """
        Save information of each iteration to the history.
        This history can be later used to analyze the efficiency of the optimization.
        """
        self.history.append([step, time, loss])
        return

class Adadelta(OptimizerABC):
    """
    Wrapper around TensorFlows Adadelta optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, rho=0.95, epsilon=1e-07,
                name='Adadelta', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Adadelta(**options, **kwargs)


class Adagrad(OptimizerABC):
    """
    Wrapper around TensorFlows Adagrad optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07,
                name='Adagrad', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Adagrad(**options, **kwargs)


class Adam(OptimizerABC):
    """
    Wrapper around TensorFlows Adam optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                name='Adam', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Adam(**options, **kwargs)


class Adamax(OptimizerABC):
    """
    Wrapper around TensorFlows Adamax optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                name='Adamax', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Adamax(**options, **kwargs)


class Ftrl(OptimizerABC):
    """
    Wrapper around TensorFlows Ftrl optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1,
                l1_regularization_strength=0.0, l2_regularization_strength=0.0,
                name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0, **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Ftrl(**options, **kwargs)


class Nadam(OptimizerABC):
    """
    Wrapper around TensorFlows Nadam optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                name='Nadam', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.Nadam(**options, **kwargs)


class RMSprop(OptimizerABC):
    """
    Wrapper around TensorFlows RMSprop optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
                name='RMSprop', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.RMSprop(**options, **kwargs)


class SGD(OptimizerABC):
    """
    Wrapper around TensorFlows SGD optimizer.
    """
    def __init__(self, maxiter, learning_rate=0.01, momentum=0.0, nesterov=False,
                name='SGD', **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['maxiter']
        del options['kwargs']
        del options['__class__']
        super().__init__(maxiter, options, kwargs)

    def create_actual_optimizer(self, options, kwargs):
        return tf.optimizers.SGD(**options, **kwargs)


class L_BFGS_B(OptimizerABC):
    """
    Wrapper around scipys L-BFGS-B optimizer.
    There is not L-BFGS-B optimizer available in TensorFlow.
    """
    # this is the type of optimizer Rainer used
    # alternatve would be optimizer.lbfgs_minimize from tensorflow-probability
    # this works similar but seemed to be slower

    # the problem with the scipy (and also the tensorflow-probability) implementation is that
    # they can only handle 1D tensors/arrays, therefore we must flatten our variables,
    # save shapes and lengths and reshape them again in objective
    # this is similar to https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
    # right now this works fine
    def __init__(self, maxiter, batch_size=30, rate=1.1, gtol=1e-10, **kwargs) -> None:
        options = locals().copy()
        del options['self']
        del options['kwargs']
        del options['__class__']
        del options['batch_size']
        del options['rate']
        super().__init__(maxiter, options, kwargs)

        self.batch_size = batch_size
        self.rate = rate
        self.step = 0
        self.status = None  # to access final output from outside

    def create_actual_optimizer(self, options, kwargs):
        """
        Implemented to allow to inherit from ABC. Is just a placeholder in this case since
        the scipy API does only provide a function not a real optimizer object.
        Used to adapt the options to fit the scipy API.
        """
        self.options = {**options, **kwargs}
        return None

    def minimize(self, objective, variables: LearnablePSFParameters, pbar):
        """
        Adapts the given variables in a way that minimizes the given objective.
        Returns the variables object with updated values.
        ABC overwritten since optimization works a bit different for the scipy optimizer.
        """
        self.step = 0
        mu = 1.0

        init_var, meta = self._flatten_variables(variables)
        self.options['maxiter'] = self.maxiter
        start_time = pbar.postfix[-1]['time']

        scipy_fun = self._make_scipy_objective(objective, meta, pbar, start_time, mu)
        result = optimize.minimize(
            fun=scipy_fun, x0=init_var, jac=True,
            method='L-BFGS-B', options=self.options,
        )

        self.status = result
        self._reshape_variables_np(result.x, meta)
        return variables

    # ------------------------------------------------------------------
    # Variable flattening / reshaping (pure functions on metadata)
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_variables(variables: LearnablePSFParameters):
        """
        Flattens and concatenates the parameters of a LearnablePSFParameters object
        to one large vector. Needed since L-BFGS-B can only handle vectors.

        Returns
        -------
        flat : np.ndarray
            1-D concatenated parameter vector.
        meta : _VariableMetadata
            Metadata (shapes, lengths, dtypes, scopes, ids, template)
            needed to reconstruct the structured variables later.
        """
        params = variables.toLearnableParameterList()
        shapes, lengths, dtypes, param_scopes, param_ids = [], [], [], [], []
        flat_variables = []

        # dtypes and casting are actually not needed anymore (we switched
        # from complex to float32) but left active since they do no harm
        # and may be useful in the future
        for param in params:
            value = param.numpy()
            shape = value.shape
            shapes.append(shape)
            lengths.append(int(np.prod(shape)))
            dtypes.append(value.dtype)
            param_scopes.append(param.scope)
            param_ids.append(param.id)
            flat_variables.append(value.flatten())

        meta = _VariableMetadata(
            shapes=shapes, lengths=lengths, dtypes=dtypes,
            param_scopes=param_scopes, param_ids=param_ids,
            variables_template=variables,
        )
        return np.concatenate(flat_variables), meta

    @staticmethod
    def _reshape_variables_np(var, meta: _VariableMetadata):
        """
        Reconstructs/reshapes the current state of the variables from the current guess
        of the optimizer (var vector). Updates the values in-place on the
        LearnablePSFParameters template using .value setter (which calls
        tf.Variable.assign).
        Called for final reconstruction and therefore implemented with numpy.
        """
        idx_count = 0
        params = meta.variables_template.toLearnableParameterList()

        for i, (shape, length, dtype) in enumerate(zip(meta.shapes, meta.lengths, meta.dtypes)):
            variable = var[idx_count : idx_count + length]
            params[i].value = np.reshape(variable, shape).astype(dtype)
            idx_count += length

        return meta.variables_template

    @staticmethod
    def _reshape_variables_tf(var, meta: _VariableMetadata):
        """
        Reconstructs/reshapes the current state of the variables from the current guess
        of the optimizer (var vector). Returns a new LearnablePSFParameters constructed
        from the reshaped tf tensors.
        Called in each iteration of the optimization and therefore implemented with tensorflow.
        """
        tensors = [None] * len(meta.shapes)
        idx_count = 0

        for i, (shape, length, dtype) in enumerate(zip(meta.shapes, meta.lengths, meta.dtypes)):
            variable = var[idx_count : idx_count + length]
            tensors[i] = tf.cast(tf.reshape(variable, shape), dtype)
            idx_count += length

        return meta.variables_template.fromTensorList(tensors)

    # ------------------------------------------------------------------
    # Objective / gradient wrappers (closure-based, no mutable state)
    # ------------------------------------------------------------------

    def _make_scipy_objective(self, objective, meta: _VariableMetadata, pbar, start_time, mu_init):
        """
        Creates and returns the function that scipy.optimize.minimize will call.

        The returned closure captures all the data it needs (objective, metadata,
        progress bar, penalty parameter state) instead of storing it on ``self``.
        This keeps data-dependent runtime state out of the optimizer instance.

        Returns a callable ``fun(var)`` that returns ``(loss_f64, grad_f64)``
        in float64 as required by L-BFGS-B's Fortran backend.
        """
        # Mutable penalty state captured by the closure
        mu = [mu_init]

        def scipy_fun(var):
            loss, gradvec = self._compute_loss_and_gradient(
                tf.cast(var, tf.float32), objective, meta, pbar, start_time, mu,
            )
            return (
                np.real(loss.numpy()).astype(np.float64),
                np.real(gradvec.numpy()).astype(np.float64),
            )

        return scipy_fun

    def _compute_loss_and_gradient(self, var, objective, meta: _VariableMetadata, pbar, start_time, mu):
        """
        Computes the loss and gradient for one L-BFGS-B iteration.

        Uses param_scopes/param_ids from *meta* instead of the old varinfo dict.
        NFIT-scoped parameters are batched along the bead dimension;
        SHARED parameters are used as-is.

        IMPORTANT: The batching loop passes raw tensor lists (not
        LearnablePSFParameters) to the objective.  This is necessary because
        tf.GradientTape watches raw tensors, and wrapping them in new
        tf.Variables via fromTensorList would break the gradient chain (new
        Variables copy values without creating a differentiable connection
        to the watched tensors).

        Parameters
        ----------
        var : tf.Tensor
            Current flat variable vector (float32).
        objective : callable
            The user's objective function.
        meta : _VariableMetadata
            Flattening metadata from ``_flatten_variables``.
        pbar : tqdm progress bar
        start_time : float
        mu : list[float]
            Single-element mutable list holding the current penalty parameter.
            Updated in-place to grow the penalty across iterations.

        Returns
        -------
        loss : tf.Tensor
        gradvec : tf.Tensor (1-D)
        """
        loss = 0.0
        Np = len(meta.shapes)
        Nfit = meta.shapes[0][0]
        start = time.time()
        grad = [None] * Np
        batchsize = self.batch_size
        variables = self._reshape_variables_tf(var, meta)
        params = variables.toLearnableParameterList()
        ind = list(np.int32(np.linspace(0, Nfit, Nfit // batchsize + 2)))
        batch_tensors = [None] * Np

        for i in range(len(ind) - 1):
            for k in range(Np):
                if meta.param_scopes[k] == ParameterScope.NFIT:
                    if meta.param_ids[k] == 0:
                        batch_tensors[k] = params[k].value[ind[i]:ind[i+1]]
                    elif meta.param_ids[k] == 1:
                        batch_tensors[k] = params[k].value[:, ind[i]:ind[i+1]]
                else:
                    batch_tensors[k] = params[k].value
                    if i == 0:
                        grad[k] = 0.0

            # Pass raw tensor list (NOT wrapped in LearnablePSFParameters)
            # so that the gradient tape can track the watched tensors through
            # the objective function. fromTensorList would create new tf.Variables
            # that have no gradient connection to the watched batch_tensors.
            with tf.GradientTape() as tape:
                tape.watch(batch_tensors)
                loss1 = objective(batch_tensors, mu[0], ind[i:i+2])
            w1 = batch_tensors[0].shape[0] / Nfit
            loss = loss + loss1 * w1
            grad1 = tape.gradient(loss1, batch_tensors)

            for k in range(Np):
                if grad1[k] is None:
                    grad1[k] = batch_tensors[k] * 0

            for k in range(Np):
                if meta.param_scopes[k] == ParameterScope.NFIT:
                    if grad[k] is None:
                        grad[k] = grad1[k]
                    else:
                        grad[k] = tf.concat((grad[k], grad1[k]), axis=meta.param_ids[k])
                else:
                    grad[k] = grad[k] + grad1[k] * w1

        pbar.postfix[-1]['loss'] = loss
        pbar.postfix[-1]['time'] = start_time + pbar._time() - pbar.start_t
        pbar.update(1)
        self.update_history(self.step + 1, time.time() - start, loss.numpy())
        self.step += 1

        gradvec = tf.reshape(grad[0], [-1])
        for g in grad[1:]:
            gradvec = tf.concat((gradvec, tf.reshape(g, [-1])), axis=0)

        mu[0] *= self.rate
        mu[0] = np.min([1e7, mu[0]])

        return loss, gradvec

