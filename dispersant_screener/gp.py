# -*- coding: utf-8 -*-
"""
Wrappers for Gaussian Process Regression models.

We typically use the GPy package as it offers most flexibility
for Gaussian processes in Python.
Typically, we use automatic relevance determination (ARD),
where one lengthscale parameter per input dimension is used.

If your task requires training on larger training sets,
you might consider replacing the models with their sparse
version but for the epsilon-PAL algorithm this typically shouldn't
be needed.

For kernel selection, you can have a look at
https://www.cs.toronto.edu/~duvenaud/cookbook/
Matérn, RBF and RationalQuadrat are good quick and dirty solutions
but have their caveats
"""
from __future__ import absolute_import

from typing import Tuple

import GPy
import numpy as np


def _get_matern_32_kernel(NFEAT: int, **kwargs) -> GPy.kern.Matern32:
    return GPy.kern.Matern32(NFEAT, ARD=True, **kwargs)


def _get_matern_52_kernel(NFEAT: int, **kwargs) -> GPy.kern.Matern52:
    return GPy.kern.Matern52(NFEAT, ARD=True, **kwargs)


def _get_ratquad_kernel(NFEAT: int, **kwargs) -> GPy.kern.RatQuad:
    return GPy.kern.RatQuad(NFEAT, ARD=True, **kwargs)


# ToDo: Generalize to more than two outputs
def build_coregionalized_model(X_train: np.array,
                               y_train: np.array,
                               kernel=None) -> GPy.models.GPCoregionalizedRegression:
    NFEAT = X_train.shape[1]
    if isinstance(kernel, GPy.kern.src.kern.Kern):
        K = kernel
    else:
        K = _get_ratquad_kernel(NFEAT)
    icm = GPy.util.multioutput.ICM(input_dim=NFEAT, num_outputs=2, kernel=K)
    m = GPy.models.GPCoregionalizedRegression([X_train, X_train],
                                              [y_train[:, 0].reshape(-1, 1), y_train[:, 1].reshape(-1, 1)],
                                              kernel=icm)
    return m


def build_model(X_train: np.array, y_train: np.array, index: int = 0, kernel=None) -> GPy.models.GPRegression:
    NFEAT = X_train.shape[1]
    if isinstance(kernel, GPy.kern.src.kern.Kern):
        K = kernel
    else:
        K = _get_ratquad_kernel(NFEAT)
    m = GPy.models.GPRegression(X_train, y_train[:, index].reshape(-1, 1), kernel=K)
    return m


def predict(model: GPy.models.GPRegression, X: np.array) -> Tuple[np.array, np.array]:
    assert isinstance(model, GPy.models.GPRegression), 'This wrapper function is written for GPy.models.GPRegression'
    mu, var = model.predict(X)
    return mu, var


def predict_coregionalized(model: GPy.models.GPCoregionalizedRegression,
                           X: np.array,
                           index: int = 0) -> Tuple[np.array, np.array]:
    assert isinstance(model, GPy.models.GPCoregionalizedRegression
                     ), 'This wrapper function is written for GPy.models.GPCoregionalizedRegression'
    newX = np.hstack([X, index * np.ones_like(X)])
    mu_c0, var_c0 = model.predict(newX, Y_metadata={'output_index': index * np.ones((newX.shape[0], 1)).astype(int)})

    return mu_c0, var_c0
