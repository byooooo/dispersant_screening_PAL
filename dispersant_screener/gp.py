# -*- coding: utf-8 -*-
"""
Implementing the Gaussian Process Regression models.
"""
from __future__ import absolute_import

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def build_virial_model():
    virial_model = GaussianProcessRegressor(kernel=RBF(), normalize_y=True, n_restarts_optimizer=30)

    return virial_model


def build_gibbs_model():
    gibbs_model = GaussianProcessRegressor(kernel=RBF(), normalize_y=True, n_restarts_optimizer=30)

    return gibbs_model
