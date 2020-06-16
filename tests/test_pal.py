# -*- coding: utf-8 -*-
"""Testing the PAL module"""
from __future__ import absolute_import

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from dispersant_screener.pal import _get_gp_predictions


def test__get_gp_predictions(make_random_dataset):
    """Check if the shapes after training and prediction make sense"""
    X, y = make_random_dataset  # pylint:disable=invalid-name
    X_train = X[:50]  # pylint:disable=invalid-name
    y_train = y[:50]

    X_input = X[50:]  # pylint:disable=invalid-name
    y_input = y[50:]

    gp0 = GaussianProcessRegressor(kernel=RBF())
    gp1 = GaussianProcessRegressor(kernel=RBF())
    gp2 = GaussianProcessRegressor(kernel=RBF())

    gprs = [gp0, gp1, gp2]

    mu, std = _get_gp_predictions(gprs, X_train, y_train, X_input)  # # pylint:disable=invalid-name

    assert mu.shape == y_input.shape
    assert std.shape == y_input.shape
