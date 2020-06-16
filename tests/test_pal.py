# -*- coding: utf-8 -*-
"""Testing the PAL module"""
from __future__ import absolute_import

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from dispersant_screener.pal import (_get_gp_predictions, _get_uncertainity_region)


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


def test__get_uncertainity_region():
    """make sure that the uncertainity windows is computed in a reasonable way"""
    mu = 1 # pylint:disable=invalid-name

    low0, high0 = _get_uncertainity_region(mu, 0, 1)
    assert low0 == mu
    assert high0 == mu

    low1, high1 = _get_uncertainity_region(mu, 1, 0)
    assert low1 == mu
    assert high1 == mu

    low2, high2 = _get_uncertainity_region(mu, 0, 0)
    assert low2 == mu
    assert high2 == mu

    low3, high3 = _get_uncertainity_region(mu, 1, 1)
    assert low3 == 0
    assert high3 == 2
