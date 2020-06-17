# -*- coding: utf-8 -*-
"""Testing the PAL module"""
from __future__ import absolute_import

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from dispersant_screener.pal import (_get_gp_predictions, _get_uncertainity_region, _union_one_dim)


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
    mu = 1  # pylint:disable=invalid-name

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


def test__union_one_dim():
    """Make sure that the intersection of the uncertainity regions works"""
    zeros = np.array([0, 0, 0])
    zero_one_one = np.array([0, 1, 1])
    # Case 1: Everything is zero, we should also return zero
    low, up = _union_one_dim([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])  # pylint:disable=invalid-name

    assert (low == zeros).all()
    assert (up == zeros).all()

    # Case 2: This should also work if this is the case for only one material
    low, up = _union_one_dim([0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1])  # pylint:disable=invalid-name
    assert (low == zero_one_one).all()
    assert (up == zero_one_one).all()

    # Case 3: Uncertainity regions do not intersect
    low, up = _union_one_dim([0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3])  # pylint:disable=invalid-name
    assert (low == zeros).all()
    assert (up == zeros).all()

    # Case 4: We have an intersection
    low, up = _union_one_dim([0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [3, 3, 3])  # pylint:disable=invalid-name
    assert (low == np.array([0.5, 0.5, 0.5])).all()
    assert (up == np.array([1, 1, 1])).all()


def test_pal():
    """
    It's a bit tricky to test this one but we can do two things easily.
    (1) If we copy the same target multiple times, we should only find one
    Pareto-optimal point.
    (2) We can check how good the hypervolume is.
    """

    ...
