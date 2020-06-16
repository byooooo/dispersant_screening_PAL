# -*- coding: utf-8 -*-
"""Implement the (epislon)-PAL sampling"""

from __future__ import absolute_import

import logging
from typing import Iterable, Union

import numpy as np
import pygmo as pg
from tqdm import tqdm


def get_hypervolume(pareto_front: np.array, reference_vector: np.array) -> float:
    """Compute the hypervolume indicator of a Pareto front"""
    hyp = pg.hypervolume(pareto_front)
    volume = hyp.compute(reference_vector)  # uses 'auto' algorithm
    return volume


def _get_gp_predictions(gps: Iterable, x_train: np.array, y_train: np.array,
                        x_input: np.array) -> Union[np.array, np.array]:
    """Train one GPR per target using x_train and predict on x_input

    Args:
        gps (Iterable): Contains the GPR models.
            We require the GPRs to have a sklearn-like API with .fit(X,y)
            and .predict(x, return_std=True) methods
        x_train (np.array): Feature matrix. Must be already standardized.
            Shape (number points, number features)
        y_train (np.array): Target vector. Shape (number points, number targets)
        x_input (np.array): Feature matrix for new query points.
            Shape (number points, number features)

    Returns:
        Union[np.array, np.array]: Means and standard deviations in arrays of shape
            (len(x_input), number targets)
    """
    mus = []
    stds = []

    # train one GP per target
    # ToDo: one can potentially parallelize this part
    for i, gp in enumerate(gps):  # pylint:disable=invalid-name
        gp.fit(x_train, y_train[:, i])
        mu, std = gp.predict(x_input, return_std=True)  # pylint:disable=invalid-name
        mus.append(mu.reshape(-1, 1))
        stds.append(std.reshape(-1, 1))

    return np.hstack(mus), np.hstack(stds)


def _get_uncertainity_region(mu: float, std: float, beta_sqrt: float):  # pylint:disable=invalid-name
    low_lim, high_lim = mu - beta_sqrt * std, mu + beta_sqrt * std
    return low_lim, high_lim
