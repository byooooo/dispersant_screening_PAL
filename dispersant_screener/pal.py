# -*- coding: utf-8 -*-
"""Implement the (epislon)-PAL sampling,
If we refer to equations in the paper we refer to mostly to the PAL paper

Marcela Zuluaga, Guillaume Sergent, Andreas Krause, Markus Püschel;
Proceedings of the 30th International Conference on Machine Learning,
PMLR 28(1):462-470, 2013. (http://proceedings.mlr.press/v28/zuluaga13.html)

Epsilon-PAL was described in Journal of Machine Learning Research, Vol. 17, No. 104, pp. 1-32, 2016
(http://jmlr.org/papers/v17/15-047.html)
and introduces some epsilon parameter that allows to tradeoff accuracy and efficiency

To make the code simpler, I'll assume that greater is better. If this is not the case, the target needs
be be transformed (e.g., multiplied with minus one).

Many parts in this module are still relatively inefficient and could be vectorized/parallelized.
Also, many parts use lookup tables that are not super efficient.
"""

from __future__ import absolute_import

import logging
from typing import Iterable, Union

import numpy as np
import pygmo as pg
from six.moves import range
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


def _get_uncertainity_region(mu: float, std: float, beta_sqrt: float) -> Union[float, float]:  # pylint:disable=invalid-name
    """

    Args:
        mu (float): mean
        std (float): standard deviation
        beta_sqrt (float): scaling factor

    Returns:
        Union[float, float]: lower bound, upper bound
    """
    low_lim, high_lim = mu - beta_sqrt * std, mu + beta_sqrt * std
    return low_lim, high_lim


def _get_uncertainity_regions(mus: np.array, stds: np.array, beta_sqrt: float) -> Union[np.array, np.array]:
    """
    Compute the lower and upper bound of the uncertainty region for each dimension (=target)

    Args:
        mus (np.array): means
        stds (np.array): standard deviations
        beta_sqrt (float): scaling factors

    Returns:
        Union[np.array, np.array]: lower bounds, upper bounds
    """
    low_lims, high_lims = [], []

    assert mus.shape[1] >= 1

    for i in range(0, mus.shape[1]):  # pylint:disable=consider-using-enumerate (I find this clearer)
        low_lim, high_lim = _get_uncertainity_region(mus[:, i], stds[:, i], beta_sqrt)
        low_lims.append(low_lim.reshape(-1, 1))
        high_lims.append(high_lim.reshape(-1, 1))

    assert len(low_lims) == len(high_lims)

    return np.hstack(low_lims), np.hstack(high_lims)


def _union_one_dim(lows: Iterable, ups: Iterable, new_lows: Iterable, new_ups: Iterable) -> Union[np.array, np.array]:
    """Used to intersect the confidence regions, for eq. 6 of the PAL paper.
    "The iterative intersection ensures that all uncertainty regions are non-increasing with t."

    We do not check for the ordering in this function.
    We really assume that the lower limits are the lower limits and the upper limits are the upper limits.

    All arrays must have the same length.

    Args:
        lows (Iterable): lower bounds from previous iteration
        ups (Iterable): upper bounds from previous iteration
        new_lows (Iterable): lower bounds from current iteration
        new_ups (Iterable): upper bounds from current iteration

    Returns:
        Union[np.array, np.array]: array of lower limits, array of upper limits
    """

    # ToDo: can probably be vectorized (?)
    out_lows = []
    out_ups = []

    assert len(lows) == len(ups) == len(new_lows) == len(new_ups)

    for i, low in enumerate(lows):
        # In one dimension we can imagine the following cases where there
        # is zero intersection
        # 1) |--old range--|   |--new range--|, i.e., lower new limit above old upper limit
        # 2) |--new range--|   |--old range--|, i.e., upper new limit below lower old limit
        # 3) ||||, i.e. no uncertainity at all
        if (new_lows[i] > ups[i]) or (new_ups[i] < low) or (low + ups[i] == 0):
            out_lows.append(0)
            out_ups.append(0)

        # In other cases, we want to intersect the ranges, i.e.
        # |---old range-|-|--new-range--| --> |-|
        # i.e. we take the max of the lower limits and the min of the upper limits
        else:
            out_lows.append(max(low, new_lows[i]))
            out_ups.append(min(ups[i], new_ups[i]))

    assert len(out_lows) == len(out_ups) == len(lows)

    return np.array(out_lows), np.array(out_ups)


def _union(lows: np.array, ups: np.array, new_lows: np.array, new_ups: np.array) -> Union[np.array, np.array]:
    """Performing iterative intersection (eq. 6 in PAL paper) in all dimensions.

    Args:
        lows (np.array): lower bounds from previous iteration
        ups (np.array): upper bounds from previous iteration
        new_lows (np.array): lower bounds from current iteration
        new_ups (np.array): upper bounds from current iteration

    Returns:
        Union[np.array, np.array]: lower bounds, upper bounds
    """
    out_lows = []
    out_ups = []

    assert lows.shape == ups.shape == new_lows.shape == new_ups.shape

    for i in range(0, lows.shape[1]):  # pylint:disable=consider-using-enumerate (I find this clearer)
        low, up = _union_one_dim(lows[:, i], ups[:, i], new_lows[:, i], new_ups[:, i])  # pylint:disable=invalid-name
        out_lows.append(low.reshape(-1, 1))
        out_ups.append(up.reshape(-1, 1))

    out_lows, out_ups = np.hstack(out_lows), np.hstack(out_ups)

    assert out_lows.shape == out_ups.shape == lows.shape

    return out_lows, out_ups


def _update_sampled(mus: np.array,
                    stds: np.array,
                    sampled: Iterable,
                    y_input: np.array,
                    noisy_sample: bool = True) -> Union[np.array, np.array]:
    """
    For the points that we sample replace the mu with the actual (evaluated)
    value.

    The arrays are not modified in place.

    Args:
        mus (np.array): means
        stds (np.array): standard deviations
        sampled (Iterable): boolean lookup table. True if sampled.
        y_input (np.array): evaluated values
        noisy_sample (bool): if True, we will keep the standard deviation for the sampled points.
            This is basically the assumption that we only have access to a noisy sample {default: True}

    Returns:
        Union[np.array, np.array]: means, standard deviations
    """
    mu_ = mus.copy()
    std_ = stds.copy()

    assert mu_.shape == std_.shape

    # this is kinda inefficient, better use some kind of indexing
    for i in range(0, len(mus)):
        if sampled[i] == 1:
            mu_[i, :] = y_input[i, :]

            if not noisy_sample:
                std_[i, :] = 0

    assert mu_.shape == std_.shape == mus.shape

    return mu_, std_


def _pareto_classify(  # pylint:disable=too-many-arguments
        pareto_optimal_0: list, not_pareto_optimal_0: list, unclassified_0: list, rectangle_lows: np.array,
        rectangle_ups: np.array, x_input: np.array, epsilon: float) -> Union[list, list, list]:
    """Performs the classification part of the algorithm
    (p. 4 of the PAL paper, see algorithm 1/2 of the epsilon-PAL paper)

    One core concept is that once a point is classified it does no longer change the class.

    Args:
        pareto_optimal_0 (list): binary encoded list of points classified as Pareto optimal
        not_pareto_optimal_0 (list): binary encoded list of points classified as non-Pareto optimal
        unclassified_0 (list): binary encoded list of unclassified points
        rectangle_lows (np.array): lower uncertainity boundaries
        rectangle_ups (np.array): upper uncertainity boundaries
        x_input (np.array): feature matrix
        epsilon (float): granularity parameter

    Returns:
        Union[list, list, list]: binary encoded list of Pareto optimal, non-Pareto optimal and unclassified points
    """

    pareto_optimal_t = pareto_optimal_0.copy()
    not_pareto_optimal_t = not_pareto_optimal_0.copy()
    unclassified_t = unclassified_0.copy()

    # loop over samples
    # this is an ugly nested loop that likely can be optimized, but for not it is better that it is readable and works.
    # first check the pareto pessimistic front of pareto_optimal_t
    for i in range(0, len(x_input)):
        # we only care about points that are unclassified. Once a point is classified, it does not change class.
        # discard points x in Ut \ ppess(Pt ∪ Ut) that are ε-dominated by some point x′ in ppess(Pt ∪ Ut), i.e.,
        # where max(Rt(x)) ≼ε min(Rt(x′)). (See Alg. 2 for m = 2)
        # the paper proposes a more efficient implementation
        if unclassified_t[i] == 1:
            for j in range(0, len(x_input)):
                if (i != j) and ((pareto_optimal_0[j] == 1) or (unclassified_t[j] == 1)):
                    # epsilon dominated by j
                    if (rectangle_lows[j] * (1 + epsilon) >= rectangle_ups[i]).all():
                        not_pareto_optimal_t[i] = 1
                        unclassified_t[i] = 0
                        break

    # now, update the pareto set
    # if there is no other point x' such that max(Rt(x')) >= min(Rt(x))
    # move x to Pareto
    for i in range(0, len(x_input)):
        # again, we only care about unclassified points
        if unclassified_t[i] == 1:
            pareto = True
            for j in range(0, len(x_input)):
                if (i != j) and ((pareto_optimal_0[j] == 1) or (unclassified_t[j] == 1)):
                    # i epsilon dominates
                    if (rectangle_ups[j] >= rectangle_lows[i] * (1 + epsilon)).all():
                        pareto = False
                        break
            if pareto:
                pareto_optimal_t[i] = 1
                unclassified_t[i] = 0

    return pareto_optimal_t, not_pareto_optimal_t, unclassified_t


def _sample(  # pylint:disable=too-many-arguments
        rectangle_lows: np.array, rectangle_ups: np.array, pareto_optimal_t: Iterable, unclassified_t: Iterable,
        sampled: Iterable, x_input: np.array, y_input: np.array, x_train: np.array,
        y_train: np.array) -> Union[np.array, np.array, Iterable]:
    """Perform sampling based on maximal diagonal of uncertainty.

    Args:
        rectangle_lows (np.array): lower limits of uncertainity
        rectangle_ups (np.array): upper limits of uncertainity
        pareto_optimal_t (Iterable): binary encoded list of points classified as Pareto optimal
        unclassified_t (Iterable):  binary encoded list of unclassified points
        sampled (Iterable):  binary encoded list of points that were sampled
        x_input (np.array): feature matrix of sampling space
        y_input (np.array): matrix of labels of sampling space
        x_train (np.array): feature matrix of training set
        y_train (np.array): labels of training set

    Returns:
        Union[np.array, np.array, Iterable]: updated feature matrix of training set,
            updated label matrix of training set, updated binary encoded list of sampled points
    """

    max_uncertainity = 0
    maxid = -1

    # not needed but can be safer if not used as expected
    x_train_ = sampled.copy()
    x_train_ = x_train.copy()
    y_train_ = y_train.copy()

    for i in range(0, len(x_input)):
        # Among the points x ∈ Pt ∪ Ut, the one with the largest wt(x) is chosen as the next sample xt to be evaluated.
        # Intuitively, this rule biases the sampling towards exploring,
        # and thus improving the model for, the points most likely to be Pareto-optimal.
        if ((unclassified_t[i] == 1) or (pareto_optimal_t[i] == 1)) and not sampled[i] == 1:
            # weight is the length of the diagonal of the uncertainity region
            uncertainity = np.linalg.norm(rectangle_ups[i, :] - rectangle_lows[i, :])
            if maxid == -1:
                max_uncertainity = uncertainity
                maxid = i
            # the point with the largest weight is chosen as the next sample
            elif uncertainity > max_uncertainity:
                max_uncertainity = uncertainity
                maxid = i

    x_train_ = np.insert(x_train_, x_train_.shape[0], x_input[maxid], axis=0)
    y_train_ = np.insert(y_train_, y_train_.shape[0], y_input[maxid], axis=0)

    x_train_[maxid] = 1

    return x_train_, y_train_, x_train_
