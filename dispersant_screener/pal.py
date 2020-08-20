# -*- coding: utf-8 -*-
"""Implements the (epislon)-PAL sampling,
If we refer to equations in the paper we refer to mostly to the PAL paper

Marcela Zuluaga, Guillaume Sergent, Andreas Krause, Markus Püschel;
Proceedings of the 30th International Conference on Machine Learning,
PMLR 28(1):462-470, 2013. (http://proceedings.mlr.press/v28/zuluaga13.html)

Epsilon-PAL was described in Journal of Machine Learning Research, Vol. 17, No. 104, pp. 1-32, 2016
(http://jmlr.org/papers/v17/15-047.html)
and introduces some epsilon parameter that allows to tradeoff accuracy and efficiency

To make the code simpler, I'll assume that greater is better, we aim to maximize all objectives.
If this is not applicable for your setting you might simply flip the sign or apply another
appropriate transformation to your data such that it is a maximization problem in all objectives.

Many parts in this module are still relatively inefficient and could be vectorized/parallelized.
Also, many parts use lookup tables that are not super efficient.

Currently, on the first run, there will still be warnings as the jit compilation is not optimized.
I currently mainly use the looplifting although additional speedups are certainly possible.
"""
import logging
from typing import List, Sequence, Tuple, Union

import numpy as np
import pygmo as pg
from numba import jit
from tqdm import tqdm

from dispersant_screener.gp import (predict_coregionalized, set_xy_coregionalized)
from dispersant_screener.utils import (dump_pickle, is_pareto_efficient, vectorized_dominance_check)

logger = logging.getLogger('PALlogger')  # pylint:disable=invalid-name
handler = logging.StreamHandler()  # pylint:disable=invalid-name
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')  # pylint:disable=invalid-name
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_hypervolume(pareto_front: np.array, reference_vector: np.array, prefactor: float = -1) -> float:
    """Compute the hypervolume indicator of a Pareto front
    I multiply it with minus one as we assume that we want to maximize all objective and then we calculate the area

    f1
    |
    |----|
    |     -|
    |       -|
    ------------ f2

    But pygmo assumes that the reference vector is larger than all the points in the Pareto front.
    For this reason, we then flip all the signs using prefactor (i.e., you can use it to toggle between maximization
    and minimization problems).

    This indicator is not needed for the epsilon-PAL algorithm itself but only to allow tracking a metric
    that might help the user to see if the algorithm converges.
    """
    hyp = pg.hypervolume(pareto_front * prefactor)
    volume = hyp.compute(reference_vector)  # uses 'auto' algorithm
    return volume


def _get_gp_predictions(gps: Sequence,
                        x_train: np.array,
                        y_train: np.array,
                        x_input: np.array,
                        coregionalized_model: bool = False,
                        optimize: bool = True,
                        parallel: bool = True) -> Union[np.array, np.array, List]:
    """Train one GPR per target using x_train and predict on x_input

    Args:
        gps (Sequence): Contains the GPR models. We assume GPy models.
        x_train (np.array): Feature matrix. Must be already standardized.
            Shape (number points, number features)
        y_train (np.array): Target vector. Shape (number points, number targets)
        x_input (np.array): Feature matrix for new query points.
            Shape (number points, number features)
        coregionalized (bool, optional): If True, we assume that there is one model in the gps
            list and that this model is a coregionalized GPy model that wil be able to
            model relationships between the output and that needs to be trained only
            once using all targets at the same time. We assume that the model was built
            using the build_coregionalized function of the gp module. Default is False.
        optimize (bool, optional): If True, run hyperparamter optimization. Default is True.
        parallel (bool, optional): If True, run the random restarts for hyperparamter optimization
            in parallel.

    Returns:
        Union[np.array, np.array, list]: Means and standard deviations in arrays of shape
            (len(x_input), number targets), list of trained  models
    """
    mus = []
    stds = []

    gps_trained = []

    num_targets = y_train.shape[1]

    # train one GP per target
    # ToDo: one can potentially parallelize this part
    if not coregionalized_model:
        for i, gp in enumerate(gps):  # pylint:disable=invalid-name
            gp.set_XY(x_train, y_train[:, i].reshape(-1, 1))
            if optimize:
                gp.optimize_restarts(20, parallel=parallel)
            mu, std = gp.predict(x_input)  # pylint:disable=invalid-name

            mus.append(mu.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

            gps_trained.append(gp)

    if coregionalized_model:
        gp = gps[0]  # pylint:disable=invalid-name
        gp = set_xy_coregionalized(gp, x_train, y_train)  # pylint:disable=invalid-name
        if optimize:
            gp.optimize_restarts(20, parallel=parallel)

        for i in range(num_targets):
            mu, std = predict_coregionalized(gp, x_input, i)  # pylint:disable=invalid-name

            mus.append(mu.reshape(-1, 1))
            stds.append(std.reshape(-1, 1))

        gps_trained.append(gp)

    return np.hstack(mus), np.hstack(stds), gps_trained


def _get_uncertainity_region(mu: np.array, std: np.array, beta_sqrt: float) -> Tuple[np.array, np.array]:  # pylint:disable=invalid-name
    """

    Args:
        mu (float): mean
        std (float): standard deviation
        beta_sqrt (float): scaling factor

    Returns:
        Tuple[float, float]: lower bound, upper bound
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

    for i in range(0, mus.shape[1]):  # pylint:disable=consider-using-enumerate (I find this clearer)
        low_lim, high_lim = _get_uncertainity_region(mus[:, i], stds[:, i], beta_sqrt)
        low_lims.append(low_lim.reshape(-1, 1))
        high_lims.append(high_lim.reshape(-1, 1))

    return np.hstack(low_lims), np.hstack(high_lims)


@jit
def _union_one_dim(lows: Sequence, ups: Sequence, new_lows: Sequence, new_ups: Sequence) -> Tuple[np.array, np.array]:
    """Used to intersect the confidence regions, for eq. 6 of the PAL paper.
    "The iterative intersection ensures that all uncertainty regions are non-increasing with t."

    We do not check for the ordering in this function.
    We really assume that the lower limits are the lower limits and the upper limits are the upper limits.

    All arrays must have the same length.

    Args:
        lows (Sequence): lower bounds from previous iteration
        ups (Sequence): upper bounds from previous iteration
        new_lows (Sequence): lower bounds from current iteration
        new_ups (Sequence): upper bounds from current iteration

    Returns:
        Tuple[np.array, np.array]: array of lower limits, array of upper limits
    """

    # ToDo: can probably be vectorized (?)
    out_lows = []
    out_ups = []

    for i, low in enumerate(lows):
        # In one dimension we can imagine the following cases where there
        # is zero intersection
        # 1) |--old range--|   |--new range--|, i.e., lower new limit above old upper limit
        # 2) |--new range--|   |--old range--|, i.e., upper new limit below lower old limit
        if (new_lows[i] >= ups[i]) or (new_ups[i] <= low):
            out_lows.append(new_lows[i])
            out_ups.append(new_ups[i])

        # In other cases, we want to intersect the ranges, i.e.
        # |---old range-|-|--new-range--| --> |-|
        # i.e. we take the max of the lower limits and the min of the upper limits
        else:
            out_lows.append(max(low, new_lows[i]))
            out_ups.append(min(ups[i], new_ups[i]))

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

    for i in range(0, lows.shape[1]):  # pylint:disable=consider-using-enumerate (I find this clearer)
        low, up = _union_one_dim(lows[:, i], ups[:, i], new_lows[:, i], new_ups[:, i])  # pylint:disable=invalid-name
        out_lows.append(low.reshape(-1, 1))
        out_ups.append(up.reshape(-1, 1))

    out_lows_array, out_ups_array = np.hstack(out_lows), np.hstack(out_ups)

    return out_lows_array, out_ups_array


@jit
def _update_sampled(mus: np.array,
                    stds: np.array,
                    sampled: Sequence,
                    y_input: np.array,
                    noisy_sample: bool = False) -> Union[np.array, np.array]:
    """
    For the points that we sample replace the mu with the actual (evaluated)
    value.

    The arrays are not modified in place.

    Args:
        mus (np.array): means
        stds (np.array): standard deviations
        sampled (Sequence): boolean lookup table. True if sampled.
        y_input (np.array): evaluated values
        noisy_sample (bool): if True, we will keep the standard deviation for the sampled points.
            This is basically the assumption that we only have access to a noisy sample {default: True}

    Returns:
        Union[np.array, np.array]: means, standard deviations
    """
    mu_ = mus
    std_ = stds
    # this is kinda inefficient, better use some kind of indexing
    for i in range(0, len(mus)):
        if sampled[i] == 1:
            mu_[i, :] = y_input[i, :]

            if not noisy_sample:
                std_[i, :] = 0

    return mu_, std_


def _pareto_classify(  # pylint:disable=too-many-arguments
        pareto_optimal_0: list, not_pareto_optimal_0: list, unclassified_0: list, rectangle_lows: np.array,
        rectangle_ups: np.array, x_input: np.array, epsilon: list) -> Tuple[list, list, list]:
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
        epsilon (list): granularity parameter (one per dimension)

    Returns:
        Tuple[list, list, list]: binary encoded list of Pareto optimal, non-Pareto optimal and unclassified points
    """
    pareto_optimal_t = pareto_optimal_0.copy()
    not_pareto_optimal_t = not_pareto_optimal_0.copy()
    unclassified_t = unclassified_0.copy()

    # This part is only relevant when we have points in set P
    # Then we can use those points to discard points from p_pess (P \cup U)
    logger.debug('Comparing against p_pess(P)')
    if sum(pareto_optimal_0) > 0:
        pareto_indices = np.where(pareto_optimal_0 == 1)[0]
        pareto_pessimistic_lows = rectangle_lows[pareto_indices]  # p_pess(P)
        for i in range(0, len(x_input)):
            if unclassified_t[i] == 1:
                if not np.any(
                        np.all(vectorized_dominance_check(rectangle_ups[i], pareto_pessimistic_lows * (1 + epsilon)),
                               axis=-1)):
                    not_pareto_optimal_t[i] = 1
                    unclassified_t[i] = 0

    # ToDo: can be probably cleaned up a bit
    pareto_unclassified_indices = np.where((pareto_optimal_0 == 1) | (unclassified_t == 1))[0]
    pareto_unclassified_lows = rectangle_lows[pareto_unclassified_indices]
    pareto_unclassified_pessimistic_mask = is_pareto_efficient(-pareto_unclassified_lows)  # pylint:disable=invalid-name
    pareto_unclassified_pessimistic_points = pareto_unclassified_lows[pareto_unclassified_pessimistic_mask]  # pylint:disable=invalid-name

    logger.debug('Comparing against p_pess(P \cup U)')
    for i in range(0, len(x_input)):
        # We can only discard points that are unclassified so far
        # We cannot discard points that are part of p_pess(P \cup U)
        if (unclassified_t[i] == 1) and (i not in pareto_unclassified_indices):
            # If the upper bound of the hyperrectangle is not dominating anywhere the pareto pessimitic set, we can discard
            if not np.any(
                    np.all(vectorized_dominance_check(rectangle_ups[i],
                                                      pareto_unclassified_pessimistic_points * (1 + epsilon)),
                           axis=-1)):
                not_pareto_optimal_t[i] = 1
                unclassified_t[i] = 0

    # now, update the pareto set
    # if there is no other point x' such that max(Rt(x')) >= min(Rt(x))
    # move x to Pareto

    unclassified_indices = np.where((unclassified_t == 1) | (pareto_optimal_t == 1))[0]
    unclassified_ups = np.ma.array(rectangle_ups[unclassified_indices])
    # The index map helps us to mask the current point from the unclassified_ups list
    index_map = dict(zip(unclassified_indices, range(len(unclassified_ups))))

    logger.debug('Pareto front covering')
    for i in range(0, len(x_input)):
        # again, we only care about unclassified points
        if unclassified_t[i] == 1:
            # We need to make sure that unclassified_ups does not contain the current point
            unclassified_ups[index_map[i]] = np.ma.masked
            # If there is no other point which up is epsilon dominating the low of the current point,
            # the current point is epsilon-accurate Pareto optimal
            if not np.any(
                    np.all(vectorized_dominance_check(unclassified_ups, rectangle_lows[i] * (1 + epsilon)), axis=1)):
                pareto_optimal_t[i] = 1
                unclassified_t[i] = 0

            # now we can demask the entry
            unclassified_ups[index_map[i]] = np.ma.nomask

    return pareto_optimal_t, not_pareto_optimal_t, unclassified_t


# ToDo: Implement this as additional stopping criterion
# def _check_wt(points, indexer, rectangle_ups, rectangle_lows):
#     uncertainity = np.linalg.norm(rectangle_ups[i, :] - rectangle_lows[i, :])


@jit
def _sample(  # pylint:disable=too-many-arguments
        rectangle_lows: np.array, rectangle_ups: np.array, pareto_optimal_t: Sequence, unclassified_t: Sequence,
        sampled: List, x_input: np.array, y_input: np.array, x_train: np.array,
        y_train: np.array) -> Tuple[np.array, np.array, List]:
    """Perform sampling based on maximal diagonal of uncertainty.

    Args:
        rectangle_lows (np.array): lower limits of uncertainity
        rectangle_ups (np.array): upper limits of uncertainity
        pareto_optimal_t (Sequence): binary encoded list of points classified as Pareto optimal
        unclassified_t (Sequence):  binary encoded list of unclassified points
        sampled (List):  binary encoded list of points that were sampled
        x_input (np.array): feature matrix of sampling space
        y_input (np.array): matrix of labels of sampling space
        x_train (np.array): feature matrix of training set
        y_train (np.array): labels of training set

    Returns:
        Tuple[np.array, np.array, Sequence]: updated feature matrix of training set,
            updated label matrix of training set, updated binary encoded list of sampled points
    """

    max_uncertainity = 0
    maxid = -1

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

    x_train = np.insert(x_train, x_train.shape[0], x_input[maxid], axis=0)
    y_train = np.insert(y_train, y_train.shape[0], y_input[maxid], axis=0)

    sampled[maxid] = 1

    return x_train, y_train, sampled


def pal(  # pylint: disable=dangerous-default-value, too-many-arguments, too-many-locals, too-many-statements
        gps: list,
        x_train: np.array,
        y_train: np.array,
        x_input: np.array,
        y_input: np.array,
        hv_reference: Sequence = [30, 30],
        delta: float = 0.05,
        epsilon: Union[float, list] = 0.05,
        iterations: int = 500,
        verbosity: str = 'info',
        batch_size: int = 1,
        beta_scale: float = 1 / 9,
        coregionalized: bool = False,
        optimize_delay: int = 20,
        optimize_always: int = 10,
        parallel: bool = True,
        noisy_sample: bool = False,
        history_dump_file: str = None) -> Tuple[list, list, list, list]:
    """Orchestrates the PAL algorithm.
    You will see a progress bar appear. The length of this progress bar is equal to iterations,
    but the algorithm might stop earlier if all samples already have been classified.

    Args:
        gps (list):  Contains the GPR models. We require the GPRs to have a GPy API with .set_XY(X,Y),
            .optimize() and .predict(). You can use the gp module to build such models.
            The length of this list must equal the number of objectives, i.e, one GPR per objective.

        x_train (np.array): feature matrix for training data

        y_train (np.array): label matrix for training data (we assume this to be a two-dimensional array,
            use .reshape(-1,1) if you need to run on one target (for testing))

        x_input (np.array): feature matrix for sample space data

        y_input (np.array): label matrix for sample space data (we assume this to be a two-dimensional array,
            use .reshape(-1,1) if you need to run on one target (for testing))

        hv_reference (Sequence, optional): Reference vector for calculation of the hypervolume indicator.
            The length of this sequence must equal the number of objectives. Defaults to [30, 30].

        delta (float, optional): Hyperparameter, equivalent to the confidence level. Defaults to 0.05.

        epsilon (float or list, optional): Percentage tolerance for the Pareto front we find with this algorithm.
            You can provide a list of floats, with one epsilon per dimension or only one float. Defaults to 0.05.

        iterations (int, optional): Maximum number of iterations. Defaults to 500.

        verbosity (str, optional): Log levels. Implemented choices 'debug', 'info', 'warning'. On 'info' it will log
            the number of Pareto optimal/non-Pareto optimal/unclassified samples and the hypervolume.
            We can only compute a hypervolume if the number of points that are classified as Pareto optimal is greater
            or equal to one. Until then the hypervolume will be nan. The hypervolume indicator should increase over the course of the sampling. Defaults to 'info'.

        batch_size (int, optional): Experimental setting for the number of materials that are sampled
            in each iteration. The original algorithm does not consider batching. Defaults to 1 (no batching).

        beta_scale (float, optional): Scaling factor for beta_t (because the theoretical estimate is too conservative).
            Defaults to 1/9, as proposed in the e-PAL paper.

        coregionalized (bool, optional): If True, we assume that there is one model in the gps
            list and that this model is a coregionalized GPy model that wil be able to
            model relationships between the output and that needs to be trained only
            once using all targets at the same time. We assume that the model was built
            using the build_coregionalized function of the gp module.

        optimize_delay (int, optional): At multiples of this value (If iteration % optimize_delay = 0)
            we will perform hyperparameter optimization for the GPR models and the pick the one
            with the largest likelihood. Default is 30.

        optimize_always (int, optional): Up to this number of iterations, we will always
            perform hyperparamter optimization. Indpendent of the value of optimize_delay.
            Default is 10.

        parallel (bool, optional): If true, it runs random restarts for the hyperparameter optimization
            of the Gaussian processes in parallel.

       noisy_sample (bool, optional): If false, the standard deviation of sampled points is set to 0.
            If True, we use the standard deviation predicted by the model. Default is False.

        history_dump_file (str, optional): if this is not None but a filename, we will track the progress in the different sets
            over the optimization and dump a pickled dictionary with this filename

    Returns:
        Tuple[list, list, list, list]: binary encoded list of Pareto optimal points found in x_input,
            history of hypervolumes, list of trained models, list of sampled indices
    """
    # x_input is the E set in the PAL paper
    # the models for now assume the sklearn API, i.e., there should be a fit function
    assert y_train.shape[1] == len(
        hv_reference), 'The hypervolume reference and the number of objectives must be the same'
    if not coregionalized:
        assert y_train.shape[1] == len(
            gps), 'If you do not set coregionalized=True we assume that you use one model per objective'
    else:
        assert len(
            gps
        ) == 1, 'You chose coregionalized=True. In this case we assume that the model list has exactly one element---the coregionalized model.'

    if isinstance(epsilon, float):
        epsilon = [epsilon] * len(hv_reference)
    elif isinstance(epsilon, list):
        assert len(epsilon) == len(hv_reference)
        for eps in epsilon:
            assert isinstance(eps, float)

    epsilon = np.array(epsilon)

    progress_dict = {'sampled': [], 'pareto': [], 'non_pareto': [], 'unclassified': []}

    hypervolumes = []

    x_input = np.vstack([x_input, x_train])
    y_input = np.vstack([y_input, y_train])
    # initalize binary list to keep track of the different sets
    # in the beginning, nothing is selected = everything is unclassified
    # ToDo: move away from these lookup tables
    pareto_optimal_0 = np.array([0] * len(x_input))
    not_pareto_optimal_0 = np.array([0] * len(x_input))
    unclassified_0 = np.array([1] * len(x_input))
    sampled = np.array([0] * len(x_input))

    iteration = 0

    if verbosity == 'info':
        logger.setLevel(logging.INFO)
    elif verbosity == 'debug':
        logger.setLevel(logging.DEBUG)
    elif verbosity == 'warning':
        logger.setLevel(logging.WARNING)

    logger.info('Starting now the PAL loop')
    logger.debug('Will use the following settings')
    logger.debug('epsilon: {}, delta: {}, iterations: {}'.format(epsilon, delta, iterations))
    logger.debug('x_train shape: {}, y_train shape: {}'.format(x_train.shape, y_train.shape))

    # stop when all points are classified or we reached the maximum iteration
    pbar = tqdm(total=iterations)
    while (np.sum(unclassified_0) > 0) and (iteration <=
                                            iterations):  # also add potential check for uncertainity of Pareto front
        iteration += 1
        logger.debug('Starting iteration {}'.format(iteration))

        # STEP 1: modeling (train and predict using GPR, one GP per target)
        logger.debug('Starting modeling step, fitting the GPs')
        if iteration <= optimize_always:
            optimize = True
        elif iteration % optimize_delay == 0:
            optimize = True
        else:
            optimize = False
        mus, stds, gps = _get_gp_predictions(gps, x_train, y_train, x_input, coregionalized, optimize, parallel)

        # update scaling parameter β
        # which is achieved by choosing βt = 2 log(n|E|π2t2/(6δ)).
        # n: number of objectives (y_input.shape[1])
        # scaled because the theoretical value is too conservative
        beta = beta_scale * 2 * np.log(y_input.shape[1] * len(x_input) * np.square(np.pi) * np.square(iteration + 1) /
                                       (6 * delta))

        logger.debug('Scaling parameter beta at the current iteration is {}'.format(beta))

        logger.debug('mean array shape: {}, std array shape: {}'.format(mus.shape, stds.shape))

        # if point is sampled we know the mu and have no epistemic uncertainity
        mus, stds = _update_sampled(mus, stds, sampled, y_input, noisy_sample)

        logger.debug('mean array shape: {}, std array shape: {}'.format(mus.shape, stds.shape))
        logger.debug('mean array mean: {}, mean array std: {}'.format(np.mean(mus), np.std(mus)))

        # get the uncertainity rectangles, sqrt only once here for efficiency
        lows, ups = _get_uncertainity_regions(mus, stds, np.sqrt(beta))

        logger.debug('lows shape {}, ups shape {}'.format(lows.shape, ups.shape))

        if iteration == 1:
            # initialization
            rectangle_lows, rectangle_ups = lows, ups
        else:
            rectangle_lows, rectangle_ups = _union(rectangle_lows, rectangle_ups, lows, ups)

        logger.debug('rectangle lows shape {}, rectangle ups shape {}'.format(rectangle_lows.shape,
                                                                              rectangle_ups.shape))
        # pareto classification
        # update lists
        pareto_optimal_t, not_pareto_optimal_t, unclassified_t = _pareto_classify(pareto_optimal_0,
                                                                                  not_pareto_optimal_0, unclassified_0,
                                                                                  rectangle_lows, rectangle_ups,
                                                                                  x_input, epsilon)

        # sampling from x_input
        for _ in range(batch_size):
            x_train, y_train, sampled = _sample(rectangle_lows, rectangle_ups, pareto_optimal_t, unclassified_t,
                                                sampled, x_input, y_input, x_train, y_train)

        pareto_optimal_0, not_pareto_optimal_0, unclassified_0 = pareto_optimal_t, not_pareto_optimal_t, unclassified_t

        optimal_indices = np.where(np.array(pareto_optimal_0) == 1)[0]

        if (len(optimal_indices) > 0) and (y_input.shape[1] > 1):
            hypervolume = get_hypervolume(y_input[optimal_indices], hv_reference)
        else:
            hypervolume = np.nan

        logger.info('Iteration {} | Pareto optimal {}, not Pareto optimal {}, unclassified {}, hypervolume: {}'.format(
            iteration,
            np.array(pareto_optimal_0).sum(),
            np.array(not_pareto_optimal_0).sum(),
            np.array(unclassified_0).sum(), hypervolume))

        hypervolumes.append(hypervolume)

        if history_dump_file:
            progress_dict['sampled'].append(sampled)
            progress_dict['pareto'].append(pareto_optimal_0)
            progress_dict['non_pareto'].append(not_pareto_optimal_0)
            progress_dict['unclassified'].append(unclassified_0)

        pbar.update(1)

    pbar.close()

    if history_dump_file:
        dump_pickle(history_dump_file, progress_dict)

    return pareto_optimal_t, hypervolumes, gps, sampled
