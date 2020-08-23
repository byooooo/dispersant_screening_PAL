# -*- coding: utf-8 -*-
"""Functions that can be handy when using epsilon-PAL"""

import pickle
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from numpy import vectorize
from scipy.spatial import distance
from sklearn import metrics
from sklearn.cluster import KMeans
from tqdm import tqdm


def dump_pickle(file, object):
    with open(file, 'wb') as fh:
        pickle.dump(object, fh)


def read_pickle(file):
    with open(file, 'rb') as fh:
        res = pickle.load(fh)
    return res


def get_summary_stats_time(history: dict):
    """The PAL function gives the function to persist a pickle with the history
    (which samples were classified to which set in which iteration).

    This function loops over the items in this dict to make plotting easier"""

    pareto = []
    non_pareto = []
    sampled = []

    pareto_indices = []
    non_pareto_indices = []
    sampled_indices = []

    for array in history['pareto']:
        pareto.append(sum(array))
        pareto_indices.append(np.where(array == 1)[0])

    for array in history['non_pareto']:
        non_pareto.append(sum(array))
        non_pareto_indices.append(np.where(array == 1)[0])

    for array in history['sampled']:
        sampled.append(sum(array))
        sampled_indices.append(np.where(array == 1)[0])

    history_ = {}

    return pareto, non_pareto, sampled, pareto_indices, non_pareto_indices, sampled_indices


def animate_2d(i, ax, y, pareto_indices, non_pareto_indices, sampled_indices):
    """Use with FuncAnimate to see which points were assigned to which set in which iteration
    Example:
        from functools import partial
        from matplotlib import animation, rc

        pareto, non_pareto, sampled, pareto_indices, non_pareto_indices, sampled_indices = get_summary_stats_time(history)
        animate = partial(animate_2d, ax=ax, y=y, pareto_indices=pareto_indices, non_pareto_indices=non_pareto_indices, sampled_indices=sampled_indices)
        ani = animation.FuncAnimation(fig, animate,frames=range(301))

        # In Jupyter notebook
        from IPython.display import HTML
        rc('animation', html='jshtml')
        rc

        HTML(ani.to_jshtml())
    """
    ax.scatter(y[:, 0], y[:, 1], c='gray', s=1)
    ax.scatter(y[non_pareto_indices[i], 0], y[non_pareto_indices[i], 1], c='red', label='discarded', s=1)
    ax.scatter(y[sampled_indices[i], 0], y[sampled_indices[i], 1], c='blue', label='sampled', s=3)
    ax.scatter(y[pareto_indices[i], 0], y[pareto_indices[i], 1], c='green', label='Pareto', s=3)


def plot_parity(objective_tuples: list, outname: str = None, titles: list = None):
    """Plot a parity plot for n targets"""
    plt.rcParams['font.family'] = 'sans-serif'

    fig, ax = plt.subplots(1, len(objective_tuples))  # pylint: disable=invalid-name

    for i, objective_tuple in enumerate(objective_tuples):

        assert len(objective_tuple[0]) == len(objective_tuple[1]) == len(objective_tuple[2])
        ax[i].scatter(objective_tuple[0], objective_tuple[1], c=objective_tuple[2], cmap='coolwarm', s=.3)

    for a in ax:  # pylint:disable=invalid-name
        a.spines['left'].set_smart_bounds(True)
        a.spines['bottom'].set_smart_bounds(True)
        a.set_xlabel(r'$y_{true}$')
        lims = [
            np.min([a.get_xlim(), a.get_ylim()]),  # min of both axes
            np.max([a.get_xlim(), a.get_ylim()]),  # max of both axes
        ]
        a.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    if isinstance(titles, list):
        assert len(titles) == len(objective_tuples)
        for a, title in zip(ax, titles):  # pylint:disable=invalid-name
            a.set_title(title)

    ax[0].set_ylabel(r'$\hat{y}$')
    fig.tight_layout()

    if outname is not None:
        fig.savefig(outname, bbox_inches='tight')
        plt.close(fig)


def plot_pareto_2d(y, sampled):  # pylint:disable=invalid-name
    """Plot Pareto frontier in two dimensions"""
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(1, 1)  # pylint:disable=invalid-name

    ax.scatter(y[:, 0], y[:, 1])
    ax.scatter(y[sampled, 0], y[sampled, 1])

    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)

    ax.set_xlabel(r'$y_{0}$')
    ax.set_ylabel(r'$y_{1}$')

    fig.tight_layout()


def get_metrics(y_true: np.array, y_pred: np.array) -> dict:
    """Get a dictionary with basic regression metrics"""
    r2 = metrics.r2_score(y_true, y_pred)  # pylint:disable=invalid-name
    mae = metrics.mean_absolute_error(y_true, y_pred)  # pylint:disable=invalid-name
    max_error = metrics.max_error(y_true, y_pred)  # pylint:disable=invalid-name

    return {'r2': r2, 'mae': mae, 'max_error': max_error}


def get_variance_descriptors(var: np.array) -> dict:
    """Calculate summary statistics for an array"""
    return {
        'max_var': np.max(var),
        'min_var': np.min(var),
        'mean_var': np.mean(var),
        'median_var': np.median(var),
        'std_var': np.std(var)
    }


def add_postfix_to_keys(d: dict, postfix: str) -> dict:  # pylint:disable=invalid-name
    """Update keys of a dictionary from "key" to "key_postfix"""
    d_ = {}  # pylint:disable=invalid-name
    postfix = str(postfix)
    for key, value in d.items():
        d_[key + '_' + postfix] = value

    return d_


def get_kmeans_samples(
        X: np.array,  # pylint:disable=invalid-name
        y: np.array,
        n_samples: int,
        **kwargs) -> Tuple[np.array, np.array, list]:
    """Get the samples that are closest to the k=n_samples centroids

    Args:
        X (np.array): Feature array, on which the KMeans clustering is run
        y (np.array): label array
        n_samples (int): number of samples are should be selected
        **kwargs passed to the KMeans

    Returns:
        Tuple[np.array, np.array, list]: Feature array, label array, selected_indices
    """
    assert len(X) == len(y), 'X and y need to have the same length'
    assert len(X) > n_samples, 'The numbers of points that shall be selected (n_samples),\
         needs to be smaller than the length of the feature matrix (X)'

    assert n_samples > 0 and isinstance(n_samples, int), 'The number of points that shall be selected (n_samples)\
             needs to be an integer greater than 0'

    kmeans = KMeans(n_samples, **kwargs).fit(X)
    cluster_centers = kmeans.cluster_centers_
    closest, _ = metrics.pairwise_distances_argmin_min(cluster_centers, X)

    return X[closest, :], y[closest, :], closest


def get_maxmin_samples(
        X: np.array,  # pylint:disable=invalid-name
        y: np.array,
        n_samples: int,
        metric: str = 'euclidean',
        init: str = 'mean',
        seed: int = None,
        **kwargs) -> Tuple[np.array, np.array, list]:
    """Greedy maxmin sampling, also known as Kennard-Stone sampling (1).
    Note that a greedy sampling is not guaranteed to give the ideal solution
    and the output will depend on the random initialization (if this is chosen).

    If you need a good solution, you can restart this algorithm multiple times
    with random initialization and different random seeds
    and use a coverage metric to quantify how well
    the space is covered. Some metrics are described in (2). In contrast to the
    code provided with (2) and (3) we do not consider the feature importance for the
    selection as this is typically not known beforehand.

    You might want to standardize your data before applying this sampling function.

    Some more sampling options are provided in our structure_comp (4) Python package.
    Also, this implementation here is quite memory hungry.

    References:
    (1) Kennard, R. W.; Stone, L. A. Computer Aided Design of Experiments.
    Technometrics 1969, 11 (1), 137–148. https://doi.org/10.1080/00401706.1969.10490666.
    (2) Moosavi, S. M.; Nandy, A.; Jablonka, K. M.; Ongari, D.; Janet, J. P.; Boyd, P. G.;
    Lee, Y.; Smit, B.; Kulik, H. J. Understanding the Diversity of the Metal-Organic Framework Ecosystem.
    Nature Communications 2020, 11 (1), 4068. https://doi.org/10.1038/s41467-020-17755-8.
    (3) Moosavi, S. M.; Chidambaram, A.; Talirz, L.; Haranczyk, M.; Stylianou, K. C.; Smit, B.
    Capturing Chemical Intuition in Synthesis of Metal-Organic Frameworks. Nat Commun 2019, 10 (1), 539.
    https://doi.org/10.1038/s41467-019-08483-9.
    (4) https://github.com/kjappelbaum/structure_comp

    Args:
        X (np.array): Feature array, this is the array that is used to perform the sampling

        y (np.array): Target array, is not used for sampling and only as utility to provide
        the correct labels for the sampled feature rows.

        n_samples (int): number of points that will be selected, needs to be lower than the length of X

        metric (str, optional): Distance metric to use for the maxmin calculation.
            Must be a valid option of scipy.spatial.distance.cdist (‘braycurtis’, ‘canberra’, ‘chebyshev’,
        ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’,
        ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
        ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’).
        Defaults to 'euclidean'

        init (str, optional): either 'mean', 'median', or 'random'.
            Determines how the initial point is chosen. Defaults to 'center'

        seed (int, optional): seed for the random number generator. Defaults to None.
        **kwargs passed to the cdist

    Returns:
        Tuple[np.array, np.array, list]: Feature array, label array, selected_indices
    """
    np.random.seed(seed)
    assert len(X) == len(y), 'X and y need to have the same length'
    assert len(X) > n_samples, 'The numbers of points that shall be selected (n_samples),\
         needs to be smaller than the length of the feature matrix (X)'

    assert n_samples > 0 and isinstance(
        n_samples, int), 'The number of points that shall be selected (n_samples) needs to be an integer greater than 0'

    greedy_data = []

    if init == 'random':
        index = np.random.randint(0, len(X) - 1)
    elif init == 'mean':
        index = np.argmin(np.linalg.norm(X - np.mean(X, axis=0), axis=1))
    else:
        index = np.argmin(np.linalg.norm(X - np.median(X, axis=0), axis=1))
    greedy_data.append(X[index])
    remaining = np.delete(X, index, 0)
    pbar = tqdm(total=n_samples)
    while len(greedy_data) < n_samples:
        dist = distance.cdist(remaining, greedy_data, metric, **kwargs)
        greedy_index = np.argmax(np.min(dist, axis=0))
        greedy_data.append(remaining[greedy_index])
        remaining = np.delete(remaining, greedy_index, 0)
        pbar.update(1)
    pbar.close()

    greedy_indices = []

    for datum in greedy_data:
        greedy_indices.append(np.array(np.where(np.all(X == datum, axis=1)))[0])

    greedy_indices = np.concatenate(greedy_indices).ravel()

    return X[greedy_indices, :], y[greedy_indices, :], greedy_indices


def is_pareto_efficient(costs, return_mask=True):
    # https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    return is_efficient


@jit(nopython=True)
def dominance_check(point1, point2):
    """One point dominates another if it is not worse in all objectives
    and strictly better in at least one. This here assumes we want to maximize"""
    if np.all(point1 >= point2) and np.any(point1 > point2):
        return True

    return False


@jit(nopython=True)
def dominance_check_jitted(point, array):
    arr_sorted = array[array[:, 0].argsort()]
    for i in range(len(arr_sorted)):
        if dominance_check(point, arr_sorted[i]):
            return True
    return False


@jit(nopython=True)
def dominance_check_jitted_2(array, point):
    arr_sorted = array[array[:, 0].argsort()[::-1]]
    for i in range(len(arr_sorted)):
        if dominance_check(arr_sorted[i], point):
            return True
    return False


vectorized_dominance_check = vectorize(dominance_check)  # pylint:disable=invalid-name
