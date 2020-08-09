# -*- coding: utf-8 -*-
"""Functions that can be handy when using epsilon-PAL"""
from __future__ import absolute_import

from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from sklearn import metrics
from sklearn.cluster import KMeans
from tqdm import tqdm


def plot_parity(y_pred0, y_true0, var0, y_pred1, y_true1, var1, outname=None):
    """Plot a parity plot for two targets"""
    plt.rcParams['font.family'] = 'sans-serif'
    fig, ax = plt.subplots(1, 2)

    ax[0].scatter(y_true0, y_pred0, c=var0, cmap=plt.cm.coolwarm, s=.3)
    ax[1].scatter(y_true1, y_pred1, c=var1, cmap=plt.cm.coolwarm, s=.3)

    for a in ax:
        a.spines['left'].set_smart_bounds(True)
        a.spines['bottom'].set_smart_bounds(True)
        a.set_xlabel(r'$y_{true}$')
        lims = [
            np.min([a.get_xlim(), a.get_ylim()]),  # min of both axes
            np.max([a.get_xlim(), a.get_ylim()]),  # max of both axes
        ]
        a.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    ax[0].set_ylabel(r'$\hat{y}$')
    fig.tight_layout()

    if outname is not None:
        fig.savefig(outname, bbox_inches='tight')

    plt.close(fig)


def get_metrics(y_true: np.array, y_pred: np.array) -> dict:
    """Get a dictionary with basic regression metrics"""
    r2 = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    max_error = metrics.max_error(y_true, y_pred)

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


def add_postfix_to_keys(d: dict, postfix: str) -> dict:
    """Update keys of a dictionary from "key" to "key_postfix"""
    d_ = {}
    postfix = str(postfix)
    for k, v in d.items():
        d_[k + '_' + postfix] = v

    return d_


def get_kmeans_samples(X: np.array, y: np.array, n_samples: int, **kwargs) -> Tuple[np.array, np.array, list]:
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
    assert len(
        X
    ) > n_samples, 'The numbers of points that shall be selected (n_samples), needs to be smaller than the length of the feature matrix (X)'

    assert n_samples > 0 and isinstance(
        n_samples, int), 'The number of points that shall be selected (n_samples) needs to be an integer greater than 0'
    kmeans = KMeans(n_samples, **kwargs).fit(X)
    cluster_centers = kmeans.cluster_centers_
    closest, _ = metrics.pairwise_distances_argmin_min(cluster_centers, X)

    return X[closest, :], y[closest, :], closest


def get_maxmin_samples(X: np.array,
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
    But you might want to standardize your data before applying this sampling function.

    Some more sampling options are provided in our structure_comp (4) Python package.
    Also, this implementation here is quite memory hungry.

    References:
    (1) Kennard, R. W.; Stone, L. A. Computer Aided Design of Experiments.
    Technometrics 1969, 11 (1), 137–148. https://doi.org/10.1080/00401706.1969.10490666.
    (2) Moosavi, S. M.; Nandy, A.; Jablonka, K. M.; Ongari, D.; Janet, J. P.; Boyd, P. G.;
    Lee, Y.; Smit, B.; Kulik, H. Understanding the Diversity of the Metal-Organic Framework Ecosystem;
    preprint; 2020. https://doi.org/10.26434/chemrxiv.12251186.v1.
    (3) Moosavi, S. M.; Chidambaram, A.; Talirz, L.; Haranczyk, M.; Stylianou, K. C.; Smit, B.
    Capturing Chemical Intuition in Synthesis of Metal-Organic Frameworks. Nat Commun 2019, 10 (1), 539.
    https://doi.org/10.1038/s41467-019-08483-9.
    (4) https://github.com/kjappelbaum/structure_comp

    Args:
        X (np.array): Feature array, this is the array that is used to perform the sampling
        y (np.array): Target array, is not used for sampling and only as utility to provide
            the correct labels for the sampled feature rows.
        n_samples (int): number of points that will be selected, needs to be lower
            than the length of X
        metric (str, optional): Distance metric to use for the maxmin calculation. Must be a
            valid option of scipy.spatial.distance.cdist (‘braycurtis’, ‘canberra’, ‘chebyshev’,
            ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’,
            ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
            ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’, ‘yule’).
             Defaults to 'euclidean'
        init (str, optional): either 'mean', 'median', or 'random'. Determines how the initial point is chosen.
            Defaults to 'center'
        seed (int, optional): seed for the random number generator. Defaults to None.
        **kwargs passed to the cdist

    Returns:
        Tuple[np.array, np.array, list]: Feature array, label array, selected_indices
    """
    np.random.seed(seed)
    assert len(X) == len(y), 'X and y need to have the same length'
    assert len(
        X
    ) > n_samples, 'The numbers of points that shall be selected (n_samples), needs to be smaller than the length of the feature matrix (X)'

    assert n_samples > 0 and isinstance(
        n_samples, int), 'The number of points that shall be selected (n_samples) needs to be an integer greater than 0'

    greedy_data = []

    if init == 'random':
        index = np.random.randint(0, len(X) - 1)
    elif init == 'mean':
        index = np.argmin(np.abs(X - np.mean(X, axis=0)))
    else:
        index = np.argmin(np.abs(X - np.median(X, axis=0)))
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

    for d in greedy_data:
        greedy_indices.append(np.array(np.where(np.all(X == d, axis=1)))[0])

    greedy_indices = np.concatenate(greedy_indices).ravel()

    return X[greedy_indices, :], y[greedy_indices, :], greedy_indices
