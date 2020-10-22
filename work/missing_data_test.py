# -*- coding: utf-8 -*-
# pylint:disable=invalid-name
"""Run PAL multiple times with the same settings"""
import os
import pickle
import time

import click
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import MinMaxScaler

import GPy
from dispersant_screener.definitions import FEATURES
from pypal import PALCoregionalized, PALGPy
from pypal.models.gpr import build_coregionalized_model
from pypal.pal.utils import get_hypervolume, get_maxmin_samples

TIMESTR = time.strftime('%Y%m%d-%H%M%S') + '_missing_data_'

DATADIR = '../data'


def load_data(n_samples, label_scaling: bool = False):
    """Take in Brian's data and spit out some numpy arrays for the PAL"""
    df_full_factorial_feat = pd.read_csv(os.path.join(DATADIR, 'new_features_full_random.csv'))[FEATURES].values
    a2 = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_new.csv'))['A2_normalized'].values
    gibbs = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_deltaG.csv'))['deltaGmin'].values * (-1)
    gibbs_max = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_new.csv'))['deltaGmax'].values
    rg = pd.read_csv(os.path.join(DATADIR, 'rg_results.csv'))['Rg'].values
    y = np.hstack([rg.reshape(-1, 1), gibbs.reshape(-1, 1), gibbs_max.reshape(-1, 1)])
    assert len(df_full_factorial_feat) == len(a2) == len(gibbs) == len(y)

    feat_scaler = StandardScaler()
    X = feat_scaler.fit_transform(df_full_factorial_feat)

    if label_scaling:
        label_scaler = MinMaxScaler()
        y = label_scaler.fit_transform(y)

    greedy_indices = get_maxmin_samples(X, n_samples)

    nan_indices = np.unique(np.random.randint(0, len(y) - 1, int(len(y) / 3)))
    y[nan_indices, 2] = np.nan
    return X, y, greedy_indices


@click.command('cli')
@click.argument('epsilon', type=float, default=0.05, required=False)
@click.argument('delta', type=float, default=0.05, required=False)
@click.argument('beta_scale', type=float, default=1 / 9, required=False)
@click.argument('repeats', type=int, default=1, required=False)
@click.argument('outdir', type=click.Path(), default='.', required=False)
@click.argument('n_samples', type=int, default=60, required=False)
def main(  # pylint:disable=invalid-name, too-many-arguments, too-many-locals
        epsilon, delta, beta_scale, repeats, outdir, n_samples):
    """Main CLI"""
    pareto_optimals = []
    hypervolumess = []
    selecteds = []
    discardeds = []
    unclassifieds = []
    print('START ePAL')
    for _ in range(repeats):
        X, y, greedy_indices = load_data(n_samples)

        max_hypervolume = get_hypervolume(y, [5, 5, 5])
        print(f'Maximum hypervolume {max_hypervolume}')
        m = [
            build_coregionalized_model(X, y, kernel=GPy.kern.Matern52(X.shape[1], ARD=False)),
        ]

        palinstance = PALCoregionalized(X, m, 3, epsilon=np.array([epsilon] * 3), beta_scale=beta_scale, delta=delta)
        palinstance.update_train_set(greedy_indices, y[greedy_indices])
        print('STARTING PAL')
        while sum(palinstance.unclassified):
            idx = palinstance.run_one_step()
            pareto_optimals.append(palinstance.pareto_optimal_indices)
            selecteds.append(palinstance.sampled_indices)
            unclassifieds.append(palinstance.unclassified_indices)
            discardeds.append(palinstance.discarded_indices)
            hv = get_hypervolume(y[palinstance.pareto_optimal_indices], [5, 5, 5])
            hypervolumess.append(hv)
            print(palinstance)
            print(f'Hypervolume {hv}')
            if idx is not None:
                palinstance.update_train_set(np.array([idx]), y[idx])
            else:
                break
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    epsilon = str(epsilon)
    delta = str(delta)
    beta_scale = str(beta_scale)
    np.save(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-y'.format(epsilon, delta, beta_scale, n_samples)),
            y)
    np.save(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-greedy_indices'.format(epsilon, delta, beta_scale, n_samples)),
            greedy_indices)
    np.save(
        os.path.join(outdir,
                     TIMESTR + '{}_{}_{}_{}-pareto_optimal_indices'.format(epsilon, delta, beta_scale, n_samples)),
        pareto_optimals)
    np.save(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-hypervolumes'.format(epsilon, delta, beta_scale, n_samples)),
            hypervolumess)
    np.save(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-selected'.format(epsilon, delta, beta_scale, n_samples)),
            selecteds)
    np.save(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-discarded'.format(epsilon, delta, beta_scale, n_samples)),
            discardeds)
    np.save(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-unclassified'.format(epsilon, delta, beta_scale, n_samples)),
            unclassifieds)

    with open(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-models.pkl'.format(epsilon, delta, beta_scale, n_samples)),
              'wb') as fh:
        pickle.dump(palinstance.models, fh)


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
