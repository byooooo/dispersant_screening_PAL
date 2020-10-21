# -*- coding: utf-8 -*-
# pylint:disable=invalid-name
import os
import time

import click
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

import GPy
from dispersant_screener.definitions import FEATURES
from pypal import PALCoregionalized
from pypal.models.gpr import build_coregionalized_model
from pypal.pal.utils import get_maxmin_samples

TIMESTR = time.strftime('%Y%m%d-%H%M%S')

DATADIR = '../data'


def load_data(n_samples, label_scaling: bool = False):
    """Take in Brian's data and spit out some numpy arrays for the PAL"""
    df_full_factorial_feat = pd.read_csv(os.path.join(DATADIR, 'new_features_full_random.csv'))[FEATURES].values
    a2 = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_new.csv'))['A2_normalized'].values
    deltaGMax = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_new.csv'))['A2_normalized'].values  # pylint:disable=unused-variable
    gibbs = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_deltaG.csv'))['deltaGmin'].values * (-1)
    gibbs_max = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_new.csv'))['deltaGmax'].values
    force_max = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_fit2.csv'))['F_repel_max'].values  # pylint:disable=unused-variable
    rg = pd.read_csv(os.path.join(DATADIR, 'rg_results.csv'))['Rg'].values
    y = np.hstack([rg.reshape(-1, 1), gibbs.reshape(-1, 1), gibbs_max.reshape(-1, 1)])
    assert len(df_full_factorial_feat) == len(a2) == len(gibbs) == len(y)

    vt = VarianceThreshold()
    X = vt.fit_transform(df_full_factorial_feat)

    feat_scaler = StandardScaler()
    X = feat_scaler.fit_transform(X)

    if label_scaling:
        label_scaler = StandardScaler()
        y = label_scaler.fit_transform(y)

    _, _, greedy_indices = get_maxmin_samples(X, y, n_samples)

    return X, y, greedy_indices


@click.command('cli')
@click.argument('epsilon', type=float, default=0.05, required=False)
@click.argument('delta', type=float, default=0.05, required=False)
@click.argument('beta_scale', type=float, default=1 / 9, required=False)
@click.argument('hv_reference', type=float, default=5, required=False)
@click.argument('outdir', type=click.Path(), default='.', required=False)
def main(epsilon, delta, beta_scale, hv_reference, outdir):
    X, y, greedy_indices = load_data(n_samples)

    m = [build_coregionalized_model(X, y, kernel=GPy.kern.Matern52(X.shape[1], ARD=False))]
    palinstance = PALCoregionalized(X, m, 3, epsilon=np.array([epsilon] * 3), beta_scale=beta_scale, delta=delta)
    palinstance.update_train_set(greedy_indices, y[greedy_indices])
    print('STARTING PAL')

    while sum(palinstance.unclassified):
        idx = palinstance.run_one_step()
        pareto_optimal.append(palinstance.pareto_optimal_indices)
        selected.append(palinstance.sampled_indices)
        hv = get_hypervolume(y[palinstance.pareto_optimal_indices], [5, 5, 5])
        print(palinstance)
        print(f'Hypervolume {hv}')
        if idx is not None:
            palinstance.update_train_set(np.array([idx]), y[idx:idx + 1, :])

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    np.save(os.path.join(outdir, TIMESTR + '-X_train'), X_train)
    np.save(os.path.join(outdir, TIMESTR + '-X_test'), X_test)
    np.save(os.path.join(outdir, TIMESTR + '-y_train'), y_train)
    np.save(os.path.join(outdir, TIMESTR + '-y_test'), y_test)
    np.save(os.path.join(outdir, TIMESTR + '-pareto_optimal_indices'), pareto_optimal)
    np.save(os.path.join(outdir, TIMESTR + '-hypervolumes'), hypervolumes)
    np.save(os.path.join(outdir, TIMESTR + '-selected'), selected)
    joblib.dump(gps, os.path.join(outdir, TIMESTR + 'models.joblib'))


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
