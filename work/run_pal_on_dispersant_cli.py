# -*- coding: utf-8 -*-
import os
import time

import GPy
import click
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from dispersant_screener.definitions import FEATURES
from dispersant_screener.gp import build_coregionalized_model, build_model
from dispersant_screener.pal import pal
from dispersant_screener.utils import get_kmeans_samples, get_maxmin_samples

TIMESTR = time.strftime('%Y%m%d-%H%M%S')

DATADIR = '../data'


def load_data(label_scaling: bool = False):
    df_full_factorial_feat = pd.read_csv(os.path.join(DATADIR, 'new_features_full_random.csv'))[FEATURES].values
    a2 = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_new.csv'))['A2_normalized'].values
    deltaGMax = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_new.csv'))['A2_normalized'].values
    gibbs = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_deltaG.csv'))['deltaGmin'].values * (-1)
    gibbs_max = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_new.csv'))['deltaGmax'].values
    force_max = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_fit2.csv'))['F_repel_max'].values
    rg = pd.read_csv(os.path.join(DATADIR, 'rg_results.csv'))['Rg'].values
    y = np.hstack([rg.reshape(-1, 1), gibbs.reshape(-1, 1), gibbs_max.reshape(-1, 1)])
    assert len(df_full_factorial_feat) == len(a2) == len(gibbs) == len(y)

    vt = VarianceThreshold(0.2)
    X = vt.fit_transform(df_full_factorial_feat)

    feat_scaler = StandardScaler()
    X = feat_scaler.fit_transform(X)

    if label_scaling:
        label_scaler = StandardScaler()
        y = label_scaler.fit_transform(y)

    X_train, y_train, greedy_indices = get_kmeans_samples(X, y, 30)

    y_test = np.delete(y, greedy_indices, 0)
    X_test = np.delete(X, greedy_indices, 0)

    return X_train, y_train, X_test, y_test


@click.command('cli')
@click.argument('epsilon', type=float, default=0.05, required=False)
@click.argument('delta', type=float, default=0.05, required=False)
@click.argument('beta_scale', type=float, default=1 / 20, required=False)
@click.argument('optimize_always', type=int, default=1, required=False)
@click.argument('optimize_delay', type=int, default=60, required=False)
@click.argument('hv_reference', type=float, default=5, required=False)
@click.argument('outdir', type=click.Path(), default='.', required=False)
def main(epsilon, delta, beta_scale, optimize_always, optimize_delay, hv_reference, outdir):
    X_train, y_train, X_test, y_test = load_data()

    m = build_coregionalized_model(X_train, y_train, kernel=GPy.kern.RBF(X_train.shape[1], ARD=False))

    pareto_optimal, hypervolumes, gps, selected = pal(
        [m],
        X_train,
        y_train,
        X_test,
        y_test,
        hv_reference=[hv_reference] * y_train.shape[1],
        epsilon=[epsilon] * y_train.shape[1],
        #verbosity='debug',
        delta=delta,
        beta_scale=beta_scale,
        coregionalized=True,
        optimize_always=optimize_always,
        optimize_delay=optimize_delay,
        history_dump_file=os.path.join(outdir, TIMESTR + '-history.pkl'))

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    np.save(os.path.join(outdir, TIMESTR + '-X_train'), X_train)
    np.save(os.path.join(outdir, TIMESTR + '-X_test'), X_test)
    np.save(os.path.join(outdir, TIMESTR + '-y_train'), y_train)
    np.save(os.path.join(outdir, TIMESTR + '-y_test'), y_test)
    np.save(os.path.join(outdir, TIMESTR + '-pareto_optimal_indices'), pareto_optimal)
    np.save(os.path.join(outdir, TIMESTR + '-hypervolumes'), pareto_optimal)
    np.save(os.path.join(outdir, TIMESTR + '-selected'), selected)
    joblib.dump(gps, os.path.join(outdir, TIMESTR + 'models.joblib'))


if __name__ == '__main__':
    main()
