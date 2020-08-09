# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

import joblib
import numpy as np
import pandas as pd
from six.moves import range
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import wandb
from dispersant_screener.gp import (build_coregionalized_model, build_model, predict, predict_coregionalized)
from dispersant_screener.utils import (add_postfix_to_keys, get_metrics, get_variance_descriptors, plot_parity)

DATADIR = '../data'
TRAIN_SIZES = [0.01, 0.1, 0.3, 0.5, 0.8]
REPEAT = 10

df_full_factorial_feat = pd.read_csv(os.path.join(DATADIR, 'new_features_full_random.csv')).values
a2 = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large.csv'))['A2_normalized'].values
gibbs = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_deltaG.csv'))['deltaGmin'].values
rg = pd.read_csv(os.path.join(DATADIR, 'rg_results.csv'))['Rg'].values
y = np.hstack([rg.reshape(-1, 1), gibbs.reshape(-1, 1)])
assert len(df_full_factorial_feat) == len(a2) == len(gibbs) == len(y)

METRICS_COREGIONALIZED = []
METRICS_UNCORRELATED = []


def get_initial_split(df_full_factorial_feat, y):
    X_train, X_test, y_train, y_test = train_test_split(df_full_factorial_feat, y, train_size=0.8)

    vt = VarianceThreshold(0)
    X_train = vt.fit_transform(X_train)
    X_test = vt.transform(X_test)

    feat_scaler = StandardScaler()
    X_train = feat_scaler.fit_transform(X_train)
    X_test = feat_scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def make_new_split(X, y, train_size):
    return train_test_split(X, y, train_size=train_size)


def main():
    X_train_, y_train_, X_test, y_test = get_initial_split(df_full_factorial_feat, y)

    for train_size in TRAIN_SIZES:
        for i in range(REPEAT):
            X_train, _, y_train, _ = train_test_split(X_train_, y_train_, train_size=train_size)

            # Train coregionalized model
            wandb.init(project='dispersant_screener', tags=['coregionalized', 'matern32'], save_code=True)
            m = build_coregionalized_model(X_train, y_train)
            y0, var0 = predict_coregionalized(m, X_test, 0)
            y1, var1 = predict_coregionalized(m, X_test, 1)
            metrics_0 = get_metrics(y0, y_test[:, 0])
            metrics_0 = add_postfix_to_keys(metrics_0, 0)

            metrics_1 = get_metrics(y1, y_test[:, 1])
            metrics_1 = add_postfix_to_keys(metrics_0, 1)

            variance_0 = get_variance_descriptors(var0)
            variance_1 = get_variance_descriptors(var1)
            variance_0 = add_postfix_to_keys(variance_0, 0)
            variance_1 = add_postfix_to_keys(variance_1, 1)

            overall_metrics = metrics_0
            overall_metrics.update(metrics_1)
            overall_metrics.update(variance_0)
            overall_metrics.update(variance_1)
            overall_metrics['train_size'] = len(X_train)
            overall_metrics['coregionalized'] = True

            plot_parity(y0, y_test[:, 0], var0, y1, y_test[:, 1], var1,
                        'coregionalized_{}_{}.pdf'.format(len(X_train), i))
            wandb.log(overall_metrics)

            # Train "simple models"
            wandb.init(project='dispersant_screener', tags=['matern32'], save_code=True)
            m0 = build_model(X_train, y_train, 0)
            m1 = build_model(X_train, y_train, 1)

            y0, var0 = predict(m0, X_test)
            y1, var1 = predict(m1, X_test)
            metrics_0 = get_metrics(y0, y_test[:, 0])
            metrics_0 = add_postfix_to_keys(metrics_0, 0)

            metrics_1 = get_metrics(y1, y_test[:, 1])
            metrics_1 = add_postfix_to_keys(metrics_0, 1)

            variance_0 = get_variance_descriptors(var0)
            variance_1 = get_variance_descriptors(var1)
            variance_0 = add_postfix_to_keys(variance_0, 0)
            variance_1 = add_postfix_to_keys(variance_1, 1)

            overall_metrics = metrics_0
            overall_metrics.update(metrics_1)
            overall_metrics.update(variance_0)
            overall_metrics.update(variance_1)
            overall_metrics['train_size'] = len(X_train)
            overall_metrics['coregionalized'] = False

            plot_parity(y0, y_test[:, 0], var0, y1, y_test[:, 1], var1, 'simple_{}_{}.pdf'.format(len(X_train), i))

            wandb.log(overall_metrics)


if __name__ == '__main__':
    main()
