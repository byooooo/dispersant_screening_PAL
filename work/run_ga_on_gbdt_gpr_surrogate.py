# -*- coding: utf-8 -*-
import os
import time
from functools import partial

import click
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dispersant_screener.ga import (FEATURES, _get_average_dist, predict_gbdt, regularizer_novelty, run_ga)

TIMESTR = time.strftime('%Y%m%d-%H%M%S')
DATADIR = '../data'

# Rg
config_0 = {
    'max_depth': 81,
    'reg_alpha': 1.0059223601214005,
    'subsample': 0.4323683292622177,
    'num_leaves': 14,
    'reg_lambda': 1.3804842496500522,
    'n_estimators': 3771,
    'colsample_bytree': 0.9897619355459844,
    'min_child_weight': 0.036693782744867405
}

# Delta G
# https://app.wandb.ai/kjappelbaum/dispersant_screener/runs/wog4qfb2/overview?workspace=user-kjappelbaum
config_1 = {
    'max_depth': 73,
    'reg_alpha': 1.392732983015451,
    'subsample': 0.5009306968568509,
    'num_leaves': 6,
    'reg_lambda': 1.0595847294980203,
    'n_estimators': 461,
    'colsample_bytree': 0.966043658485258,
    'min_child_weight': 0.0039362945584385705
}

# repulsion
# https://app.wandb.ai/kjappelbaum/dispersant_screener/runs/ljzi9uad/overview?workspace=user-kjappelbaum
config_2 = {
    'max_depth': 22,
    'reg_alpha': 1.4445428983500173,
    'subsample': 0.37540621157955995,
    'num_leaves': 11,
    'reg_lambda': 1.246760700982355,
    'n_estimators': 56,
    'colsample_bytree': 0.9850898928749316,
    'min_child_weight': 0.05716405492260722
}

FEATURES = ['max_[W]', 'max_[Tr]', 'max_[Ta]', 'max_[R]', '[W]', '[Tr]', '[Ta]', '[R]', 'length']


@click.command('cli')
@click.argument('target', type=int, default=0)
@click.argument('runs', type=int, default=5)
@click.argument('outdir', type=click.Path(), default='.')
@click.option('--all', is_flag=True)
def main(target, runs, outdir, all):
    if all:
        targets = [0, 1, 2]
    else:
        targets = [target]

    X_train = np.load('X_train_GBDT.npy')
    y = np.load('y_train_GBDT.npy')

    FEAT_DICT = dict(zip(FEATURES, range(len(FEATURES))))

    for target in targets:
        y_train = y[:, target]

        if target == 0:
            lgbm = LGBMRegressor(**config_0)
        elif target == 1:
            lgbm = LGBMRegressor(**config_1)
        elif target == 2:
            lgbm = LGBMRegressor(**config_2)

        lgbm.fit(X_train, y_train)

        regularizer_novelty_partial = partial(regularizer_novelty,
                                              y=y_train,
                                              X_data=X_train,
                                              average_dist=_get_average_dist(X_train))
        predict_partial = partial(predict_gbdt, model=lgbm)

        gas = []

        for novelty_penalty_ratio in [0, 0.2, 0.5, 0.8, 1.0]:
            for _ in range(runs):
                
                ga = run_ga(predict_partial,
                           regularizer_novelty_partial,
                           features=FEATURES,
                           novelty_pentaly_ratio=novelty_penalty_ratio)
                gas.append(ga)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    joblib.dump(gas, os.path.join(outdir, TIMESTR + '-ga_{}.joblib'.format(target)))


if __name__ == '__main__':
    main()
