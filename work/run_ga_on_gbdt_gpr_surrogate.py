# -*- coding: utf-8 -*-
"""running the GA on the GBDT models that are trained to the predictions of the GPR models of the epislon-PAL loop"""
import os
import time
from functools import partial

import click
import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from dispersant_screener.ga import (FEATURES, _get_average_dist, predict_gbdt, regularizer_novelty, run_ga)

TIMESTR = time.strftime('%Y%m%d-%H%M%S')
DATADIR = '../data'

# Rg
config_0 = {  # pylint:disable=invalid-name
    'max_depth': 15,
    'reg_alpha': 1.365,
    'subsample': 0.7084,
    'num_leaves': 34,
    'reg_lambda': 1.028,
    'n_estimators': 376,
    'colsample_bytree': 0.9487,
    'min_child_weight': 0.06395
}

# Delta G
# https://app.wandb.ai/kjappelbaum/dispersant_screener/runs/wog4qfb2/overview?workspace=user-kjappelbaum
config_1 = {  # pylint:disable=invalid-name
    'max_depth': 17,
    'reg_alpha': 1.402,
    'subsample': 0.2033,
    'num_leaves': 38,
    'reg_lambda': 1.091,
    'n_estimators': 4676,
    'colsample_bytree': 0.9934,
    'min_child_weight': 0.005694
}

# repulsion
# https://app.wandb.ai/kjappelbaum/dispersant_screener/runs/ljzi9uad/overview?workspace=user-kjappelbaum
config_2 = {  # pylint:disable=invalid-name
    'max_depth': 73,
    'reg_alpha': 1.266,
    'subsample': 0.9824,
    'num_leaves': 398,
    'reg_lambda': 1.221,
    'n_estimators': 4974,
    'colsample_bytree': 0.9615,
    'min_child_weight': 0.09413
}

FEATURES = ['max_[W]', 'max_[Tr]', 'max_[Ta]', 'max_[R]', '[W]', '[Tr]', '[Ta]', '[R]', 'length']


@click.command('cli')
@click.argument('target', type=int, default=0)
@click.argument('runs', type=int, default=5)
@click.argument('outdir', type=click.Path(), default='.')
@click.option('--all', is_flag=True)
def main(target, runs, outdir, all):
    """CLI"""
    if all:
        targets = [0, 1, 2]
    else:
        targets = [target]

    X = np.load('X_train_GBDT.npy')  # pylint:disable=invalid-name
    y = np.load('y_train_GBDT.npy')  # pylint:disable=invalid-name
    X_train, _, y_t, _ = train_test_split(X, y, train_size=0.9999)
    # FEAT_DICT = dict(zip(FEATURES, range(len(FEATURES))))
    for target in targets:
        y_train = y_t[:, target].reshape(-1, 1)
        print(X_train.shape, y_train.shape)
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

        for novelty_penalty_ratio in [0, 0.1, 0.2, 0.5, 1, 2]:
            for _ in range(runs):

                ga = run_ga(  # pylint:disable=invalid-name
                    predict_partial,
                    regularizer_novelty_partial,
                    features=FEATURES,
                    y_mean=y_train.mean(),
                    novelty_pentaly_ratio=novelty_penalty_ratio)
                gas.append(ga)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    joblib.dump(gas, os.path.join(outdir, TIMESTR + '-ga_{}.joblib'.format(target)))


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
