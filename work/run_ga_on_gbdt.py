# -*- coding: utf-8 -*-
import os
import time
from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import click
import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from dispersant_screener.ga import FEATURES, predict_gbdt, run_ga

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

df_full_factorial_feat = pd.read_csv(os.path.join(DATADIR, 'new_features_full_random.csv'))[FEATURES].values
a2 = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_new.csv'))['A2_normalized'].values
deltaGMax = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_new.csv'))['A2_normalized'].values
gibbs = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_deltaG.csv'))['deltaGmin'].values
gibbs_max = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_new.csv'))['deltaGmax'].values
force_max = pd.read_csv(os.path.join(DATADIR, 'b1-b21_random_virial_large_fit2.csv'))['F_repel_max'].values
rg = pd.read_csv(os.path.join(DATADIR, 'rg_results.csv'))['Rg'].values
y = np.hstack([
    rg.reshape(-1, 1),
    gibbs.reshape(-1, 1) * (-1),
    gibbs_max.reshape(-1, 1),
])
assert len(df_full_factorial_feat) == len(a2) == len(gibbs) == len(y)


@click.command('cli')
@click.argument('target', type=int, default=0)
@click.argument('runs', type=int, default=10)
@click.argument('outdir', type=click.Path(), default='.')
@click.option('--all', is_flag=True)
def main(target, runs, outdir, all):
    if all:
        targets = [0, 1, 2]
    else:
        targets = [target]

    for target in targets:
        y_selected = y[:, target]
        X_train, X_test, y_train, y_test = train_test_split(df_full_factorial_feat, y_selected, train_size=0.8)
        if target == 0:
            lgbm = LGBMRegressor(**config_0)
        elif target == 1:
            lgbm = LGBMRegressor(**config_1)
        elif target == 2:
            lgbm = LGBMRegressor(**config_2)

        lgbm.fit(X_train, y_train)

        predict_partial = partial(predict_gbdt, model=lgbm)

        gas = []

        for _ in range(runs):
            gas.append(run_ga(predict_partial, df_full_factorial_feat))

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    joblib.dump(gas, os.path.join(outdir, TIMESTR + '-ga_{}.joblib'.format(target)))


if __name__ == '__main__':
    main()
