# -*- coding: utf-8 -*-
# pylint:disable=invalid-name
"""Use a wandb sweep to optimize hyperparameters for the GBDT models"""
import os
from functools import partial

import click
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

import wandb
from dispersant_screener.definitions import FEATURES

DATADIR = '/scratch/kjablonk/dispersant_basf'

config = {
    'n_estimators': {
        'distribution': 'int_uniform',
        'min': 10,
        'max': 5000
    },
    'max_depth': {
        'distribution': 'int_uniform',
        'min': 5,
        'max': 100
    },
    'num_leaves': {
        'distribution': 'int_uniform',
        'min': 5,
        'max': 500
    },
    'reg_alpha': {
        'distribution': 'log_uniform',
        'min': 0.00001,
        'max': 0.4
    },
    'reg_lambda': {
        'distribution': 'log_uniform',
        'min': 0.00001,
        'max': 0.4
    },
    'subsample': {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 1.0
    },
    'colsample_bytree': {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 1.0
    },
    'min_child_weight': {
        'distribution': 'uniform',
        'min': 0.001,
        'max': 0.1,
    },
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
    gibbs.reshape(-1, 1),
    gibbs_max.reshape(-1, 1),
    force_max.reshape(-1, 1),
    deltaGMax.reshape(-1, 1),
    a2.reshape(-1, 1)
])
assert len(df_full_factorial_feat) == len(a2) == len(gibbs) == len(y)

X_train, X_test, y_train, y_test = train_test_split(df_full_factorial_feat, y, train_size=0.8)

vt = VarianceThreshold(0)
X_train = vt.fit_transform(X_train)
X_test = vt.transform(X_test)

feat_scaler = StandardScaler()
X_train = feat_scaler.fit_transform(X_train)
X_test = feat_scaler.transform(X_test)


def get_sweep_id(method):
    """return us a sweep id (required for running the sweep)"""
    sweep_config = {
        'method': method,
        'metric': {
            'name': 'cv_mean',
            'goal': 'minimize'
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 3,
            'max_iter': 30
        },
        'parameters': config,
    }
    sweep_id = wandb.sweep(sweep_config, project='dispersant_screener')

    return sweep_id


def train(index):
    # Config is a variable that holds and saves hyperparameters and inputs

    configs = {
        'n_estimators': 100,
        'max_depth': 10,
        'num_leaves': 50,
        'reg_alpha': 0.00001,
        'reg_lambda': 0.00001,
        'subsample': 0.2,
        'colsample_bytree': 0.2,
        'min_child_weight': 0.001,
    }

    # Initilize a new wandb run
    wandb.init(project='dispersant_screener', config=configs)

    config = wandb.config

    regressor = LGBMRegressor(**config)

    cv = cross_val_score(regressor, X_train, y_train[:, index], n_jobs=-1)

    mean = np.abs(cv.mean())
    std = np.abs(cv.std())
    wandb.log({'cv_mean': mean})
    wandb.log({'cv_std': std})

    wandb.run.summary['cv_mean'] = mean
    wandb.run.summary['cv_std'] = std


@click.command('cli')
@click.argument('index')
def main(index):
    sweep_id = get_sweep_id('bayes')

    train_func = partial(train, index=int(index))
    wandb.agent(sweep_id, function=train_func)


if __name__ == '__main__':
    main()
