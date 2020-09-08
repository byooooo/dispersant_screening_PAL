# -*- coding: utf-8 -*-
# pylint:disable=invalid-name
"""Use a wandb sweep to optimize hyperparameters for the GBDT models"""
import os
from functools import partial

import click
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

import wandb

X = np.load('X_train_GBDT.npy')  # pylint:disable=invalid-name
y = np.load('y_train_GBDT.npy')  # pylint:disable=invalid-name

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

# feat_scaler = StandardScaler()
# X_train = feat_scaler.fit_transform(X_train)


def get_sweep_id(method):
    """return us a sweep id (required for running the sweep)"""
    sweep_config = {
        'method': method,
        'metric': {
            'name': 'cv_mean',
            'goal': 'maximize'
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
    scores = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y[:, index], train_size=.8)
        regressor.fit(X_train, y_train)
        scores.append(r2_score(regressor.predict(X_test), y_test))

    mean = np.mean(scores)
    std = np.std(scores)

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
