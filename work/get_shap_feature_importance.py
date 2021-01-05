# -*- coding: utf-8 -*-
"""
RUN SHAP feature importance analysis
"""
import os
import pickle
from functools import partial

import click
import numpy as np
import pandas as pd
import shap
from shap import KernelExplainer
from sklearn.preprocessing import StandardScaler

DATADIR = '../data'
from pypal.models.gpr import predict_coregionalized

from dispersant_screener.definitions import FEATURES


def load_data():
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

    feat_scaler = StandardScaler()
    X = feat_scaler.fit_transform(df_full_factorial_feat)

    #greedy_indices = get_maxmin_samples(X, n_samples)

    return X, y  #, greedy_indice


X, y = load_data()


@click.command('cli')
@click.argument('model')
@click.argument('i')
def main(model, i):

    with open(f'{model}-models.pkl', 'rb') as fh:
        models = pickle.load(fh)

    sampled = np.load(f'{model}-selected.npy', allow_pickle=True)

    sampled_indices = sampled[-1]

    def model_wrapper(X, i=0):
        mu, var = predict_coregionalized(models[0], X, int(i))
        return mu.flatten()

    shap0 = KernelExplainer(partial(model_wrapper, i=int(i)), shap.kmeans(X, 40))

    shap0_values = shap0.shap_values(X)

    np.save(f'{model}-shap-{i}', shap0_values)


if __name__ == '__main__':
    main()
