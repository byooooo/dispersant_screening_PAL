# -*- coding: utf-8 -*-
# Copyright 2020 PyPAL authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint:disable=invalid-name
"""Run PAL multiple times with the same settings"""
import os
import pickle
import time

import click
import GPy
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.data import MinMaxScaler

TARGETS = ['deltaGmin', 'A2_normalized']

FEATURES = [
    'num_[W]', 'max_[W]', 'num_[Tr]', 'max_[Tr]', 'num_[Ta]', 'max_[Ta]', 'num_[R]', 'max_[R]', '[W]', '[Tr]', '[Ta]',
    '[R]', 'rel_shannon', 'length'
]

from pyepal import PALCoregionalized
from pyepal.models.gpr import build_coregionalized_model
from pyepal.pal.utils import (get_hypervolume, get_kmeans_samples,
                              get_maxmin_samples)

TIMESTR = time.strftime('%Y%m%d-%H%M%S') + '-dispersant-'

DATADIR = '../data'


def load_data(n_samples, label_scaling: bool = False, method: str = 'maxmin'):
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

    if label_scaling:
        label_scaler = MinMaxScaler()
        y = label_scaler.fit_transform(y)

    if method == 'maxmin':
        greedy_indices = get_maxmin_samples(X, n_samples)

    elif method == 'kmeans':
        greedy_indices = get_kmeans_samples(X, n_samples)

    return X, y, greedy_indices


@click.command('cli')
@click.argument('epsilon', type=float, default=0.05, required=False)
@click.argument('delta', type=float, default=0.05, required=False)
@click.argument('beta_scale', type=float, default=1 / 20, required=False)
@click.argument('repeats', type=int, default=1, required=False)
@click.argument('outdir', type=click.Path(), default='.', required=False)
@click.argument('n_samples', type=int, default=60, required=False)
@click.argument('pooling', type=str, default='fro', required=False)
@click.argument('w_rank', type=int, default=1, required=False)
@click.argument('sampling_method', type=str, default='maxmin', required=False)
@click.option('--sample_discarded', is_flag=True)
def main(  # pylint:disable=invalid-name, too-many-arguments, too-many-locals
        epsilon, delta, beta_scale, repeats, outdir, n_samples, pooling, w_rank, sampling_method, sample_discarded):
    """Main CLI"""
    pareto_optimals = []
    hypervolumess = []
    selecteds = []
    discardeds = []
    unclassifieds = []
    print('START ePAL')
    for _ in range(repeats):
        X, y, greedy_indices = load_data(n_samples, method=sampling_method)

        max_hypervolume = get_hypervolume(y, [5, 5, 5])
        print(f'Maximum hypervolume {max_hypervolume}')
        m = [
            build_coregionalized_model(X, y, kernel=GPy.kern.Matern52(X.shape[1], ARD=False), w_rank=w_rank),
        ]

        palinstance = PALCoregionalized(X, m, 3, epsilon=np.array([epsilon] * 3), beta_scale=beta_scale, delta=delta)
        palinstance.update_train_set(greedy_indices, y[greedy_indices])
        print('STARTING PAL')
        while sum(palinstance.unclassified):
            idx = palinstance.run_one_step(pooling_method=pooling, sample_discarded=sample_discarded)
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

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    epsilon = str(epsilon)
    delta = str(delta)
    beta_scale = str(beta_scale)
    np.save(
        os.path.join(
            outdir, TIMESTR +
            f'{epsilon}_{delta}_{beta_scale}_{n_samples}_{sample_discarded}_{pooling}_{w_rank}_{sampling_method}-greedy_indices'
        ), greedy_indices)
    np.save(
        os.path.join(
            outdir, TIMESTR +
            f'{epsilon}_{delta}_{beta_scale}_{n_samples}_{sample_discarded}_{pooling}_{w_rank}_{sampling_method}-pareto_optimal_indices'
        ), pareto_optimals)
    np.save(
        os.path.join(
            outdir, TIMESTR +
            f'{epsilon}_{delta}_{beta_scale}_{n_samples}_{sample_discarded}_{pooling}_{w_rank}_{sampling_method}-hypervolumes'
        ), hypervolumess)
    np.save(
        os.path.join(
            outdir, TIMESTR +
            f'{epsilon}_{delta}_{beta_scale}_{n_samples}_{sample_discarded}_{pooling}_{w_rank}_{sampling_method}-selected'
        ), selecteds)
    np.save(
        os.path.join(
            outdir, TIMESTR +
            f'{epsilon}_{delta}_{beta_scale}_{n_samples}_{sample_discarded}_{pooling}_{w_rank}_{sampling_method}-discarded'
        ), discardeds)
    np.save(
        os.path.join(
            outdir, TIMESTR +
            f'{epsilon}_{delta}_{beta_scale}_{n_samples}_{sample_discarded}_{pooling}_{w_rank}_{sampling_method}-unclassified'
        ), unclassifieds)

    with open(
            os.path.join(
                outdir, TIMESTR +
                f'{epsilon}_{delta}_{beta_scale}_{n_samples}_{sample_discarded}_{pooling}_{w_rank}_{sampling_method}-models.pkl'
            ), 'wb') as fh:
        pickle.dump(palinstance.models, fh)


if __name__ == '__main__':
    main()  # pylint:disable=no-value-for-parameter
