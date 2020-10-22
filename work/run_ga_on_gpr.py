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
import os
import time
from functools import partial
import pickle
import click
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dispersant_screener.ga import FEATURES, predict_gpy_coregionalized, run_ga, regularizer_novelty, _get_average_dist

TIMESTR = time.strftime('%Y%m%d-%H%M%S')
DATADIR = '../data'

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

with open('sweeps3/20201021-235927_dispersant_0.01_0.05_0.05_60-models.pkl', 'rb') as fh:
    coregionalized_model = pickle.load(fh)[0]

feat_scaler = StandardScaler()
X = feat_scaler.fit_transform(df_full_factorial_feat)


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

        predict_partial = partial(predict_gpy_coregionalized, model=coregionalized_model, i=target)

        regularizer_novelty_partial = partial(regularizer_novelty,
                                              y=y_selected,
                                              X_data=X,
                                              average_dist=_get_average_dist(X))

        gas = []
        for novelty_penalty_ratio in [0, 0.1, 0.2, 0.5, 1, 2]:
            for _ in range(runs):
                gas.append(
                    run_ga(  # pylint:disable=invalid-name
                        predict_partial,
                        regularizer_novelty_partial,
                        features=FEATURES,
                        y_mean=np.median(y_selected),
                        novelty_pentaly_ratio=novelty_penalty_ratio))

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    joblib.dump(gas, os.path.join(outdir, TIMESTR + '-ga_{}.joblib'.format(target)))


if __name__ == '__main__':
    main()
