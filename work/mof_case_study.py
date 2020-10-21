# -*- coding: utf-8 -*-
"""Running EPSILON-PAL on a MOF dataset"""
import os
import pickle
import time

import click
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import GPy
from dispersant_screener.utils import get_maxmin_samples
from pypal import PALCoregionalized
from pypal.models.gpr import build_coregionalized_model
from pypal.pal.utils import get_hypervolume

N_SAMPLES = 300
TIMESTR = time.strftime('%Y%m%d-%H%M%S') + '_MOF_'

other_descriptors = ['CellV [A^3]']  # pylint:disable=invalid-name

geometric_descirptors = [  # pylint:disable=invalid-name
    'Di', 'Df', 'Dif', 'density [g/cm^3]', 'total_SA_volumetric', 'total_SA_gravimetric', 'total_POV_volumetric',
    'total_POV_gravimetric'
]

linker_descriptors = [  # pylint:disable=invalid-name
    'f-lig-chi-0', 'f-lig-chi-1', 'f-lig-chi-2', 'f-lig-chi-3', 'f-lig-Z-0', 'f-lig-Z-1', 'f-lig-Z-2', 'f-lig-Z-3',
    'f-lig-I-0', 'f-lig-I-1', 'f-lig-I-2', 'f-lig-I-3', 'f-lig-T-0', 'f-lig-T-1', 'f-lig-T-2', 'f-lig-T-3', 'f-lig-S-0',
    'f-lig-S-1', 'f-lig-S-2', 'f-lig-S-3', 'lc-chi-0-all', 'lc-chi-1-all', 'lc-chi-2-all', 'lc-chi-3-all', 'lc-Z-0-all',
    'lc-Z-1-all', 'lc-Z-2-all', 'lc-Z-3-all', 'lc-I-0-all', 'lc-I-1-all', 'lc-I-2-all', 'lc-I-3-all', 'lc-T-0-all',
    'lc-T-1-all', 'lc-T-2-all', 'lc-T-3-all', 'lc-S-0-all', 'lc-S-1-all', 'lc-S-2-all', 'lc-S-3-all', 'lc-alpha-0-all',
    'lc-alpha-1-all', 'lc-alpha-2-all', 'lc-alpha-3-all', 'D_lc-chi-0-all', 'D_lc-chi-1-all', 'D_lc-chi-2-all',
    'D_lc-chi-3-all', 'D_lc-Z-0-all', 'D_lc-Z-1-all', 'D_lc-Z-2-all', 'D_lc-Z-3-all', 'D_lc-I-0-all', 'D_lc-I-1-all',
    'D_lc-I-2-all', 'D_lc-I-3-all', 'D_lc-T-0-all', 'D_lc-T-1-all', 'D_lc-T-2-all', 'D_lc-T-3-all', 'D_lc-S-0-all',
    'D_lc-S-1-all', 'D_lc-S-2-all', 'D_lc-S-3-all', 'D_lc-alpha-0-all', 'D_lc-alpha-1-all', 'D_lc-alpha-2-all',
    'D_lc-alpha-3-all'
]

metalcenter_descriptors = [  # pylint:disable=invalid-name
    'mc_CRY-chi-0-all', 'mc_CRY-chi-1-all', 'mc_CRY-chi-2-all', 'mc_CRY-chi-3-all', 'mc_CRY-Z-0-all', 'mc_CRY-Z-1-all',
    'mc_CRY-Z-2-all', 'mc_CRY-Z-3-all', 'mc_CRY-I-0-all', 'mc_CRY-I-1-all', 'mc_CRY-I-2-all', 'mc_CRY-I-3-all',
    'mc_CRY-T-0-all', 'mc_CRY-T-1-all', 'mc_CRY-T-2-all', 'mc_CRY-T-3-all', 'mc_CRY-S-0-all', 'mc_CRY-S-1-all',
    'mc_CRY-S-2-all', 'mc_CRY-S-3-all', 'D_mc_CRY-chi-0-all', 'D_mc_CRY-chi-1-all', 'D_mc_CRY-chi-2-all',
    'D_mc_CRY-chi-3-all', 'D_mc_CRY-Z-0-all', 'D_mc_CRY-Z-1-all', 'D_mc_CRY-Z-2-all', 'D_mc_CRY-Z-3-all',
    'D_mc_CRY-I-0-all', 'D_mc_CRY-I-1-all', 'D_mc_CRY-I-2-all', 'D_mc_CRY-I-3-all', 'D_mc_CRY-T-0-all',
    'D_mc_CRY-T-1-all', 'D_mc_CRY-T-2-all', 'D_mc_CRY-T-3-all', 'D_mc_CRY-S-0-all', 'D_mc_CRY-S-1-all',
    'D_mc_CRY-S-2-all', 'D_mc_CRY-S-3-all'
]

functionalgroup_descriptors = [  # pylint:disable=invalid-name
    'func-chi-0-all', 'func-chi-1-all', 'func-chi-2-all', 'func-chi-3-all', 'func-Z-0-all', 'func-Z-1-all',
    'func-Z-2-all', 'func-Z-3-all', 'func-I-0-all', 'func-I-1-all', 'func-I-2-all', 'func-I-3-all', 'func-T-0-all',
    'func-T-1-all', 'func-T-2-all', 'func-T-3-all', 'func-S-0-all', 'func-S-1-all', 'func-S-2-all', 'func-S-3-all',
    'func-alpha-0-all', 'func-alpha-1-all', 'func-alpha-2-all', 'func-alpha-3-all', 'D_func-chi-0-all',
    'D_func-chi-1-all', 'D_func-chi-2-all', 'D_func-chi-3-all', 'D_func-Z-0-all', 'D_func-Z-1-all', 'D_func-Z-2-all',
    'D_func-Z-3-all', 'D_func-I-0-all', 'D_func-I-1-all', 'D_func-I-2-all', 'D_func-I-3-all', 'D_func-T-0-all',
    'D_func-T-1-all', 'D_func-T-2-all', 'D_func-T-3-all', 'D_func-S-0-all', 'D_func-S-1-all', 'D_func-S-2-all',
    'D_func-S-3-all', 'D_func-alpha-0-all', 'D_func-alpha-1-all', 'D_func-alpha-2-all', 'D_func-alpha-3-all'
]

summed_linker_descriptors = [  # pylint:disable=invalid-name
    'sum-f-lig-chi-0', 'sum-f-lig-chi-1', 'sum-f-lig-chi-2', 'sum-f-lig-chi-3', 'sum-f-lig-Z-0', 'sum-f-lig-Z-1',
    'sum-f-lig-Z-2', 'sum-f-lig-Z-3', 'sum-f-lig-I-0', 'sum-f-lig-I-1', 'sum-f-lig-I-2', 'sum-f-lig-I-3',
    'sum-f-lig-T-0', 'sum-f-lig-T-1', 'sum-f-lig-T-2', 'sum-f-lig-T-3', 'sum-f-lig-S-0', 'sum-f-lig-S-1',
    'sum-f-lig-S-2', 'sum-f-lig-S-3', 'sum-lc-chi-0-all', 'sum-lc-chi-1-all', 'sum-lc-chi-2-all', 'sum-lc-chi-3-all',
    'sum-lc-Z-0-all', 'sum-lc-Z-1-all', 'sum-lc-Z-2-all', 'sum-lc-Z-3-all', 'sum-lc-I-0-all', 'sum-lc-I-1-all',
    'sum-lc-I-2-all', 'sum-lc-I-3-all', 'sum-lc-T-0-all', 'sum-lc-T-1-all', 'sum-lc-T-2-all', 'sum-lc-T-3-all',
    'sum-lc-S-0-all', 'sum-lc-S-1-all', 'sum-lc-S-2-all', 'sum-lc-S-3-all', 'sum-lc-alpha-0-all', 'sum-lc-alpha-1-all',
    'sum-lc-alpha-2-all', 'sum-lc-alpha-3-all', 'sum-D_lc-chi-0-all', 'sum-D_lc-chi-1-all', 'sum-D_lc-chi-2-all',
    'sum-D_lc-chi-3-all', 'sum-D_lc-Z-0-all', 'sum-D_lc-Z-1-all', 'sum-D_lc-Z-2-all', 'sum-D_lc-Z-3-all',
    'sum-D_lc-I-0-all', 'sum-D_lc-I-1-all', 'sum-D_lc-I-2-all', 'sum-D_lc-I-3-all', 'sum-D_lc-T-0-all',
    'sum-D_lc-T-1-all', 'sum-D_lc-T-2-all', 'sum-D_lc-T-3-all', 'sum-D_lc-S-0-all', 'sum-D_lc-S-1-all',
    'sum-D_lc-S-2-all', 'sum-D_lc-S-3-all', 'sum-D_lc-alpha-0-all', 'sum-D_lc-alpha-1-all', 'sum-D_lc-alpha-2-all',
    'sum-D_lc-alpha-3-all'
]

summed_metalcenter_descriptors = [  # pylint:disable=invalid-name
    'sum-mc_CRY-chi-0-all', 'sum-mc_CRY-chi-1-all', 'sum-mc_CRY-chi-2-all', 'sum-mc_CRY-chi-3-all',
    'sum-mc_CRY-Z-0-all', 'sum-mc_CRY-Z-1-all', 'sum-mc_CRY-Z-2-all', 'sum-mc_CRY-Z-3-all', 'sum-mc_CRY-I-0-all',
    'sum-mc_CRY-I-1-all', 'sum-mc_CRY-I-2-all', 'sum-mc_CRY-I-3-all', 'sum-mc_CRY-T-0-all', 'sum-mc_CRY-T-1-all',
    'sum-mc_CRY-T-2-all', 'sum-mc_CRY-T-3-all', 'sum-mc_CRY-S-0-all', 'sum-mc_CRY-S-1-all', 'sum-mc_CRY-S-2-all',
    'sum-mc_CRY-S-3-all', 'sum-D_mc_CRY-chi-0-all', 'sum-D_mc_CRY-chi-1-all', 'sum-D_mc_CRY-chi-2-all',
    'sum-D_mc_CRY-chi-3-all', 'sum-D_mc_CRY-Z-0-all', 'sum-D_mc_CRY-Z-1-all', 'sum-D_mc_CRY-Z-2-all',
    'sum-D_mc_CRY-Z-3-all', 'sum-D_mc_CRY-I-0-all', 'sum-D_mc_CRY-I-1-all', 'sum-D_mc_CRY-I-2-all',
    'sum-D_mc_CRY-I-3-all', 'sum-D_mc_CRY-T-0-all', 'sum-D_mc_CRY-T-1-all', 'sum-D_mc_CRY-T-2-all',
    'sum-D_mc_CRY-T-3-all', 'sum-D_mc_CRY-S-0-all', 'sum-D_mc_CRY-S-1-all', 'sum-D_mc_CRY-S-2-all',
    'sum-D_mc_CRY-S-3-all'
]

summed_functionalgroup_descriptors = [  # pylint:disable=invalid-name
    'sum-func-chi-0-all', 'sum-func-chi-1-all', 'sum-func-chi-2-all', 'sum-func-chi-3-all', 'sum-func-Z-0-all',
    'sum-func-Z-1-all', 'sum-func-Z-2-all', 'sum-func-Z-3-all', 'sum-func-I-0-all', 'sum-func-I-1-all',
    'sum-func-I-2-all', 'sum-func-I-3-all', 'sum-func-T-0-all', 'sum-func-T-1-all', 'sum-func-T-2-all',
    'sum-func-T-3-all', 'sum-func-S-0-all', 'sum-func-S-1-all', 'sum-func-S-2-all', 'sum-func-S-3-all',
    'sum-func-alpha-0-all', 'sum-func-alpha-1-all', 'sum-func-alpha-2-all', 'sum-func-alpha-3-all',
    'sum-D_func-chi-0-all', 'sum-D_func-chi-1-all', 'sum-D_func-chi-2-all', 'sum-D_func-chi-3-all',
    'sum-D_func-Z-0-all', 'sum-D_func-Z-1-all', 'sum-D_func-Z-2-all', 'sum-D_func-Z-3-all', 'sum-D_func-I-0-all',
    'sum-D_func-I-1-all', 'sum-D_func-I-2-all', 'sum-D_func-I-3-all', 'sum-D_func-T-0-all', 'sum-D_func-T-1-all',
    'sum-D_func-T-2-all', 'sum-D_func-T-3-all', 'sum-D_func-S-0-all', 'sum-D_func-S-1-all', 'sum-D_func-S-2-all',
    'sum-D_func-S-3-all', 'sum-D_func-alpha-0-all', 'sum-D_func-alpha-1-all', 'sum-D_func-alpha-2-all',
    'sum-D_func-alpha-3-all'
]


@click.command('cli')
@click.argument('epsilon', type=float, default=0.05, required=False)
@click.argument('delta', type=float, default=0.05, required=False)
@click.argument('beta_scale', type=float, default=1 / 20, required=False)
@click.argument('outdir', type=click.Path(), default='.', required=False)
def main(epsilon, delta, beta_scale, outdir):

    pmof_test = pd.read_csv('data/PMOF20K_traindata_7000_test.csv')  # pylint:disable=invalid-name
    pmof_train = pd.read_csv('data/PMOF20K_traindata_7000_train.csv')  # pylint:disable=invalid-name
    pmof = pd.concat([pmof_test, pmof_train])  # pylint:disable=invalid-name
    pmof['CO2_DC'] = pmof['pure_uptake_CO2_298.00_1600000'] - pmof['pure_uptake_CO2_298.00_15000']
    y = pmof[['CO2_DC', 'CH4DC']].values  # pylint:disable=invalid-name

    #label_scaler = StandardScaler()
    #y = label_scaler.fit_transform(y)

    feat = set([  # pylint:disable=invalid-name
        'func-chi-0-all', 'D_func-S-3-all', 'total_SA_volumetric', 'Di', 'Dif', 'mc_CRY-Z-0-all',
        'total_POV_volumetric', 'density [g/cm^3]', 'total_SA_gravimetric', 'D_func-S-1-all', 'Df', 'mc_CRY-S-0-all',
        'total_POV_gravimetric', 'D_func-alpha-1-all', 'func-S-0-all', 'D_mc_CRY-chi-3-all', 'D_mc_CRY-chi-1-all',
        'func-alpha-0-all', 'D_mc_CRY-T-2-all', 'mc_CRY-Z-2-all', 'D_mc_CRY-chi-2-all', 'total_SA_gravimetric',
        'total_POV_gravimetric', 'Di', 'density [g/cm^3]', 'func-S-0-all', 'func-chi-2-all', 'func-alpha-0-all',
        'total_POV_volumetric', 'D_func-alpha-1-all', 'total_SA_volumetric', 'func-alpha-1-all', 'func-alpha-3-all',
        'Dif', 'Df', 'func-chi-3-all'
    ])

    X = pmof[feat]

    X = StandardScaler().fit_transform(X)

    X_train, y_train, indices = get_maxmin_samples(X, y, N_SAMPLES)  # pylint:disable=invalid-name
    X_test = np.delete(X, indices, 0)  # pylint:disable=invalid-name
    y_test = np.delete(y, indices, 0)  # pylint:disable=invalid-name

    models = [build_coregionalized_model(X_train, y_train, kernel=GPy.kern.Matern52(X_train.shape[1], ARD=False))]  # pylint:disable=invalid-name

    X = np.vstack([X_train, X_test])
    y = np.vstack([y_train, y_test])
    print(y.shape, X.shape)

    palinstance = PALCoregionalized(X, models, 2, epsilon=epsilon, beta_scale=beta_scale, delta=delta)
    palinstance.update_train_set(indices, y[indices])

    print('STARTING PAL')
    pareto_optimals = []
    hypervolumess = []
    selecteds = []
    discardeds = []
    unclassifieds = []
    outdir = '.'
    while sum(palinstance.unclassified):
        idx = palinstance.run_one_step()
        pareto_optimals.append(palinstance.pareto_optimal_indices)
        selecteds.append(palinstance.sampled_indices)
        unclassifieds.append(palinstance.unclassified_indices)
        discardeds.append(palinstance.discarded_indices)
        hv = get_hypervolume(y[palinstance.pareto_optimal_indices], [5, 5])
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
    np.save(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-greedy_indices'.format(epsilon, delta, beta_scale, N_SAMPLES)),
            indices)
    np.save(
        os.path.join(outdir,
                     TIMESTR + '{}_{}_{}_{}-pareto_optimal_indices'.format(epsilon, delta, beta_scale, N_SAMPLES)),
        pareto_optimals)
    np.save(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-hypervolumes'.format(epsilon, delta, beta_scale, N_SAMPLES)),
            hypervolumess)
    np.save(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-selected'.format(epsilon, delta, beta_scale, N_SAMPLES)),
            selecteds)
    np.save(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-discarded'.format(epsilon, delta, beta_scale, N_SAMPLES)),
            discardeds)
    np.save(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-unclassified'.format(epsilon, delta, beta_scale, N_SAMPLES)),
            unclassifieds)

    with open(os.path.join(outdir, TIMESTR + '{}_{}_{}_{}-models.pkl'.format(epsilon, delta, beta_scale, N_SAMPLES)),
              'wb') as fh:
        pickle.dump(palinstance.models, fh)


if __name__ == '__main__':
    main()
