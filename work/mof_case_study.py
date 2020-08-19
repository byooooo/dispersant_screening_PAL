# -*- coding: utf-8 -*-
import GPy
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from dispersant_screener.gp import (build_coregionalized_model, build_model, predict_coregionalized,
                                    set_xy_coregionalized)
from dispersant_screener.pal import pal
from dispersant_screener.utils import get_kmeans_samples

other_descriptors = ['CellV [A^3]']

geometric_descirptors = [
    'Di', 'Df', 'Dif', 'density [g/cm^3]', 'total_SA_volumetric', 'total_SA_gravimetric', 'total_POV_volumetric',
    'total_POV_gravimetric'
]

linker_descriptors = [
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

metalcenter_descriptors = [
    'mc_CRY-chi-0-all', 'mc_CRY-chi-1-all', 'mc_CRY-chi-2-all', 'mc_CRY-chi-3-all', 'mc_CRY-Z-0-all', 'mc_CRY-Z-1-all',
    'mc_CRY-Z-2-all', 'mc_CRY-Z-3-all', 'mc_CRY-I-0-all', 'mc_CRY-I-1-all', 'mc_CRY-I-2-all', 'mc_CRY-I-3-all',
    'mc_CRY-T-0-all', 'mc_CRY-T-1-all', 'mc_CRY-T-2-all', 'mc_CRY-T-3-all', 'mc_CRY-S-0-all', 'mc_CRY-S-1-all',
    'mc_CRY-S-2-all', 'mc_CRY-S-3-all', 'D_mc_CRY-chi-0-all', 'D_mc_CRY-chi-1-all', 'D_mc_CRY-chi-2-all',
    'D_mc_CRY-chi-3-all', 'D_mc_CRY-Z-0-all', 'D_mc_CRY-Z-1-all', 'D_mc_CRY-Z-2-all', 'D_mc_CRY-Z-3-all',
    'D_mc_CRY-I-0-all', 'D_mc_CRY-I-1-all', 'D_mc_CRY-I-2-all', 'D_mc_CRY-I-3-all', 'D_mc_CRY-T-0-all',
    'D_mc_CRY-T-1-all', 'D_mc_CRY-T-2-all', 'D_mc_CRY-T-3-all', 'D_mc_CRY-S-0-all', 'D_mc_CRY-S-1-all',
    'D_mc_CRY-S-2-all', 'D_mc_CRY-S-3-all'
]

functionalgroup_descriptors = [
    'func-chi-0-all', 'func-chi-1-all', 'func-chi-2-all', 'func-chi-3-all', 'func-Z-0-all', 'func-Z-1-all',
    'func-Z-2-all', 'func-Z-3-all', 'func-I-0-all', 'func-I-1-all', 'func-I-2-all', 'func-I-3-all', 'func-T-0-all',
    'func-T-1-all', 'func-T-2-all', 'func-T-3-all', 'func-S-0-all', 'func-S-1-all', 'func-S-2-all', 'func-S-3-all',
    'func-alpha-0-all', 'func-alpha-1-all', 'func-alpha-2-all', 'func-alpha-3-all', 'D_func-chi-0-all',
    'D_func-chi-1-all', 'D_func-chi-2-all', 'D_func-chi-3-all', 'D_func-Z-0-all', 'D_func-Z-1-all', 'D_func-Z-2-all',
    'D_func-Z-3-all', 'D_func-I-0-all', 'D_func-I-1-all', 'D_func-I-2-all', 'D_func-I-3-all', 'D_func-T-0-all',
    'D_func-T-1-all', 'D_func-T-2-all', 'D_func-T-3-all', 'D_func-S-0-all', 'D_func-S-1-all', 'D_func-S-2-all',
    'D_func-S-3-all', 'D_func-alpha-0-all', 'D_func-alpha-1-all', 'D_func-alpha-2-all', 'D_func-alpha-3-all'
]

summed_linker_descriptors = [
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

summed_metalcenter_descriptors = [
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

summed_functionalgroup_descriptors = [
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

pmof_test = pd.read_csv('PMOF20K_traindata_7000_test.csv')
pmof_train = pd.read_csv('PMOF20K_traindata_7000_train.csv')
pmof = pd.concat([pmof_test, pmof_train])
pmof['CO2_DC'] = pmof['pure_uptake_CO2_298.00_1600000'] - pmof['pure_uptake_CO2_298.00_15000']
y = pmof[['CO2_DC', 'CH4DC']].values

feat = set([
    'func-chi-0-all', 'D_func-S-3-all', 'total_SA_volumetric', 'Di', 'Dif', 'mc_CRY-Z-0-all', 'total_POV_volumetric',
    'density [g/cm^3]', 'total_SA_gravimetric', 'D_func-S-1-all', 'Df', 'mc_CRY-S-0-all', 'total_POV_gravimetric',
    'D_func-alpha-1-all', 'func-S-0-all', 'D_mc_CRY-chi-3-all', 'D_mc_CRY-chi-1-all', 'func-alpha-0-all',
    'D_mc_CRY-T-2-all', 'mc_CRY-Z-2-all', 'D_mc_CRY-chi-2-all', 'total_SA_gravimetric', 'total_POV_gravimetric', 'Di',
    'density [g/cm^3]', 'func-S-0-all', 'func-chi-2-all', 'func-alpha-0-all', 'total_POV_volumetric',
    'D_func-alpha-1-all', 'total_SA_volumetric', 'func-alpha-1-all', 'func-alpha-3-all', 'Dif', 'Df', 'func-chi-3-all'
])

X = pmof[feat]

X = StandardScaler().fit_transform(X)
X = VarianceThreshold(0.2).fit_transform(X)

X_train, y_train, indices = get_kmeans_samples(X, y, 170)
X_test = np.delete(X, indices, 0)
y_test = np.delete(y, indices, 0)

models = [
    build_model(X_train, y_train, 0, kernel=GPy.kern.RBF(X_train.shape[1], ARD=False)),
    build_model(X_train, y_train, 1, kernel=GPy.kern.RBF(X_train.shape[1], ARD=False))
]
pareto_optimal, hypervolumes, gps, sampled = pal(models,
                                                 X_train,
                                                 y_train,
                                                 X_test,
                                                 y_test,
                                                 hv_reference=[5, 5],
                                                 iterations=1000,
                                                 epsilon=[0.01, 0.01],
                                                 delta=0.02,
                                                 beta_scale=1 / 3,
                                                 coregionalized=False)

np.save('pareto_optimal.npy', pareto_optimal)
np.save('hypervolumes.npy', hypervolumes)
np.save('sampled.npy', sampled)
np.save('x_test.npy', X_test)
np.save('x_train.npy', X_train)
np.save('y_test.npy', y_test)
np.save('y_train.npy', y_train)
np.save('sampled.npy', sampled)
joblib.dump(gps, 'gps.joblib')
