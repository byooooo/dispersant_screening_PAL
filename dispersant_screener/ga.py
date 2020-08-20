# -*- coding: utf-8 -*-
"""Module that implements the genetic algorithm, i.e., mainly the fitness function"""
from typing import Tuple

import numpy as np
from geneticalgorithm import geneticalgorithm as ga  # pylint:disable=unused-import
from functools import partial
from .smiles2feat import get_smiles

DEFAULT_GA_PARAM = {
    'max_num_iteration': 5000,
    'elit_ratio': 0.02,
    'population_size': 100,
    'mutation_probability': 0.1,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}

FEATURES = ['max_[Ta]', 'max_[W]', 'max_[R]', 'max_[Tr]', '[W]', '[Tr]', '[Ta]', '[R]', 'length']

FEAT_DICT = dict(zip(FEATURES, range(len(FEATURES))))


def get_bounds(features: list) -> Tuple[np.array, np.array]:
    """Given a list of feature names, generate a numpy array
    with the bounds for the optimization and the variable types.

    Args:
        features (list): List of feature names, following the convention
            from the LinearPolymer featurizer class

    Returns:
        Tuple(np.array np.array): Bounds and variable type in format for the
            geneticalgorithm package
    """
    lower_bound = []
    upper_bound = []
    vartype = []

    for feat in features:
        if 'head' in feat:
            lower_bound.append(0)
            upper_bound.append(2)
            vartype.append(['int'])
        elif feat == 'length':
            lower_bound.append(16)
            upper_bound.append(48)
            vartype.append(['int'])
        elif 'max' in feat:
            lower_bound.append(0)
            upper_bound.append(32)
            vartype.append(['int'])
        else:
            lower_bound.append(0)
            upper_bound.append(1)
            vartype.append(['real'])

    return np.array(list(zip(lower_bound, upper_bound))), np.array(vartype)


def constrain_cluster(x: np.array, feat_dict: dict) -> float:  # pylint: disable=invalid-name
    """Return a penalty for non-matching cluster sizes and number of beads

    Args:
        x (np.array): feature vector
        feat_dict (dict): dictionary that maps indices in the feature vector
            to feature names that follow the convention from the LinearPolymer
            featurizer

    Returns:
        float: penalty
    """
    pentalty = 0
    if x[feat_dict['max_[W]']] > x[feat_dict['[W]']] * x[feat_dict['length']]:
        pentalty += 30
    if x[feat_dict['max_[Ta]']] > x[feat_dict['[Ta]']] * x[feat_dict['length']]:
        pentalty += 30
    if x[feat_dict['max_[Tr]']] > x[feat_dict['[Tr]']] * x[feat_dict['length']]:
        pentalty += 30
    if x[feat_dict['max_[R]']] > x[feat_dict['[R]']] * x[feat_dict['length']]:
        pentalty += 30

    return pentalty


def constrain_validity(x: np.array, features: list) -> float:  # pylint: disable=invalid-name
    """Return a penality for invalid features (i.e., such that we cannot convert to a SMILES)
    and a reward for such features that we convert to a SMILES

    Args:
        x (np.array): feature vector
        features (list): list of feature names following the convention
            from the LinearPolymer featurizatin class

    Returns:
        float: penalty
    """
    smiles = get_smiles(dict(zip(features, x)), 1, 10)
    if smiles:
        return -5
    return 50


def constrain_length(x: np.array, feat_dict: dict) -> float:  # pylint: disable=invalid-name
    """Return a penality for feature vector in which the number of beads does
    not add up to the total length feature

    Args:
        x (np.array): feature vector
        feat_dict (dict): mapping indices of the feature vector to feature names
            following the convention from the LinearPolymer featurizer

    Returns:
        float: penalty
    """
    w = x[feat_dict['[W]']]  # pylint: disable=invalid-name
    tr = x[feat_dict['[Tr]']]  # pylint: disable=invalid-name
    ta = x[feat_dict['[Ta]']]  # pylint: disable=invalid-name
    r = x[feat_dict['[R]']]  # pylint: disable=invalid-name

    feat_length = x[feat_dict['length']]
    length = round(w * feat_length) + round(tr * feat_length) + round(ta * feat_length) + round(r * feat_length)
    if length != feat_length:
        return 100

    return 0


def predict_gbdt(model, X):
    return model.predict(X)


def predict_gpy(model, X):
    mu, _ = model.predict(X)
    return mu


def objective(x: np.array, predict, X_data: np.array) -> float:  # pylint: disable=invalid-name
    """An opjective function that an optimizer should attempt to minimize.
    I.e. penalities are positive and if it is getting better, the output is negative/smaller
    This fitness function also includes a "not-novel" penality", i.e. it adds a term that it inversely
    propertional to the smallest distance from the training set as "regularizer"

    Args:
        x (np.array): feature vector
        model (func): a function that takes the feature vector as argument and return the prediction
        X_data (np.array): dataset which is used as reference data for the novelty penality.
            To make this more efficient, one might try using K-means or other summarized forms
            of a dataset

    Returns:
        float: loss
    """
    y = predict(x.reshape(1, -1))  # pylint: disable=invalid-name

    regularize_cluster = constrain_cluster(x, FEAT_DICT)
    regularize_validity = constrain_validity(x, FEATURES)
    regularizer_noverly = 1 / (np.min(np.linalg.norm(x - X_data, axis=1)))**2

    return -y + regularizer_noverly + regularize_validity + regularize_cluster


def run_ga(predict: function,
           background_data: np.array,
           ga_param: dict = DEFAULT_GA_PARAM,
           features: list = FEATURES,
           feat_dict: dict = FEAT_DICT) -> ga:
    """

    Args:
        predict (function): function that takes a feature vector and 
            returns a flot
        background_data (np.array): Data which is used to compute the 
            novelty penality
        ga_param (dict, optional): Parameters for the genetic algorithm. 
            Defaults to DEFAULT_GA_PARAM.
        features (list, optional): Feature names following the convention from LinearPolymerFeaturizer. 
            Defaults to FEATURES.
        feat_dict (dict, optional): Mapping indices in the feaature matrix to feature names. 
            Defaults to FEAT_DICT.

    Returns:
        ga: geneticalgorithm object
    """
    boundaries, types = get_bounds(features)
    objective_partial = partial(objective, predict=predict, X_data=background_data)
    ga_model = ga(function=objective_partial,
                  dimension=len(features),
                  variable_type_mixed=types,
                  variable_boundaries=boundaries,
                  algorithm_parameters=ga_param)

    ga_model.run()

    return ga_model