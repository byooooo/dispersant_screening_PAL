# -*- coding: utf-8 -*-
"""Setting up pytest"""
from __future__ import absolute_import

import pytest
from sklearn.datasets import make_regression


@pytest.fixture()
def make_random_dataset(targets=3):
    return make_regression(n_samples=100, n_features=10, n_informative=8, n_targets=targets)
