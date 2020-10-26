# -*- coding: utf-8 -*-
from __future__ import absolute_import

from dispersant_screener.featurizer import LinearPolymerSmilesFeaturizer


def test_head_tail():
    pmsf = LinearPolymerSmilesFeaturizer('[W][W][W][W][Ta][Ta][Ta][Ta][W][W][W]')
    head_tail = LinearPolymerSmilesFeaturizer.get_head_tail_features('[W][W][W][W][Ta][Ta][Ta][Ta][W][W][W]', pmsf.characters)
    assert head_tail == {'head_tail_[W]': 2, 'head_tail_[Tr]': 0, 'head_tail_[Ta]': 0, 'head_tail_[R]': 0}


def test_clusters():
    pmsf = LinearPolymerSmilesFeaturizer('[W][W][W][W][Ta][Ta][Ta][Ta][W][W][W]')
    cluster_feat = LinearPolymerSmilesFeaturizer.get_cluster_stats(pmsf.smiles, pmsf.replacement_dict)
    assert cluster_feat['total_clusters'] == 3
    assert cluster_feat['num_[W]'] == 2 / 3
    assert cluster_feat['num_[Ta]'] == 1 / 3
    assert cluster_feat['num_[Tr]'] == 0
    assert cluster_feat['num_[R]'] == 0
