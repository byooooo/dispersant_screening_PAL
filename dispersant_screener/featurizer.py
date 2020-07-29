# -*- coding: utf-8 -*-
from __future__ import absolute_import

import math
import re

import numpy as np
import pandas as pd 
from six.moves import zip

def featurize_many(smiless: list) -> pd.DataFrame:
    features = []
    for smiles in smiless: 
        pmsf = PolymerSmilesFeaturizer(smiles)
        features.append(pmsf.featurize())
    return pd.DataFrame(features)

class PolymerSmilesFeaturizer:

    def __init__(self, smiles: str, normalized_cluster_stats: bool = True):
        self.smiles = smiles
        self.characters = ['[W]', '[Tr]', '[Ta]', '[R]']
        self.replacement_dict = dict(list(zip(self.characters, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])))
        self.normalized_cluster_stats = normalized_cluster_stats

    @staticmethod
    def get_head_tail_features(string: str, characters: list) -> dict:
        """0/1/2 encoded feature indicating if the building block is at start/end of the polymer chain"""
        is_head_tail = [0] * len(characters)

        for i, char in enumerate(characters):
            if string.startswith(char):
                is_head_tail[i] += 1
            if string.endswith(char):
                is_head_tail[i] += 1

        new_keys = ['head_tail_' + char for char in characters]
        return dict(list(zip(new_keys, is_head_tail)))

    @staticmethod
    def get_cluster_stats(s: str, replacement_dict: dict, normalized: bool = True) -> dict:
        clusters = PolymerSmilesFeaturizer.find_clusters(s, replacement_dict)
        cluster_stats = {}
        cluster_stats['total_clusters'] = 0
        for key, value in clusters.items():
            if value:
                cluster_stats['num' + '_' + key] = len(value)
                cluster_stats['total_clusters'] += len(value)
                cluster_stats['max' + '_' + key] = max(value)
                cluster_stats['min' + '_' + key] = min(value)
                cluster_stats['mean' + '_' + key] = np.mean(value)
            else:
                cluster_stats['num' + '_' + key] = 0
                cluster_stats['max' + '_' + key] = 0
                cluster_stats['min' + '_' + key] = 0
                cluster_stats['mean' + '_' + key] = 0

        if normalized:
            for key, value in cluster_stats.items():
                if 'num' in key:
                    cluster_stats[key] = value / cluster_stats['total_clusters']

        return cluster_stats

    @staticmethod
    def find_clusters(s: str, replacement_dict: dict) -> dict:
        clusters = re.findall(r'((\w)\2{1,})', PolymerSmilesFeaturizer._multiple_replace(s, replacement_dict))
        cluster_dict = dict(list(zip(replacement_dict.keys(), [[] for i in replacement_dict.keys()])))
        inv_replacement_dict = {v: k for k, v in replacement_dict.items()}
        for cluster, character in clusters:
            cluster_dict[inv_replacement_dict[character]].append(len(cluster))

        return cluster_dict

    @staticmethod
    def _multiple_replace(s: str, replacement_dict: dict) -> str:
        for word in replacement_dict:
            s = s.replace(word, replacement_dict[word])
        return s

    @staticmethod
    def get_counts(smiles: str, characters: list) -> dict:
        counts = [smiles.count(char) for char in characters]
        return dict(list(zip(characters, counts)))

    @staticmethod
    def get_relative_shannon(character_count: dict) -> float:
        counts = [c for c in character_count.values() if c > 0]
        length = sum(counts)
        probs = [count / length for count in counts]
        ideal_entropy = PolymerSmilesFeaturizer._entropy_max(length)
        entropy = -sum([p * math.log(p) / math.log(2.0) for p in probs])

        return entropy / ideal_entropy

    @staticmethod
    def _entropy_max(length: int) -> float:
        'Calculates the max Shannon entropy of a string with given length'

        prob = 1.0 / length

        return -1.0 * length * prob * math.log(prob) / math.log(2.0)

    @staticmethod
    def get_balance(character_count: dict) -> dict:
        counts = [c for c in character_count.values()]
        length = sum(counts)
        frequencies = [c / length for c in counts]
        return dict(list(zip(character_count.keys(), frequencies)))

    def _featurize(self):
        self._character_count = PolymerSmilesFeaturizer.get_counts(self.smiles, self.characters)
        self._balance = PolymerSmilesFeaturizer.get_balance(self._character_count)
        self._relative_shannon = PolymerSmilesFeaturizer.get_relative_shannon(self._character_count)
        self._cluster_stats = PolymerSmilesFeaturizer.get_cluster_stats(self.smiles, self.replacement_dict,
                                                                        self.normalized_cluster_stats)
        self._head_tail_feat = PolymerSmilesFeaturizer.get_head_tail_features(self.smiles, self.characters)

        self.features = self._head_tail_feat
        self.features.update(self._cluster_stats)
        self.features.update(self._balance)
        self.features['rel_shannon'] = self._relative_shannon
        self.features['length'] = sum(self._character_count.values())

    def featurize(self) -> dict:
        self._featurize()
        return self.features
