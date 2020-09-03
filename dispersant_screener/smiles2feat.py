# -*- coding: utf-8 -*-
"""This module provides code to attempt to convert features to polymer SMILES
The code is not very fancy and uses several heuristics so it might not be able to find
a solution even though there is one.

So far (2020/08/19) we consider only length, the bead fractions and the maximum cluster size
"""

import math
import random
from functools import partial

import numpy as np

from .featurizer import LinearPolymerSmilesFeaturizer


def solve(chars: list, safe_up_to: callable):
    """Finds a solution to a backtracking problem. Based on  https://www.geeksforgeeks.org/backtracking-algorithms/

    Args:
        chars (list): a sequence of values to try, in order]
        safe_up_to (callable): returns whether the values assigned to slots 0..pos in
                  the solution list, satisfy the problem constraints.

    Returns:
        list:  Return the solution as a list of values.
    """
    values = chars.copy()
    size = len(values)
    solution = [None] * size

    def extend_solution(position):
        if values:
            for i, value in enumerate(values):
                solution[position] = value
                if safe_up_to(solution, position):
                    _ = values.pop(i)
                    if position >= size - 1 or extend_solution(position + 1):
                        return solution
        return solution

    return extend_solution(0)


def _cluster_not_too_large(found_features, expected_features, beads=['[W]', '[Ta]', '[R]', '[Tr]']):  # pylint:disable=dangerous-default-value
    for bead in beads:
        max_feat = 'max_' + bead
        if found_features[max_feat] > expected_features[max_feat]:
            return False
    return True


def _get_available_counts(counter, found_counter):
    available = {}
    for k, v in counter.items():
        available[k] = v - found_counter[k]

    return available


def _cluster_still_possible(available, found_features, expected_features):
    beads = list(available.keys())

    for bead in beads:
        max_feat = 'max_' + bead
        if (available[bead] < expected_features[max_feat]) & (found_features[max_feat] < expected_features[max_feat]):
            return False

    return True


def _no_unallowed_clusters_form(available, expected_features):
    beads = list(available.keys())
    total_available = sum(available.values())
    for bead in beads:
        others = total_available - available[bead]
        max_feat = 'max_' + bead
        if expected_features[max_feat] > 0:
            max_n = math.floor(available[bead] / expected_features[max_feat])
        else:
            max_n = available[bead]
        if available[bead] > max_n * expected_features[max_feat] - (max_n - 1) * others:
            return False

    return True


def safe_up_to_full(characters, position, expected_features, chars):
    c = characters[:position + 1]
    smiles = ''.join(c)

    alls = ''.join(chars)
    lp = LinearPolymerSmilesFeaturizer(alls)
    counter = lp.get_counts(alls, lp.characters)

    lp = LinearPolymerSmilesFeaturizer(smiles)

    try:
        found_features = lp.get_cluster_stats(smiles, lp.replacement_dict)
        found_counts = lp.get_counts(smiles, lp.characters)

        available = _get_available_counts(counter, found_counts)

        if not _cluster_not_too_large(found_features, expected_features):
            return False

        if not _cluster_still_possible(available, found_features, expected_features):
            return False

        if not _no_unallowed_clusters_form(available, expected_features):
            return False

    except ZeroDivisionError as e:
        return True

    return True


def get_cap(pool: list, exclude: str):
    """Given a pool of characters remove on character
    that is not exclude from the pool and return it
    with the smaller pool.

    The cap can be used to protect a cluster from
    getting too large in a random permuation
    """
    for i, char in enumerate(pool):
        if char != exclude:
            cap = pool.pop(i)
            break

    return pool, cap


def bundle_indv(pool: list) -> list:
    """Given a pool try to recursively build maximally disordered fragments.
    This is thought to be a heuristic to avoid cluster growth and to expedite
    the assembly of polymers.
    One needs to be cautious though as it makes the growth of clusters (which might
    be allowed for a given feature set) unfavorable. But it will speed up the permutation
    as there are now less building blocks and less possible permutations.

    Example:
        >pool = ['a', 'b', 'a', 'c', 'b', 'a',  'b', 'c']
        >bundle_indv(pool)
        ['ab', 'cba', 'abc']

    Args:
        pool (list): Character pool

    Returns:
        [list]: List of bundled characters
    """
    if len(pool) > 0:
        pairs = []
        char_in_pool = set(pool)

        character_lists = []

        lengths = []

        for poolchar in char_in_pool:
            sublist = []
            for char in pool:
                if poolchar == char:
                    sublist.append(char)
            character_lists.append(sublist)
            lengths.append(len(sublist))

        num_pairs = min(lengths)

        for i in reversed(range(num_pairs)):
            pair = []
            random.shuffle(character_lists)
            for sublist in character_lists:
                pair.append(sublist.pop(i))

            pairs.append(''.join(pair))

        flat_sublists = sum(character_lists, [])

        flat_sublists = bundle_indv(flat_sublists)
        return flat_sublists + pairs
    return pool


def get_building_blocks(feat_dict: dict, bundle: bool = True, cap: bool = True) -> list:  # pylint:disable=too-many-locals, too-many-statements, too-many-branches
    """Given a feature dictionary (input for supervised ML or output of the GA)
    construct the building blocks that can be used to assemble a polymer that would give
    those features. This mapping is not unique and we apply some heuristics like
    random capping of clusters and (optionally) bundling of monomers to make it easier to find a solution.

    Args:
        feat_dict (dict): dictionary mapping feature name to feature value.
            It assummes the feature names from the LinearPolymer featurizer class
        bundle (bool, optional): If true, it tries to bundle up monomers to maximally disordered fragmetns.
            Defaults to True.
        cap (bool, optional): If true, if will use random characters from the pool for possible characters
            to cap the clusters. Defaults to True.

    Raises:
        ValueError: Will be raised if the features are not consistent. E.g., if the length feature does
            not equal the sum of the number of monomers.

    Returns:
        list: Bulding blocks of the monomers like clusters and (bundled) monormers
    """
    W, Tr, R, Ta, W_cl, Tr_cl, R_cl, Ta_cl, length = feat_dict['[W]'], feat_dict['[Tr]'], feat_dict['[R]'], feat_dict[  # pylint:disable=invalid-name
        '[Ta]'], feat_dict['max_[W]'], feat_dict['max_[Tr]'], feat_dict['max_[R]'], feat_dict['max_[Ta]'], feat_dict[
            'length']
    length_ = length
    length = round(length)
    w = round(W * length_)  # pylint:disable=invalid-name
    tr = round(Tr * length_)  # pylint:disable=invalid-name
    r = round(R * length_)  # pylint:disable=invalid-name
    ta = round(Ta * length_)  # pylint:disable=invalid-name

    total_length = w + tr + r + ta
    if total_length == length:  # pylint:disable=no-else-return (it is clearer now with the else)
        w_cluster = round(W_cl)
        tr_cluster = round(Tr_cl)
        r_cluster = round(R_cl)
        ta_cluster = round(Ta_cl)

        w_indv = w - w_cluster
        tr_indv = tr - tr_cluster
        r_indv = r - r_cluster
        ta_indv = ta - ta_cluster

        building_blocks = []

        indv_pool = []

        if w_indv:
            indv_pool.extend(['[W]'] * w_indv)
        if tr_indv:
            indv_pool.extend(['[Tr]'] * tr_indv)
        if r_indv:
            indv_pool.extend(['[R]'] * r_indv)
        if ta_indv:
            indv_pool.extend(['[Ta]'] * ta_indv)

        if w_cluster:
            if cap:
                try:
                    indv_pool, cap_a = get_cap(indv_pool, '[W]')
                    indv_pool, cap_b = get_cap(indv_pool, '[W]')
                except Exception:  # pylint:disable=broad-except
                    cap_a, cap_b = '', ''
                building_blocks.append(cap_a + '[W]' * w_cluster + cap_b)
            else:
                building_blocks.append('[W]' * w_cluster)
        if tr_cluster:
            if cap:
                try:
                    indv_pool, cap_a = get_cap(indv_pool, '[Tr]')
                    indv_pool, cap_b = get_cap(indv_pool, '[Tr]')
                except Exception:  # pylint:disable=broad-except
                    cap_a, cap_b = '', ''
                building_blocks.append(cap_a + '[Tr]' * tr_cluster + cap_b)
            else:
                building_blocks.append('[Tr]' * tr_cluster)
        if r_cluster:
            if cap:
                try:
                    indv_pool, cap_a = get_cap(indv_pool, '[R]')
                    indv_pool, cap_b = get_cap(indv_pool, '[R]')
                except Exception:  # pylint:disable=broad-except
                    cap_a, cap_b = '', ''
                building_blocks.append(cap_a + '[R]' * r_cluster + cap_b)
            else:
                building_blocks.append('[R]' * r_cluster)
        if ta_cluster:
            if cap:
                try:
                    indv_pool, cap_a = get_cap(indv_pool, '[Ta]')
                    indv_pool, cap_b = get_cap(indv_pool, '[Ta]')
                except Exception:  # pylint:disable=broad-except
                    cap_a, cap_b = '', ''
                building_blocks.append(cap_a + '[Ta]' * ta_cluster + cap_b)
            else:
                building_blocks.append('[Ta]' * ta_cluster)

        if bundle:
            indv_pool = bundle_indv(indv_pool)

        building_blocks.extend(indv_pool)
        random.shuffle(building_blocks)

        return building_blocks
    else:
        # ToDo make this error handling a bit more robust. In general, it is not only the length that can be wrong
        raise ValueError('Length does not match {}, {}'.format(length, total_length))


def check_validity(smiles: str, feat_dict: dict) -> bool:
    """Given a SMILES and a feature dict check if they match.
    I.e., if we featurize the SMILES do we obtain the feature dictionary?

    Args:
        smiles (str): [description]
        feat_dict (dict): [description]

    Returns:
        bool: [description]
    """
    lp = LinearPolymerSmilesFeaturizer(smiles)  # pylint:disable=invalid-name
    feat = lp.featurize()
    # check length
    # if feat['length'] % 2 != 0:
    #     return False

    if feat['max_[Ta]'] != feat_dict['max_[Ta]']:
        return False

    if feat['max_[Tr]'] != feat_dict['max_[Tr]']:
        return False

    if feat['max_[R]'] != feat_dict['max_[R]']:
        return False

    if feat['max_[W]'] != feat_dict['max_[W]']:
        return False

    return True


def get_smiles(feat_dict, max_smiles: int = 5, max_trials: int = 100) -> list:

    solutions = []
    trials = 0
    while len(solutions) < max_smiles and (trials < max_trials):
        chars = get_building_blocks(feat_dict=feat_dict, cap=False, bundle=False)
        safe_up_to = partial(safe_up_to_full, expected_features=feat_dict, chars=chars)
        solution = solve(chars, safe_up_to)
        if not None in solution:
            solutions.append(solution)
        trials += 1

    return solutions


def get_smiles_legacy(feat_dict: dict, max_smiles: int = 5, max_trials: int = 100) -> list:
    """Try to generate polymer smiles based on a feature dictionary

    Args:
        feat_dict (dict): mapping feature names to values, folliowing the convention from
            the LinearPolymer featurizer class
        max_smiles (int, optional): Maximum SMILES that are returned for a given feature dictionary.
            Defaults to 5.
        max_trials (int, optional): Maximum attempts for randomm permutation of building blocks.
            Defaults to 100.

    Returns:
        list: SMILES strings
    """
    try:
        polymer = get_building_blocks(feat_dict)
        perms = []
        reversed_perms = []
        for _ in range(max_smiles):
            trials = 0
            while True and (trials < max_trials):
                polymer = get_building_blocks(feat_dict)
                perm = np.random.permutation(polymer)
                rev = reversed(perm)
                key = ''.join(perm)
                rev_key = ''.join(rev)

                # Check especially important to verify that we didn't create too large clusters
                # With the individual characters
                valid = check_validity(key, feat_dict)
                if valid:
                    # Doesn't matter in which order the smiles is written
                    if (key or rev_key not in perms) and (key or rev_key not in perms):
                        perms.append(key)
                        reversed_perms.append(rev_key)
                        break
                trials += 1

        if len(perms) == 0:
            trials = 0
            while True and (trials < max_trials):
                polymer = get_building_blocks(feat_dict, bundle=False)
                perm = np.random.permutation(polymer)
                rev = reversed(perm)
                key = ''.join(perm)
                rev_key = ''.join(rev)

                # Check especially important to verify that we didn't create too large clusters
                # With the individual characters
                valid = check_validity(key, feat_dict)
                if valid:
                    # Doesn't matter in which order the smiles is written
                    if (key or rev_key not in perms) and (key or rev_key not in perms):
                        perms.append(key)
                        reversed_perms.append(rev_key)
                        break
                trials += 1
        return perms
    except ValueError:
        return []
