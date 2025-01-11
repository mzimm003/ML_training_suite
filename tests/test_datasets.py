"""
"""
import pytest

import torch
import numpy as np
import pathlib

from ml_training_suite.datasets.base import ClusterManager, train_test_split

@pytest.mark.parametrize(
    "idxs,clusters,train_size,test_size",
    [
        (
            np.arange(30),
            np.array([0,2,0,0,4,4,0,3,1,0,2,1,2,2,0,0,1,1,3,3,3,2,4,4,4,1,1,0,1,3]),
            20,
            10,
        ),
        (
            np.arange(30),
            np.array([0,2,0,0,4,4,0,3,1,0,2,1,2,2,0,0,1,1,3,3,3,2,4,4,4,1,1,0,1,3]),
            15,
            15,
        ),
    ])
def test_cluster_manager(idxs, clusters, train_size, test_size):
    c_m = ClusterManager()
    train_set, test_set = c_m.preserve_cluster_split(idxs, clusters, train_size, test_size)
    assert not np.any(np.isin(train_set, test_set))
    assert not np.any(np.isin(test_set, train_set))
    assert len(train_set)+len(test_set) == len(idxs)
    assert np.all(np.logical_or(np.isin(idxs, test_set), np.isin(idxs, train_set)))

@pytest.mark.parametrize(
    "idxs,clusters,train_size,test_size",
    [
        (
            np.arange(30),
            np.array([0,2,0,0,4,4,0,3,1,0,2,1,2,2,0,0,1,1,3,3,3,2,4,4,4,1,1,0,1,3]),
            20,
            5,
        ),
        (
            np.arange(30),
            np.array([0,2,0,0,4,4,0,3,1,0,2,1,2,2,0,0,1,1,3,3,3,2,4,4,4,1,1,0,1,3]),
            5,
            5,
        ),
    ])
def test_cluster_manager_partial(idxs, clusters, train_size, test_size):
    c_m = ClusterManager()
    train_set, test_set = c_m.preserve_cluster_split(idxs, clusters, train_size, test_size)
    assert not np.any(np.isin(train_set, test_set))
    assert not np.any(np.isin(test_set, train_set))
    assert len(train_set) > 0
    assert len(test_set) > 0

@pytest.mark.parametrize(
    "idxs,stratify,clusters,train_size,test_size",
    [
        (
            np.arange(30),
            None,
            np.array([0,2,0,0,4,4,0,3,1,0,2,1,2,2,0,0,1,1,3,3,3,2,4,4,4,1,1,0,1,3]),
            20,
            10,
        ),
    ])
def test_train_test_split_clustered(idxs, stratify, clusters, train_size, test_size):
    if not stratify is None:
        stratify = [stratify]
    if not clusters is None:
        clusters = [clusters]
    train_set, test_set = next(train_test_split(
        idxs,
        train_size = train_size,
        test_size = test_size,
        shuffle = False,
        clusters = clusters,
        stratify = stratify,
        seed = 123,
    ))
    assert not np.any(np.isin(train_set, test_set))
    assert not np.any(np.isin(test_set, train_set))
    assert len(train_set)+len(test_set) == len(idxs)
    assert np.all(np.logical_or(np.isin(idxs, test_set), np.isin(idxs, train_set)))
    check_clusters(
        idxs=idxs,
        clusters=clusters[0],
        train_set=train_set,
        test_set=test_set)

@pytest.mark.parametrize(
    "idxs,stratify,clusters,train_size,test_size",
    [
        (
            np.arange(30),
            np.array([0,1,0,0,0,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            np.array([0,2,0,0,4,4,0,3,1,0,2,1,2,2,0,0,1,1,3,3,3,2,4,4,4,1,1,0,1,3]),
            20,
            10,
        ),
    ])
def test_train_test_split_stratified_clustered(idxs, stratify, clusters, train_size, test_size):
    if not stratify is None:
        stratify = [stratify]
    if not clusters is None:
        clusters = [clusters]
    train_set, test_set = next(train_test_split(
        idxs,
        train_size = train_size,
        test_size = test_size,
        shuffle = False,
        clusters = clusters,
        stratify = stratify,
        seed = 123,
    ))
    assert not np.any(np.isin(train_set, test_set))
    assert not np.any(np.isin(test_set, train_set))
    assert len(train_set)+len(test_set) == len(idxs)
    assert np.all(np.logical_or(np.isin(idxs, test_set), np.isin(idxs, train_set)))
    check_stratification(
        idxs=idxs,
        stratify=stratify[0],
        train_set=train_set,
        test_set=test_set)
    check_clusters(
        idxs=idxs,
        clusters=clusters[0],
        train_set=train_set,
        test_set=test_set)

@pytest.mark.parametrize(
    "idxs,stratify,clusters,train_size,test_size",
    [
        (
            np.arange(30),
            None,
            np.array([0,2,0,0,4,4,0,3,1,0,2,1,2,2,0,0,1,1,3,3,3,2,4,4,4,1,1,0,1,3]),
            20,
            10,
        ),
    ])
def test_train_test_shuffled_split_clustered(idxs, stratify, clusters, train_size, test_size):
    if not stratify is None:
        stratify = [stratify]
    if not clusters is None:
        clusters = [clusters]
    train_set, test_set = next(train_test_split(
        idxs,
        train_size = train_size,
        test_size = test_size,
        shuffle = True,
        clusters = clusters,
        stratify = stratify,
        seed = 123,
    ))
    assert not np.any(np.isin(train_set, test_set))
    assert not np.any(np.isin(test_set, train_set))
    assert len(train_set)+len(test_set) == len(idxs)
    assert np.all(np.logical_or(np.isin(idxs, test_set), np.isin(idxs, train_set)))
    check_clusters(
        idxs=idxs,
        clusters=clusters[0],
        train_set=train_set,
        test_set=test_set)

@pytest.mark.parametrize(
    "idxs,stratify,clusters,train_size,test_size",
    [
        (
            np.arange(30),
            np.array([0,1,0,0,0,0,0,0,1,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
            np.array([0,2,0,0,4,4,0,3,1,0,2,1,2,2,0,0,1,1,3,3,3,2,4,4,4,1,1,0,1,3]),
            20,
            10,
        ),
    ])
def test_train_test_split_shuffled_stratified_clustered(idxs, stratify, clusters, train_size, test_size):
    if not stratify is None:
        stratify = [stratify]
    if not clusters is None:
        clusters = [clusters]
    train_set, test_set = next(train_test_split(
        idxs,
        train_size = train_size,
        test_size = test_size,
        shuffle = True,
        clusters = clusters,
        stratify = stratify,
        seed = 123,
    ))
    assert not np.any(np.isin(train_set, test_set))
    assert not np.any(np.isin(test_set, train_set))
    assert len(train_set)+len(test_set) == len(idxs)
    assert np.all(np.logical_or(np.isin(idxs, test_set), np.isin(idxs, train_set)))
    check_stratification(
        idxs=idxs,
        stratify=stratify[0],
        train_set=train_set,
        test_set=test_set)
    check_clusters(
        idxs=idxs,
        clusters=clusters[0],
        train_set=train_set,
        test_set=test_set)

def check_stratification(idxs, stratify, train_set, test_set):
    for s in np.unique(stratify):
        assert np.isin(idxs[stratify==s], train_set).any()
        assert np.isin(idxs[stratify==s], test_set).any()

def check_clusters(idxs, clusters, train_set, test_set):
    for c in np.unique(clusters):
        if np.isin(idxs[clusters==c], train_set).any():
            assert not np.isin(idxs[clusters==c], test_set).any()
        if np.isin(idxs[clusters==c], test_set).any():
            assert not np.isin(idxs[clusters==c], train_set).any()