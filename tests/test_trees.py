import numpy as np
import pytest

from trees import Binner, apply_prune, fit_tree, prune_path


def grow(X, y, task="classification", **kw):
    b = Binner().fit(X)
    Xb = b.transform(X).astype(np.int64)
    K = int(y.max()) + 1 if task == "classification" else None
    t = fit_tree(Xb, y, b.edges, task, K, **kw)
    return t, b, Xb


def test_binner_threshold_semantics():
    X = np.array([[1.0], [2.0], [3.0], [10.0]])
    b = Binner().fit(X)
    assert b.edges[0].tolist() == [1.5, 2.5, 6.5]
    assert b.transform(X)[:, 0].tolist() == [0, 1, 2, 3]  # bin k <=> x <= edges[k]


def test_interaction_needs_depth_two():
    rs = np.random.RandomState(0)
    X = rs.rand(400, 2)
    y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(int)
    t1, _, Xb = grow(X, y, max_depth=1, min_samples_leaf=1, min_samples_split=2)
    t2, _, _ = grow(X, y, max_depth=2, min_samples_leaf=1, min_samples_split=2)
    acc = lambda t: (t.value[t.apply_binned(Xb)].argmax(1) == y).mean()
    assert acc(t1) < 0.8 and acc(t2) > 0.97  # greedy CART cannot learn pure XOR, but does learn this interaction


def test_regression_tree_recovers_step_function():
    X = np.linspace(0, 1, 200)[:, None]
    y = np.where(X[:, 0] < 0.3, 1.0, 5.0)
    t, _, Xb = grow(X, y, "regression", max_depth=1, min_samples_leaf=1, min_samples_split=2)
    assert np.allclose(t.value[t.apply_binned(Xb)][:, 0], y)
    assert abs(t.thr[0] - 0.3) < 0.01


def test_sample_weights_change_leaf_prediction():
    X = np.zeros((10, 1))
    y = np.array([0] * 5 + [1] * 5)
    t_even, _, Xb = grow(X, y, max_depth=3)
    w = np.where(y == 1, 9.0, 1.0)
    t_w, _, _ = grow(X, y, sample_weight=w, max_depth=3)
    assert t_even.value[0].round(3).tolist() == [0.5, 0.5]
    assert t_w.value[0].round(3).tolist() == [0.1, 0.9]


def test_root_split_matches_sklearn():
    sk = pytest.importorskip("sklearn.tree")
    rs = np.random.RandomState(1)
    X = rs.rand(500, 4)
    y = ((X[:, 2] > 0.4) + (X[:, 0] > 0.7)).astype(int)
    t, b, Xb = grow(X, y, max_depth=1, min_samples_leaf=1, min_samples_split=2)
    ref = sk.DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xb, y)
    assert t.feature[0] == ref.tree_.feature[0]
    assert t.thr_bin[0] + 0.5 == ref.tree_.threshold[0]


def test_min_samples_leaf_respected():
    rs = np.random.RandomState(2)
    X = rs.rand(300, 3)
    y = (X[:, 0] > 0.5).astype(int)
    t, _, _ = grow(X, y, min_samples_leaf=20)
    assert t.count[t.feature == -1].min() >= 20


def test_pruning_path_is_monotone_and_ends_at_root():
    rs = np.random.RandomState(3)
    X = rs.rand(400, 3)
    y = ((X[:, 0] + 0.3 * rs.randn(400)) > 0.5).astype(int)
    t, _, Xb = grow(X, y, max_depth=8, min_samples_leaf=2)
    path = prune_path(t)
    alphas = [a for a, _ in path]
    leaves = [apply_prune(t, m).n_leaves for _, m in path]
    assert alphas == sorted(alphas)
    assert leaves == sorted(leaves, reverse=True) and leaves[-1] == 1 and leaves[0] == t.n_leaves
