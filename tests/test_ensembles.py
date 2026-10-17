import numpy as np

import ensembles as E
from trees import Binner


def blobs(n=600, seed=0):
    rs = np.random.RandomState(seed)
    X = rs.randn(n, 4)
    score = X[:, 0] + 0.7 * X[:, 1] + 0.5 * rs.randn(n)
    return X, np.digitize(score, [-0.5, 0.8])


def test_adaboost_improves_on_a_single_weak_tree():
    X, y = blobs()
    Xt, yt, Xv, yv = X[:400], y[:400], X[400:], y[400:]
    b = Binner().fit(Xt)
    one = E.AdaBoost(b, 3, n_rounds=1, max_depth=1).fit(Xt, yt)
    many = E.AdaBoost(b, 3, n_rounds=60, max_depth=1).fit(Xt, yt)
    assert (many.predict(Xv) == yv).mean() > (one.predict(Xv) == yv).mean() + 0.03
    assert all(a > 0 for a in many.alphas)


def test_adaboost_weights_are_a_distribution_and_focus_on_errors():
    X, y = blobs()
    b = Binner().fit(X)
    m = E.AdaBoost(b, 3, n_rounds=10, max_depth=1, snapshot_rounds=[0, 5]).fit(X, y)
    for w in m.snapshots.values():
        assert abs(w.sum() - 1) < 1e-9
    assert m.snapshots[5].max() > m.snapshots[0].max()


def test_gradient_boosting_is_an_additive_model():
    rs = np.random.RandomState(0)
    X = rs.rand(500, 3)
    y = np.sin(4 * X[:, 0]) + X[:, 1] + 0.1 * rs.randn(500)
    b = Binner().fit(X[:350])
    g = E.GradientBoosting(b, n_stages=80, lr=0.1, max_depth=3).fit(X[:350], y[:350], X[350:], y[350:])
    assert g.train_rmse[-1] < g.train_rmse[0]
    manual = g.f0 + g.lr * sum(t.predict_value(X[:5])[:, 0] for t in g.trees[:g.best_stages])
    assert np.allclose(g.predict(X[:5]), manual)  # F(x) = F0 + lr * sum of trees
    assert abs(g.staged(X[0])[-1] - g.predict(X[:1])[0]) < 1e-9
    assert 1 <= g.best_stages <= 80


def test_forest_oob_is_close_to_holdout_error():
    X, y = blobs(900, 1)
    Xt, yt, Xv, yv = X[:600], y[:600], X[600:], y[600:]
    f = E.Forest("classification", Binner().fit(Xt), 3, n_trees=60, max_depth=8).fit(Xt, yt)
    assert abs(f.oob_curve[-1] - (f.predict(Xv) != yv).mean()) < 0.07
    assert f.oob_curve[-1] <= f.oob_curve[5] + 0.02  # more trees do not hurt


def test_bma_weights_sum_to_one_and_prefer_the_better_model():
    y = np.array([0, 1, 0, 1, 1, 0] * 20)
    good = np.eye(2)[y] * 0.9 + 0.05
    bad = np.full((len(y), 2), 0.5)
    w = E.bma_weights_classification([good, bad], y)
    assert abs(w.sum() - 1) < 1e-9 and w[0] > 0.99
    wt = E.bma_weights_classification([good, bad], y, tempering=0.01)
    assert wt[1] > w[1]  # tempering spreads the weight


def test_regression_bma_mean_is_weighted_average():
    rs = np.random.RandomState(0)
    y = rs.randn(200)
    p1, p2 = y + 0.1 * rs.randn(200), y + 0.5 * rs.randn(200)
    w, s, mean, sd, between = E.bma_regression([p1, p2], y, [p1, p2])
    assert abs(w.sum() - 1) < 1e-9 and w[0] > w[1]
    assert np.allclose(mean, w[0] * p1 + w[1] * p2)
    assert (sd >= between - 1e-12).all()


def test_nnls_stack_weights_nonnegative_and_normalised():
    rs = np.random.RandomState(0)
    y = rs.rand(100) + 5
    w = E.nnls_stack([y + 0.1 * rs.randn(100), y + 0.4 * rs.randn(100), rs.rand(100) + 5], y)
    assert (w >= 0).all() and abs(w.sum() - 1) < 1e-9


def test_conformal_interval_has_the_promised_coverage():
    rs = np.random.RandomState(0)
    cal, test = np.abs(rs.randn(500)), np.abs(rs.randn(20000))
    for alpha in (0.2, 0.1, 0.05):
        q = E.conformal_quantile(cal, alpha)
        assert (test <= q).mean() >= 1 - alpha - 0.025


def test_ridge_recovers_linear_relationship():
    rs = np.random.RandomState(0)
    X = rs.randn(300, 3)
    y = 2 * X[:, 0] - X[:, 1] + 0.5 + 0.01 * rs.randn(300)
    assert np.sqrt(np.mean((E.Ridge(0.01).fit(X, y).predict(X) - y) ** 2)) < 0.05
