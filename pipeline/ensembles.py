"""Ensemble methods built on the from-scratch CART in trees.py (NumPy only, scipy only for NNLS).

Bagging / random forest (committees), AdaBoost (SAMME), gradient boosting (additive model),
ridge regression, stacking and Bayesian model averaging.
"""
import math

import numpy as np
from scipy.optimize import nnls

from trees import apply_prune, fit_tree, prune_path


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Forest:
    """Bagging (max_features=None) or random forest (max_features='sqrt') for either task."""

    def __init__(self, task, binner, n_classes=None, n_trees=100, max_depth=12, min_samples_leaf=3,
                 max_features=None, seed=0):
        self.task, self.binner, self.K = task, binner, n_classes
        self.n_trees, self.max_depth, self.min_leaf, self.mf, self.seed = n_trees, max_depth, min_samples_leaf, max_features, seed

    def fit(self, X, y):
        Xb = self.binner.transform(X).astype(np.int64)
        n, p = Xb.shape
        mf = {"sqrt": max(1, int(np.sqrt(p))), None: None}.get(self.mf, self.mf)
        rng = np.random.RandomState(self.seed)
        self.trees, self.oob_curve = [], []
        out = self.K if self.task == "classification" else 1
        oob_sum, oob_cnt = np.zeros((n, out)), np.zeros(n)
        for b in range(self.n_trees):
            idx = rng.randint(0, n, n)
            w = np.bincount(idx, minlength=n).astype(float)
            t = fit_tree(Xb, y, self.binner.edges, self.task, self.K, w, self.max_depth, self.min_leaf,
                         max(2 * self.min_leaf, 6), mf, rng)
            self.trees.append(t)
            oob = w == 0
            oob_sum[oob] += t.value[t.apply_binned(Xb[oob])]
            oob_cnt[oob] += 1
            m = oob_cnt > 0
            if self.task == "classification":
                self.oob_curve.append(float((oob_sum[m].argmax(1) != y[m]).mean()))
            else:
                self.oob_curve.append(float(np.sqrt(np.mean((oob_sum[m, 0] / oob_cnt[m] - y[m]) ** 2))))
        self.oob_pred = oob_sum / np.maximum(oob_cnt, 1)[:, None]
        self.oob_mask = oob_cnt > 0
        return self

    def tree_values(self, X):
        """(n_trees, n, out) predictions of the individual trees."""
        return np.stack([t.predict_value(X) for t in self.trees])

    def predict_proba(self, X):
        return self.tree_values(X).mean(0)

    def predict(self, X):
        return self.predict_proba(X).argmax(1) if self.task == "classification" else self.tree_values(X).mean(0)[:, 0]

    def predict_std(self, X):
        return self.tree_values(X)[:, :, 0].std(0)


class AdaBoost:
    """Multi-class AdaBoost (SAMME) with depth-limited CART base learners."""

    def __init__(self, binner, n_classes, n_rounds=100, max_depth=3, lr=1.0, seed=0, snapshot_rounds=()):
        self.binner, self.K, self.n_rounds, self.max_depth, self.lr, self.seed = binner, n_classes, n_rounds, max_depth, lr, seed
        self.snapshot_rounds = set(snapshot_rounds)

    def fit(self, X, y):
        Xb = self.binner.transform(X).astype(np.int64)
        n = len(y)
        w = np.full(n, 1.0 / n)
        self.trees, self.alphas, self.errors, self.snapshots = [], [], [], {}
        rng = np.random.RandomState(self.seed)
        for m in range(self.n_rounds):
            if m in self.snapshot_rounds:
                self.snapshots[m] = w.copy()
            t = fit_tree(Xb, y, self.binner.edges, "classification", self.K, w, self.max_depth, 5, 10, None, rng)
            miss = t.value[t.apply_binned(Xb)].argmax(1) != y
            err = float((w * miss).sum() / w.sum())
            err = min(max(err, 1e-10), 1 - 1e-10)
            alpha = self.lr * (np.log((1 - err) / err) + np.log(self.K - 1))
            self.errors.append(err)
            if alpha <= 0:
                break
            self.trees.append(t)
            self.alphas.append(float(alpha))
            w = w * np.exp(alpha * miss)
            w /= w.sum()
        self.snapshots[len(self.trees)] = w.copy()
        self.tau = 1.0
        return self

    def scores(self, X):
        s = np.zeros((len(X), self.K))
        for t, a in zip(self.trees, self.alphas):
            pred = t.predict_value(X).argmax(1)
            s[np.arange(len(X)), pred] += a
        return s

    def calibrate(self, X, y):
        """Temperature-scale the vote scores on held-out data so that probabilities are usable in BMA."""
        s = self.scores(X)
        best, bt = np.inf, 1.0
        for tau in np.exp(np.linspace(np.log(0.05), np.log(20), 60)) * np.mean(self.alphas):
            p = softmax(s / tau)
            nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
            if nll < best:
                best, bt = nll, tau
        self.tau = bt
        return self

    def predict_proba(self, X):
        return softmax(self.scores(X) / self.tau)

    def predict(self, X):
        return self.scores(X).argmax(1)


class GradientBoosting:
    """Additive model F(x) = F0 + lr * sum_m h_m(x), each h_m a small CART fitted to the residuals (squared loss)."""

    def __init__(self, binner, n_stages=400, lr=0.06, max_depth=4, min_samples_leaf=8, subsample=0.8, seed=0):
        self.binner, self.n_stages, self.lr, self.max_depth, self.min_leaf, self.subsample, self.seed = binner, n_stages, lr, max_depth, min_samples_leaf, subsample, seed

    def fit(self, X, y, Xv=None, yv=None):
        Xb = self.binner.transform(X).astype(np.int64)
        n = len(y)
        rng = np.random.RandomState(self.seed)
        self.f0 = float(y.mean())
        F = np.full(n, self.f0)
        Fv = np.full(len(yv), self.f0) if Xv is not None else None
        self.trees, self.train_rmse, self.val_rmse = [], [], []
        for m in range(self.n_stages):
            resid = y - F
            idx = rng.choice(n, int(self.subsample * n), replace=False)
            t = fit_tree(Xb[idx], resid[idx], self.binner.edges, "regression", None, None, self.max_depth,
                         self.min_leaf, 2 * self.min_leaf, None, rng)
            self.trees.append(t)
            F += self.lr * t.value[t.apply_binned(Xb)][:, 0]
            self.train_rmse.append(float(np.sqrt(np.mean((y - F) ** 2))))
            if Xv is not None:
                Fv += self.lr * t.predict_value(Xv)[:, 0]
                self.val_rmse.append(float(np.sqrt(np.mean((yv - Fv) ** 2))))
        if Xv is not None:  # early stopping on validation error
            self.best_stages = int(np.argmin(self.val_rmse)) + 1
        else:
            self.best_stages = self.n_stages
        return self

    def predict(self, X, stages=None):
        s = self.best_stages if stages is None else stages
        F = np.full(len(X), self.f0)
        for t in self.trees[:s]:
            F += self.lr * t.predict_value(X)[:, 0]
        return F

    def staged(self, x):
        """Cumulative prediction for one sample after each stage (for the build-up visual)."""
        out, F = [self.f0], self.f0
        for t in self.trees[:self.best_stages]:
            F += self.lr * float(t.predict_value(x[None, :])[0, 0])
            out.append(F)
        return out


class Ridge:
    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        A = Z.T @ Z + self.lam * np.diag([0] + [1] * X.shape[1])
        self.coef = np.linalg.solve(A, Z.T @ y)
        return self

    def predict(self, X):
        return np.c_[np.ones(len(X)), (X - self.mu) / self.sd] @ self.coef


def fit_pruned_cart(Xb, y, Xvb, yv, binner, K, max_depth=12, min_leaf=3, seed=0):
    """Grow a deep CART, compute the cost-complexity path and choose alpha by validation accuracy."""
    full = fit_tree(Xb, y, binner.edges, "classification", K, None, max_depth, min_leaf, 2 * min_leaf, None, np.random.RandomState(seed))
    path = prune_path(full)
    rows, best = [], (-1, None)
    for alpha, mask in path:
        t = apply_prune(full, mask)
        acc = float((t.value[t.apply_binned(Xvb)].argmax(1) == yv).mean())
        rows.append({"alpha": alpha, "leaves": t.n_leaves, "val_acc": acc})
        if acc >= best[0] - 1e-12 and (acc > best[0] + 1e-12 or t.n_leaves < best[1].n_leaves):
            best = (acc, t)
    return best[1], full, rows


# ------------------------------------------------------------------ combining models
class SoftmaxStack:
    """Multinomial logistic regression (L2) on concatenated log-probabilities of the base models."""

    def __init__(self, l2=1e-2, iters=600, lr=0.5):
        self.l2, self.iters, self.lr = l2, iters, lr

    def _feat(self, P):
        return np.log(np.clip(np.concatenate(P, axis=1), 1e-4, 1))

    def fit(self, P, y, K):
        Z = self._feat(P)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        Z = np.c_[np.ones(len(Z)), (Z - self.mu) / self.sd]
        W = np.zeros((Z.shape[1], K))
        Y = np.eye(K)[y]
        for _ in range(self.iters):
            G = Z.T @ (softmax(Z @ W) - Y) / len(Z) + self.l2 * np.r_[np.zeros((1, K)), W[1:]]
            W -= self.lr * G
        self.W = W
        return self

    def predict_proba(self, P):
        Z = (self._feat(P) - self.mu) / self.sd
        return softmax(np.c_[np.ones(len(Z)), Z] @ self.W)


def bma_weights_classification(prob_list, y, tempering=1.0):
    """Posterior model weights: p(M|D) proportional to p(D|M) with uniform prior; likelihood = held-out log-likelihood."""
    ll = np.array([np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).sum() for P in prob_list]) * tempering
    ll -= ll.max()
    w = np.exp(ll)
    return w / w.sum()


def bma_regression(preds_val, y_val, preds, tempering=1.0):
    """Gaussian BMA: each model is N(f_m(x), s_m^2) with s_m from validation residuals; weights from validation likelihood."""
    s = np.array([np.sqrt(np.mean((y_val - p) ** 2)) for p in preds_val])
    ll = np.array([(-0.5 * np.log(2 * np.pi * sm ** 2) - 0.5 * ((y_val - p) / sm) ** 2).sum() for p, sm in zip(preds_val, s)]) * tempering
    ll -= ll.max()
    w = np.exp(ll)
    w /= w.sum()
    P = np.stack(preds)
    mean = (w[:, None] * P).sum(0)
    between = (w[:, None] * (P - mean) ** 2).sum(0)
    within = (w * s ** 2).sum()
    return w, s, mean, np.sqrt(between + within), np.sqrt(between)


def nnls_stack(preds_val, y_val):
    """Non-negative least squares stacking weights (sum-normalised) for regression members."""
    A = np.stack(preds_val, axis=1)
    w, _ = nnls(A, y_val)
    return w / w.sum()


def conformal_quantile(scores, alpha):
    """Split-conformal quantile: the ceil((n+1)(1-alpha))/n empirical quantile of calibration scores."""
    n = len(scores)
    return float(np.quantile(scores, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
