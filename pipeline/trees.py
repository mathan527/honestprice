"""CART decision trees implemented from scratch (NumPy only).

* classification (weighted Gini) and regression (weighted squared error)
* histogram-based exact split search (features are binned once; low-cardinality features keep every value)
* sample weights (needed by AdaBoost) and per-split random feature subsets (needed by random forests)
* minimal cost-complexity pruning (weakest-link) for classification trees
"""
import numpy as np


class Binner:
    """Maps raw features to small integer bins; bin k <=> x <= edges[k]."""

    def __init__(self, max_bins=64):
        self.max_bins = max_bins

    def fit(self, X):
        self.edges = []
        for j in range(X.shape[1]):
            u = np.unique(X[:, j])
            if len(u) <= self.max_bins:
                e = (u[:-1] + u[1:]) / 2.0
            else:
                e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, self.max_bins + 1)[1:-1]))
            self.edges.append(e)
        return self

    def transform(self, X):
        out = np.empty(X.shape, dtype=np.int16)
        for j, e in enumerate(self.edges):
            out[:, j] = np.searchsorted(e, X[:, j], side="left")
        return out


class Tree:
    """A fitted tree stored as flat arrays. value: class probabilities (K) or mean (1)."""

    def __init__(self, task, n_out):
        self.task, self.n_out = task, n_out
        self.feature, self.thr_bin, self.thr, self.left, self.right = [], [], [], [], []
        self.value, self.weight, self.count, self.leaf_err = [], [], [], []

    def _new(self, value, w, n, err):
        self.feature.append(-1); self.thr_bin.append(-1); self.thr.append(np.nan)
        self.left.append(-1); self.right.append(-1)
        self.value.append(value); self.weight.append(w); self.count.append(n); self.leaf_err.append(err)
        return len(self.feature) - 1

    def finalize(self):
        for name in ("feature", "thr_bin", "thr", "left", "right", "weight", "count", "leaf_err"):
            setattr(self, name, np.asarray(getattr(self, name)))
        self.value = np.asarray(self.value, dtype=float).reshape(len(self.feature), -1)
        return self

    @property
    def n_nodes(self):
        return len(self.feature)

    @property
    def n_leaves(self):
        return int((self.feature == -1).sum())

    def apply_binned(self, Xb):
        node = np.zeros(len(Xb), dtype=np.int64)
        while True:
            f = self.feature[node]
            act = f >= 0
            if not act.any():
                return node
            a = np.where(act)[0]
            go_left = Xb[a, f[a]] <= self.thr_bin[node[a]]
            node[a] = np.where(go_left, self.left[node[a]], self.right[node[a]])

    def apply(self, X):
        node = np.zeros(len(X), dtype=np.int64)
        while True:
            f = self.feature[node]
            act = f >= 0
            if not act.any():
                return node
            a = np.where(act)[0]
            go_left = X[a, f[a]] <= self.thr[node[a]]
            node[a] = np.where(go_left, self.left[node[a]], self.right[node[a]])

    def predict_value(self, X):
        return self.value[self.apply(X)]

    def path(self, x):
        """Nodes visited by one sample (for explanations)."""
        node, out = 0, [0]
        while self.feature[node] >= 0:
            node = self.left[node] if x[self.feature[node]] <= self.thr[node] else self.right[node]
            out.append(int(node))
        return out

    def to_dict(self, digits=4):
        return {"f": self.feature.tolist(), "t": [None if np.isnan(t) else round(float(t), 7) for t in self.thr],
                "l": self.left.tolist(), "r": self.right.tolist(),
                "v": np.round(self.value, digits).tolist() if self.n_out > 1 else np.round(self.value[:, 0], digits).tolist(),
                "n": self.count.tolist()}


def fit_tree(Xb, y, edges, task="classification", n_classes=None, sample_weight=None, max_depth=8,
             min_samples_leaf=5, min_samples_split=10, max_features=None, rng=None, min_gain=1e-12):
    """Grow a CART tree on binned data Xb (n, p). y: int labels (classification) or floats (regression)."""
    n, p = Xb.shape
    w_all = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    clf = task == "classification"
    K = n_classes if clf else 1
    nb = [len(e) + 1 for e in edges]
    tree = Tree(task, K)
    rng = rng or np.random.RandomState(0)

    def leaf_stats(idx):
        w = w_all[idx]
        W = w.sum()
        if clf:
            h = np.bincount(y[idx], weights=w, minlength=K)
            val = h / W if W > 0 else np.full(K, 1.0 / K)
            return val, W, (W - h.max()) if W > 0 else 0.0
        mu = (w * y[idx]).sum() / W if W > 0 else 0.0
        return np.array([mu]), W, float((w * (y[idx] - mu) ** 2).sum())

    def best_split(idx, feats):
        w, yy = w_all[idx], y[idx]
        W = w.sum()
        best = (min_gain, -1, -1)
        if clf:
            ph = np.bincount(yy, weights=w, minlength=K)
            parent = W - (ph ** 2).sum() / W  # W * gini
        else:
            S, S2 = (w * yy).sum(), (w * yy * yy).sum()
            parent = S2 - S * S / W
        for j in feats:
            if nb[j] < 2:
                continue
            b = Xb[idx, j]
            cnt = np.cumsum(np.bincount(b, minlength=nb[j]))[:-1]
            if clf:
                h = np.bincount(b * K + yy, weights=w, minlength=nb[j] * K).reshape(nb[j], K)
                L = np.cumsum(h, axis=0)[:-1]
                R = ph[None, :] - L
                WL, WR = L.sum(1), R.sum(1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    child = (WL - (L ** 2).sum(1) / WL) + (WR - (R ** 2).sum(1) / WR)
            else:
                sw = np.cumsum(np.bincount(b, weights=w, minlength=nb[j]))[:-1]
                s1 = np.cumsum(np.bincount(b, weights=w * yy, minlength=nb[j]))[:-1]
                s2 = np.cumsum(np.bincount(b, weights=w * yy * yy, minlength=nb[j]))[:-1]
                WL, WR = sw, W - sw
                with np.errstate(divide="ignore", invalid="ignore"):
                    child = (s2 - s1 * s1 / WL) + ((S2 - s2) - (S - s1) ** 2 / WR)
            ok = (cnt >= min_samples_leaf) & ((len(idx) - cnt) >= min_samples_leaf) & (WL > 0) & (WR > 0)
            if not ok.any():
                continue
            gain = np.where(ok, parent - child, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best[0]:
                best = (float(gain[k]), j, k)
        return best

    def grow(idx, depth):
        val, W, err = leaf_stats(idx)
        node = tree._new(val, W, len(idx), err)
        pure = (err <= 1e-12) if clf else (err <= 1e-12)
        if depth >= max_depth or len(idx) < min_samples_split or pure:
            return node
        feats = np.arange(n_features) if max_features is None else rng.choice(n_features, max_features, replace=False)
        gain, j, k = best_split(idx, feats)
        if j < 0:
            return node
        go_left = Xb[idx, j] <= k
        tree.feature[node], tree.thr_bin[node] = j, k
        tree.thr[node] = edges[j][k]
        tree.left[node] = grow(idx[go_left], depth + 1)
        tree.right[node] = grow(idx[~go_left], depth + 1)
        return node

    n_features = p
    grow(np.arange(n)[w_all > 0], 0)
    return tree.finalize()


# ------------------------------------------------------------------ cost-complexity pruning
def prune_path(tree):
    """Minimal cost-complexity pruning (Breiman et al.). Returns [(alpha, leaf_mask_after_pruning)].
    Node cost R(t) is the weighted misclassification mass of t acting as a leaf (classification trees)."""
    n = tree.n_nodes
    total = tree.weight[0]
    R = tree.leaf_err / total
    left, right = tree.left.copy(), tree.right.copy()
    is_leaf = tree.feature == -1
    steps = [(0.0, is_leaf.copy())]

    def sub(t):
        if is_leaf[t]:
            return R[t], 1
        a, la = sub(left[t])
        b, lb = sub(right[t])
        return a + b, la + lb

    while not is_leaf[0]:
        best_g, best_t = np.inf, -1
        stack = [0]
        while stack:
            t = stack.pop()
            if is_leaf[t]:
                continue
            rs, nl = sub(t)
            g = (R[t] - rs) / (nl - 1)
            if g < best_g - 1e-15:
                best_g, best_t = g, t
            stack += [left[t], right[t]]
        is_leaf[best_t] = True
        steps.append((float(max(best_g, 0.0)), is_leaf.copy()))
    return steps


def apply_prune(tree, is_leaf_mask):
    """Return a new (re-indexed) Tree where nodes flagged as leaves are collapsed."""
    new = Tree(tree.task, tree.n_out)
    mp = {}

    def copy(t):
        i = new._new(tree.value[t], tree.weight[t], tree.count[t], tree.leaf_err[t])
        if not is_leaf_mask[t] and tree.feature[t] >= 0:
            new.feature[i], new.thr_bin[i], new.thr[i] = int(tree.feature[t]), int(tree.thr_bin[t]), tree.thr[t]
            new.left[i] = copy(tree.left[t])
            new.right[i] = copy(tree.right[t])
        return i

    copy(0)
    return new.finalize()
