"""Train, evaluate, cross-check and export every HonestPrice model.

Run:  python pipeline/build.py        ->  web/data/model.json (+ web/data/golden.json for the JS parity test)
"""
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import data as D  # noqa: E402
import ensembles as E  # noqa: E402
from trees import Binner, fit_tree  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "data"
SEED = 7
N_TREES, EXPORT_DEPTH = 60, 10
t_start = time.time()
rnd = lambda a, n=4: np.round(np.asarray(a, dtype=float), n).tolist()

# ---------------------------------------------------------------- data
df, info = D.load_clean()
sp = D.make_split(df, SEED)
parts = {k: df.iloc[v].reset_index(drop=True) for k, v in sp.items()}
fb = D.FeatureBuilder().fit(parts["train"])
X = {k: fb.transform(v, train=(k == "train")) for k, v in parts.items()}
yr = {k: v["log_price"].values for k, v in parts.items()}
price = {k: v["selling_price"].values.astype(float) for k, v in parts.items()}
q1, q2 = np.quantile(yr["train"], [1 / 3, 2 / 3])
tier = lambda lp: (lp > q1).astype(int) + (lp > q2).astype(int)
yc = {k: tier(v) for k, v in yr.items()}
K = 3
binner = Binner().fit(X["train"])
Xb = {k: binner.transform(v).astype(np.int64) for k, v in X.items()}
print(info, {k: len(v) for k, v in parts.items()}, "tier cut-offs (INR):", round(math.exp(q1)), round(math.exp(q2)))

# ================================================================ TASK A: classification (price tier)
cls = {}
cart, cart_full, prune_rows = E.fit_pruned_cart(Xb["train"], yc["train"], Xb["val"], yc["val"], binner, K)
bag_c = E.Forest("classification", binner, K, N_TREES, 12, 3, None, SEED).fit(X["train"], yc["train"])
rf_c = E.Forest("classification", binner, K, N_TREES, 12, 3, "sqrt", SEED).fit(X["train"], yc["train"])
SNAP = [0, 1, 2, 4, 9, 19, 49, 99]
ada = E.AdaBoost(binner, K, 120, 3, 1.0, SEED, SNAP).fit(X["train"], yc["train"]).calibrate(X["val"], yc["val"])
members_c = {"CART (pruned)": lambda Z: cart.predict_value(Z), "Bagging": bag_c.predict_proba,
             "Random forest": rf_c.predict_proba, "AdaBoost": ada.predict_proba}
Pv = {n: f(X["val"]) for n, f in members_c.items()}
Pt = {n: f(X["test"]) for n, f in members_c.items()}
names_c = list(members_c)
stack_c = E.SoftmaxStack().fit([Pv[n] for n in names_c], yc["val"], K)
w_bma_c = E.bma_weights_classification([Pv[n] for n in names_c], yc["val"], 1.0)
temper = 1 / math.sqrt(len(yc["val"]))
w_bma_ct = E.bma_weights_classification([Pv[n] for n in names_c], yc["val"], temper)
P_stack_t = stack_c.predict_proba([Pt[n] for n in names_c])
P_bma_t = sum(w * Pt[n] for w, n in zip(w_bma_c, names_c))
P_bmat_t = sum(w * Pt[n] for w, n in zip(w_bma_ct, names_c))
P_avg_t = sum(Pt[n] for n in names_c) / len(names_c)


def clf_metrics(P, y):
    pred = P.argmax(1)
    cm = np.zeros((K, K), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    f1 = []
    for k in range(K):
        tp = cm[k, k]
        pr, rc = tp / max(cm[:, k].sum(), 1), tp / max(cm[k].sum(), 1)
        f1.append(2 * pr * rc / max(pr + rc, 1e-12))
    return {"acc": float((pred == y).mean()), "macro_f1": float(np.mean(f1)),
            "log_loss": float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean()),
            "off_by_two": float((np.abs(pred - y) == 2).mean()), "confusion": cm.tolist()}


cls["results"] = {n: clf_metrics(Pt[n], yc["test"]) for n in names_c}
cls["results"].update({"Simple average": clf_metrics(P_avg_t, yc["test"]), "BMA (exact posterior)": clf_metrics(P_bma_t, yc["test"]),
                       "BMA (tempered)": clf_metrics(P_bmat_t, yc["test"]), "Stacking (softmax)": clf_metrics(P_stack_t, yc["test"])})
cls["results"]["Majority-class baseline"] = clf_metrics(np.eye(K)[np.full(len(yc["test"]), np.bincount(yc["train"]).argmax())] * 0.98 + 0.01, yc["test"])
cls["bma_weights"] = {"exact": dict(zip(names_c, rnd(w_bma_c))), "tempered": dict(zip(names_c, rnd(w_bma_ct)))}
# committee disagreement: does a split vote signal a wrong answer?
votes = np.stack([Pt[n].argmax(1) for n in names_c])
unan = (votes == votes[0]).all(0)
pred_s = P_stack_t.argmax(1)
conf = P_stack_t.max(1)
order = np.argsort(-conf)
sel = [{"coverage": c, "acc": float((pred_s[order[:int(c * len(order))]] == yc["test"][order[:int(c * len(order))]]).mean())} for c in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)]
cls["disagreement"] = {"unanimous_share": float(unan.mean()), "acc_unanimous": float((pred_s[unan] == yc["test"][unan]).mean()),
                       "acc_split": float((pred_s[~unan] == yc["test"][~unan]).mean()), "selective": sel}
cls["oob_curve"] = {"Bagging": rnd(bag_c.oob_curve), "Random forest": rnd(rf_c.oob_curve)}
cls["prune_path"] = prune_rows
cls["ada"] = {"errors": rnd(ada.errors), "alphas": rnd(ada.alphas), "n": len(ada.trees)}
print("classification test:", {n: round(m["acc"], 3) for n, m in cls["results"].items()})

# ================================================================ TASK B: regression (log price)
reg = {}
ridge = E.Ridge(3.0).fit(X["train"], yr["train"])
best_cart = None
for md in (4, 6, 8, 10):
    for ml in (5, 10, 20):
        t = fit_tree(Xb["train"], yr["train"], binner.edges, "regression", None, None, md, ml, 2 * ml, None, np.random.RandomState(SEED))
        r = np.sqrt(np.mean((t.value[t.apply_binned(Xb["val"])][:, 0] - yr["val"]) ** 2))
        if best_cart is None or r < best_cart[0]:
            best_cart = (r, t, md, ml)
cart_r = best_cart[1]
bag_r = E.Forest("regression", binner, None, N_TREES, 14, 3, None, SEED).fit(X["train"], yr["train"])
rf_r = E.Forest("regression", binner, None, N_TREES, 14, 3, "sqrt", SEED).fit(X["train"], yr["train"])
gbm = E.GradientBoosting(binner, 500, 0.06, 4, 8, 0.8, SEED).fit(X["train"], yr["train"], X["val"], yr["val"])
members_r = {"Ridge (linear)": ridge.predict, "CART": lambda Z: cart_r.predict_value(Z)[:, 0], "Bagging": bag_r.predict,
             "Random forest": rf_r.predict, "Gradient boosting": gbm.predict}
names_r = list(members_r)
pv = {n: f(X["val"]) for n, f in members_r.items()}
pt = {n: f(X["test"]) for n, f in members_r.items()}
pc = {n: f(X["cal"]) for n, f in members_r.items()}
PV, PT, PC = ([d[n] for n in names_r] for d in (pv, pt, pc))
w_b, s_b, _, _, _ = E.bma_regression(PV, yr["val"], PV, 1.0)
w_bt, s_bt, _, _, _ = E.bma_regression(PV, yr["val"], PV, temper)
w_nn = E.nnls_stack(PV, yr["val"])


def combo(w, preds):
    return sum(wi * p for wi, p in zip(w, preds))


def reg_metrics(p_log, y_log, y_price):
    pr = np.exp(p_log)
    ss = ((y_price - pr) ** 2).sum()
    return {"rmse_log": float(np.sqrt(np.mean((p_log - y_log) ** 2))), "mae_inr": float(np.abs(pr - y_price).mean()),
            "median_ape": float(np.median(np.abs(pr - y_price) / y_price)),
            "r2_price": float(1 - ss / ((y_price - y_price.mean()) ** 2).sum()),
            "r2_log": float(1 - ((p_log - y_log) ** 2).sum() / ((y_log - y_log.mean()) ** 2).sum())}


reg["results"] = {n: reg_metrics(pt[n], yr["test"], price["test"]) for n in names_r}
point = {"Simple average": np.mean(PT, 0), "BMA (exact posterior)": combo(w_b, PT), "BMA (tempered)": combo(w_bt, PT), "NNLS stacking": combo(w_nn, PT)}
reg["results"].update({n: reg_metrics(p, yr["test"], price["test"]) for n, p in point.items()})
reg["results"]["Mean baseline"] = reg_metrics(np.full(len(yr["test"]), yr["train"].mean()), yr["test"], price["test"])
reg["weights"] = {"bma_exact": dict(zip(names_r, rnd(w_b))), "bma_tempered": dict(zip(names_r, rnd(w_bt))), "nnls": dict(zip(names_r, rnd(w_nn)))}
reg["gbm_curve"] = {"train": rnd(gbm.train_rmse), "val": rnd(gbm.val_rmse), "best": gbm.best_stages}
reg["oob_curve"] = {"Bagging": rnd(bag_r.oob_curve), "Random forest": rnd(rf_r.oob_curve)}
print("regression test RMSE(log):", {n: round(m["rmse_log"], 3) for n, m in reg["results"].items()})

# ---------------------------------------------------------------- honest error bars (conformal prediction)
FINAL_W = w_bt  # tempered BMA is the production point predictor
_, _, mean_c, sd_c, between_c = E.bma_regression(PV, yr["val"], PC, temper)
mean_t_ = combo(FINAL_W, PT)
mean_c_ = combo(FINAL_W, PC)
between_t = np.sqrt((FINAL_W[:, None] * (np.stack(PT) - mean_t_) ** 2).sum(0))
between_cal = np.sqrt((FINAL_W[:, None] * (np.stack(PC) - mean_c_) ** 2).sum(0))
rf_sd_t, rf_sd_c = rf_r.predict_std(X["test"]), rf_r.predict_std(X["cal"])
FLOOR = 0.05
sig_t, sig_c = np.sqrt(rf_sd_t ** 2 + between_t ** 2) + FLOOR, np.sqrt(rf_sd_c ** 2 + between_cal ** 2) + FLOOR


conf_q = E.conformal_quantile


levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
res_c = np.abs(yr["cal"] - mean_c_)
res_t = np.abs(yr["test"] - mean_t_)
sd_bma_t = E.bma_regression(PV, yr["val"], PT, temper)[3]
from scipy.stats import norm  # noqa: E402
iv = {"constant-width conformal": [], "normalised conformal (HonestPrice)": [], "Gaussian BMA (no guarantee)": []}
for lv in levels:
    a = 1 - lv
    qa, qn = conf_q(res_c, a), conf_q(res_c / sig_c, a)
    cov = lambda half: float((res_t <= half).mean())
    wid = lambda half: float(np.median(np.exp(2 * half)))  # upper/lower price ratio
    iv["constant-width conformal"].append({"level": lv, "coverage": cov(qa), "width": wid(qa)})
    iv["normalised conformal (HonestPrice)"].append({"level": lv, "coverage": cov(qn * sig_t), "width": wid(qn * sig_t)})
    iv["Gaussian BMA (no guarantee)"].append({"level": lv, "coverage": cov(norm.ppf(1 - a / 2) * sd_bma_t), "width": wid(norm.ppf(1 - a / 2) * sd_bma_t)})
qn_final = {str(lv): conf_q(res_c / sig_c, 1 - lv) for lv in (0.8, 0.9, 0.95)}
half90 = qn_final["0.9"] * sig_t
terc = np.digitize(mean_t_, np.quantile(mean_t_, [1 / 3, 2 / 3]))
by_tier = [{"tier": D.TIERS[k], "coverage90": float((res_t[terc == k] <= half90[terc == k]).mean()), "n": int((terc == k).sum()),
            "median_ratio": float(np.median(np.exp(2 * half90[terc == k])))} for k in range(3)]
reg["intervals"] = {"methods": iv, "by_tier": by_tier, "q": qn_final, "floor": FLOOR, "n_cal": len(res_c)}
print("90% coverage:", {m: round(v[4]["coverage"], 3) for m, v in iv.items()}, "median ratio:", {m: round(v[4]["width"], 2) for m, v in iv.items()})

# ---------------------------------------------------------------- permutation importance (gradient boosting, test set)
base = np.sqrt(np.mean((gbm.predict(X["test"]) - yr["test"]) ** 2))
rs = np.random.RandomState(0)
imp = []
for j, nm in enumerate(fb.names):
    d = []
    for _ in range(5):
        Z = X["test"].copy()
        Z[:, j] = rs.permutation(Z[:, j])
        d.append(np.sqrt(np.mean((gbm.predict(Z) - yr["test"]) ** 2)) - base)
    imp.append(float(np.mean(d)))
importance = sorted(zip(fb.names, imp), key=lambda t: -t[1])

# ---------------------------------------------------------------- cross-check against scikit-learn
sk = {}
try:
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import Ridge as SkRidge
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    pth = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X["train"], yc["train"]).ccp_alphas
    bestv = max(((DecisionTreeClassifier(ccp_alpha=a, random_state=0, min_samples_leaf=3).fit(X["train"], yc["train"]).score(X["val"], yc["val"]), a) for a in pth[::max(1, len(pth) // 40)]))
    skt = DecisionTreeClassifier(ccp_alpha=bestv[1], random_state=0, min_samples_leaf=3).fit(X["train"], yc["train"])
    skrf = RandomForestClassifier(N_TREES, max_depth=12, min_samples_leaf=3, random_state=0).fit(X["train"], yc["train"])
    skada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3, min_samples_leaf=5), n_estimators=120, algorithm="SAMME", random_state=0).fit(X["train"], yc["train"])
    skrfr = RandomForestRegressor(N_TREES, max_depth=14, min_samples_leaf=3, max_features="sqrt", random_state=0).fit(X["train"], yr["train"])
    skg = GradientBoostingRegressor(n_estimators=gbm.best_stages, learning_rate=0.06, max_depth=4, min_samples_leaf=8, subsample=0.8, random_state=0).fit(X["train"], yr["train"])
    rm = lambda p: float(np.sqrt(np.mean((p - yr["test"]) ** 2)))
    sk = {"classification_acc": {"CART (pruned)": [cls["results"]["CART (pruned)"]["acc"], float(skt.score(X["test"], yc["test"]))],
                                 "Random forest": [cls["results"]["Random forest"]["acc"], float(skrf.score(X["test"], yc["test"]))],
                                 "AdaBoost": [cls["results"]["AdaBoost"]["acc"], float(skada.score(X["test"], yc["test"]))]},
          "regression_rmse_log": {"Ridge (linear)": [reg["results"]["Ridge (linear)"]["rmse_log"], rm(SkRidge(3.0).fit(X["train"], yr["train"]).predict(X["test"]))],
                                  "Random forest": [reg["results"]["Random forest"]["rmse_log"], rm(skrfr.predict(X["test"]))],
                                  "Gradient boosting": [reg["results"]["Gradient boosting"]["rmse_log"], rm(skg.predict(X["test"]))]}}
    # exact algorithmic check: identical root split on the same data
    from sklearn.tree import DecisionTreeClassifier as DT
    sroot = DT(max_depth=1, random_state=0).fit(Xb["train"], yc["train"])
    oroot = fit_tree(Xb["train"], yc["train"], binner.edges, "classification", K, None, 1, 1, 2)
    sk["root_split_match"] = bool(sroot.tree_.feature[0] == oroot.feature[0] and sroot.tree_.threshold[0] == oroot.thr_bin[0] + 0.5)
    print("sklearn cross-check:", sk)
except Exception as e:  # pragma: no cover
    sk = {"error": str(e)}
    print("sklearn unavailable:", e)

# ---------------------------------------------------------------- export
def trees_export(trs, **kw):
    return [t.to_dict() for t in trs]


# a leaner copy of the forests for the browser: first N_TREES trees, shallow-capped by min leaf of the training run
cart_dict = cart.to_dict()
models = {
    "features": fb.names,
    "encoder": {"brands": fb.brands, "prior": round(fb.prior, 8), "brand_te": {k: round(v, 7) for k, v in fb.brand_te.items()},
                "model_te": {k: round(v, 7) for k, v in fb.model_te.items()}, "name_te": {k: round(v, 7) for k, v in fb.name_te.items()},
                "engine_med": fb.engine_med, "model_n": fb.model_n, "owner": D.OWNER, "ref_year": D.REF_YEAR},
    "tiers": {"names": D.TIERS, "cuts_inr": [round(math.exp(q1)), round(math.exp(q2))]},
    "classification": {
        "names": names_c,
        "cart": cart_dict,
        "cart_full_leaves": cart_full.n_leaves,
        "bagging": trees_export(bag_c.trees), "rf": trees_export(rf_c.trees),
        "adaboost": {"trees": trees_export(ada.trees), "alphas": rnd(ada.alphas, 4), "tau": float(ada.tau)},
        "stack": {"W": rnd(stack_c.W, 4), "mu": rnd(stack_c.mu, 4), "sd": rnd(stack_c.sd, 4)},
        "bma": {"exact": rnd(w_bma_c), "tempered": rnd(w_bma_ct)},
    },
    "regression": {
        "names": names_r,
        "ridge": {"coef": rnd(ridge.coef, 5), "mu": rnd(ridge.mu, 5), "sd": rnd(ridge.sd, 5)},
        "cart": cart_r.to_dict(), "bagging": trees_export(bag_r.trees), "rf": trees_export(rf_r.trees),
        "gbm": {"f0": gbm.f0, "lr": gbm.lr, "stages": gbm.best_stages, "trees": trees_export(gbm.trees[:gbm.best_stages])},
        "weights": {"final": rnd(FINAL_W), "bma_exact": rnd(w_b), "nnls": rnd(w_nn)}, "floor": FLOOR, "q": qn_final,
    },
}
# AdaBoost sample-weight snapshots on a 2-D projection (age vs log km) for the "weights grow on hard cases" visual
pick = np.random.RandomState(1).choice(len(yc["train"]), 300, replace=False)
snaps = {str(k): rnd(v[pick] / v[pick].max(), 3) for k, v in sorted(ada.snapshots.items())}
ada_view = {"age": rnd(X["train"][pick, 0], 1), "log_km": rnd(X["train"][pick, 1], 2), "tier": yc["train"][pick].tolist(), "snapshots": snaps}

# example cars from the test set
ex = []
for i in np.random.RandomState(5).choice(len(parts["test"]), 6, replace=False):
    r = parts["test"].iloc[i]
    ex.append({"brand": r.brand, "model": r.model, "age": int(r.age), "km": int(r.km_driven), "fuel": r.fuel, "seller": r.seller_type,
               "transmission": r.transmission, "owner": r.owner, "name": r["name"], "actual": int(r.selling_price)})
catalog = {}
for (b, m), g in df.groupby(["brand", "model"]):
    if len(g) >= 3:
        catalog.setdefault(b, {})[m] = g["name"].value_counts().index.tolist()[:25]

out = {
    "meta": {"dataset": "CarDekho used-car listings (India), 2020", "rows_raw": info["rows_raw"], "rows_clean": info["rows_clean"],
             "duplicates_removed": info["duplicates_removed"], "split": {k: len(v) for k, v in parts.items()}, "seed": SEED,
             "price_range_inr": [int(df.selling_price.min()), int(df.selling_price.max())], "ref_year": D.REF_YEAR},
    "catalog": catalog, "models": models,
    "classification": cls, "regression": reg, "importance": [[n, round(v, 4)] for n, v in importance],
    "sklearn": sk, "ada_view": ada_view, "examples": ex,
    "gbm_features_note": "permutation importance = increase in test RMSE(log price) when a feature is shuffled",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "model.json").write_text(json.dumps(out, separators=(",", ":")))

# golden predictions so the JavaScript implementation can be verified against Python
idx = np.random.RandomState(9).choice(len(parts["test"]), 25, replace=False)
gold = []
for i in idx:
    r = parts["test"].iloc[i]
    pcs = [Pt[n][i] for n in names_c]
    ps = stack_c.predict_proba([p[None, :] for p in pcs])[0]
    gold.append({"in": {"name": r["name"], "brand": r.brand, "model": r.model, "age": int(r.age), "km": int(r.km_driven), "fuel": r.fuel, "seller": r.seller_type,
                        "transmission": r.transmission, "owner": r.owner},
                 "features": rnd(X["test"][i], 6), "class_stack": rnd(ps, 5), "price_log": float(mean_t_[i]), "sigma": float(sig_t[i]),
                 "lo90": float(mean_t_[i] - qn_final["0.9"] * sig_t[i]), "hi90": float(mean_t_[i] + qn_final["0.9"] * sig_t[i])})
(OUT / "golden.json").write_text(json.dumps(gold))
print(f"wrote model.json {(OUT / 'model.json').stat().st_size / 1e6:.2f} MB in {time.time() - t_start:.0f}s")
