# ₹ HonestPrice

**Used-car valuation with error bars you can check.** A council of tree-based models prices the car, places it in a tier, and reports a 90% interval that is *calibrated on held-out data* rather than assumed.

[**Live demo →**](https://honestprice.vercel.app) &nbsp;·&nbsp; Course: Machine Learning, **Unit 5** (combining models) &nbsp;·&nbsp; [![CI](https://github.com/mathan527/honestprice/actions/workflows/ci.yml/badge.svg)](https://github.com/mathan527/honestprice/actions/workflows/ci.yml)

Most price predictors give one number. A single number hides how uncertain a used-car price really is, so HonestPrice answers three questions:

1. **What is it worth?** Bayesian-averaged ensemble of five models (including gradient boosting).
2. **How sure are we?** Normalised conformal prediction: the 90% interval contained the true price **90.5%** of the time on 537 unseen cars.
3. **Do the models agree?** When the council is split on the price tier, accuracy drops from 86% to 69%, and the app says so.

---

## What's inside (everything written from scratch in NumPy)

| Unit 5 topic | Implementation | Where to see it |
|---|---|---|
| **CART** | Gini and squared-error splits, sample weights, histogram split search, minimal cost-complexity pruning | Tree explorer |
| **Committees** | Bagging and random forests with out-of-bag error | Boosting lab |
| **Boosting / AdaBoost** | Multi-class SAMME with a sample-weight visualiser | Boosting lab |
| **Additive models** | Gradient boosting as a stagewise sum of small trees, with early stopping | Valuation tab (stage-by-stage build-up) |
| **Bayesian model averaging** | Posterior weights from validation likelihood (exact and tempered) | Evidence tab |
| **Ensemble learning** | Stacking (softmax and NNLS), averaging, full comparison | Evidence tab |
| **Practice 1: CART for categorisation** | Price-tier classification (Budget / Mid-range / Premium) | Tree explorer |
| **Practice 2: ensembles for classification** | Bagging, random forest, AdaBoost, stacking on the same task | Evidence tab |

The browser runs the exported trees in JavaScript, so there is no server. A parity test confirms the JavaScript reproduces the Python predictions.

## Results (537 held-out cars)

**Price tier (3 classes, majority baseline 33.5%)**

| Model | Accuracy | Macro F1 | Log loss |
|---|---|---|---|
| CART (pruned) | 73.4% | 0.726 | 0.642 |
| Bagging | 79.0% | 0.793 | 0.511 |
| Random forest | 77.7% | 0.780 | 0.482 |
| AdaBoost | 76.2% | 0.764 | 0.495 |
| **Stacking** | **80.6%** | **0.805** | **0.459** |

**Price (log scale; mean baseline RMSE 0.808)**

| Model | RMSE (log) | Median error | R² (price) |
|---|---|---|---|
| Ridge (linear) | 0.331 | 19.8% | 0.51 |
| CART | 0.345 | 20.5% | 0.39 |
| Random forest | 0.315 | 19.9% | 0.47 |
| Gradient boosting | 0.300 | 17.2% | 0.53 |
| **BMA (tempered, production)** | **0.294** | **17.1%** | 0.52 |

**Are the error bars honest?** (nominal 90%)

| Interval method | Coverage | Width (upper/lower price) |
|---|---|---|
| Gaussian BMA (assumes model is right) | 92.9% | ×2.81 |
| Constant-width conformal | 90.7% | ×2.63 |
| **Normalised conformal (used)** | **90.5%** | **×2.61** |

The Gaussian interval over-covers at every level and is wider; conformal intervals track the nominal level. Coverage by price third (90% target): Budget 87.2%, Mid-range 92.2%, Premium 92.2%.

**Cross-check against scikit-learn:** the root split of our CART is identical; ensemble accuracies and RMSEs are within about 1 point of scikit-learn (for example gradient boosting RMSE 0.300 vs 0.295, random forest accuracy 77.7% vs 78.6%).

## Honest limitations

- Prices are **asking prices from one marketplace in 2020**, with no condition, accident history or city. A 90% interval of about ×2.6 (e.g. ₹3.0–12.3 lakh for a 2015 Honda City) is wide because the data is noisy, not because the method is cautious.
- The conformal guarantee is **marginal** (averaged over cars). The cheapest third is slightly under-covered (87%).
- Differences of 1–2 points between ensembles are within sampling noise at n = 537 (about ±3.5 points). Solid findings: ensembles beat a lone CART on log loss, and stacking/BMA add little over simple averaging.
- Exact BMA collapses onto one model (gradient boosting gets 100% weight). Tempering the likelihood by 1/√n keeps a real committee and gave the best error; this is a pragmatic choice, not a theorem.
- Rare models fall back to brand-level averages, and nothing after 2020 is represented.

## Rigor

- **No leakage:** 763 exact duplicate listings removed before splitting; target-encoded features (typical price of a model/variant) are computed out-of-fold on training rows.
- **Four-way split:** train / validation (pruning, early stopping, model weights) / calibration (conformal quantiles only) / test (used once).
- **21 unit tests** (`pytest`): tree splits, weighted leaves, pruning monotonicity, AdaBoost weights, additive property of gradient boosting, BMA/NNLS weights, conformal coverage, leakage checks.
- **Cross-language parity test:** `node tests/parity.test.js` checks 25 cars end-to-end (features, council, price, interval) against Python.
- CI runs both on every push.

## Run it

```bash
pip install -r requirements.txt
python -m pytest tests            # unit tests
python pipeline/build.py          # train, evaluate, export web/data/model.json (~2 min)
node tests/parity.test.js         # JavaScript vs Python
python -m http.server -d web      # open http://localhost:8000
```

```
pipeline/trees.py       CART: classification + regression, weights, pruning
pipeline/ensembles.py   bagging, random forest, AdaBoost, gradient boosting, ridge, stacking, BMA, conformal
pipeline/data.py        cleaning, leakage-free features, splits
pipeline/build.py       training, evaluation, scikit-learn cross-check, export
web/                    static app (model.js = inference, app.js = UI), deployed on Vercel
tests/                  pytest suite + JS parity test
```

## Data

CarDekho used-car listings (India, 4,340 rows; 3,577 after de-duplication), originally published on Kaggle as *Vehicle dataset from CarDekho*. Please check the original listing for its licence before reuse.

*Series: [WattWise AI](https://github.com/mathan527/wattwise-ai) (Unit 3, clustering) · [HarmonyHMM](https://github.com/mathan527/harmonyhmm) (Unit 4, HMMs) · HonestPrice (Unit 5, ensembles).*
