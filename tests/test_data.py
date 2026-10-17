import numpy as np
import pytest

import data as D


@pytest.fixture(scope="module")
def prepared():
    df, info = D.load_clean()
    return df, info, D.make_split(df, 7)


def test_exact_duplicates_removed(prepared):
    df, info, _ = prepared
    assert info["duplicates_removed"] > 0
    assert not df[["name", "year", "selling_price", "km_driven", "fuel", "seller_type", "transmission", "owner"]].duplicated().any()


def test_splits_are_disjoint_and_complete(prepared):
    df, _, sp = prepared
    allidx = np.concatenate(list(sp.values()))
    assert len(set(allidx)) == len(allidx) == len(df)


def test_out_of_fold_encoding_does_not_leak_own_price(prepared):
    df, _, sp = prepared
    tr = df.iloc[sp["train"]]
    fb = D.FeatureBuilder().fit(tr)
    X = fb.transform(tr, train=True)
    full = np.array([fb.model_te.get(m, fb.prior) for m in tr["model"]])
    y = tr["log_price"].values
    # an in-sample encoding sees its own label, so it must correlate more with the target than the out-of-fold one
    assert np.corrcoef(X[:, 8], y)[0, 1] < np.corrcoef(full, y)[0, 1]


def test_unseen_model_falls_back_to_brand_then_prior(prepared):
    df, _, sp = prepared
    fb = D.FeatureBuilder().fit(df.iloc[sp["train"]])
    row = fb.row("Maruti", "Maruti Nonexistent", 5, 50000, "Petrol", "Individual", "Manual", "First Owner")
    assert abs(row[8] - fb.brand_te["Maruti"]) < 1e-12
    row = fb.row("Unknown", "Unknown Car", 5, 50000, "Petrol", "Individual", "Manual", "First Owner")
    assert abs(row[8] - fb.prior) < 1e-12


def test_engine_size_parser():
    assert D.engine_size("Hyundai Verna 1.6 SX") == 1.6
    assert np.isnan(D.engine_size("Maruti 800 AC"))
