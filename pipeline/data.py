"""Data preparation for HonestPrice: cleaning, leak-free feature engineering and the train/val/calibration/test split."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REF_YEAR = 2020  # newest model year in the data
OWNER = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2, "Fourth & Above Owner": 3, "Test Drive Car": 0}
TIERS = ["Budget", "Mid-range", "Premium"]


def brand_model(name):
    t = name.split()
    brand = {"Land": "Land Rover", "Mercedes-Benz": "Mercedes-Benz"}.get(t[0], t[0])
    if t[0] == "Land":
        t = ["Land", "Rover"] + t[2:]
    model = " ".join(t[:3] if t[0] == "Land" else t[:2])
    return brand, model


def engine_size(name):
    """Litre displacement written in the listing name (e.g. 'Verna 1.6 SX' -> 1.6); NaN when absent."""
    m = re.search(r"(?<![0-9.])([0-9][.][0-9])(?![0-9])", name)
    return float(m.group(1)) if m else np.nan


def load_clean():
    raw = pd.read_csv(ROOT / "data" / "cars_raw.csv")
    n0 = len(raw)
    df = raw.drop_duplicates().reset_index(drop=True)  # identical listings would leak between train and test
    df[["brand", "model"]] = df["name"].apply(lambda s: pd.Series(brand_model(s)))
    df["engine"] = df["name"].apply(engine_size)
    df["age"] = REF_YEAR - df["year"]
    df["log_km"] = np.log1p(df["km_driven"])
    df["log_price"] = np.log(df["selling_price"])
    return df, {"rows_raw": n0, "rows_clean": len(df), "duplicates_removed": n0 - len(df)}


def smooth_te(keys_train, y_train, prior, m=5):
    g = pd.DataFrame({"k": keys_train, "y": y_train}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + m * prior) / (g["count"] + m)).to_dict(), g["count"].to_dict()


def make_split(df, seed=7, fr=(0.60, 0.13, 0.12, 0.15)):
    rs = np.random.RandomState(seed)
    p = rs.permutation(len(df))
    a, b, c = (int(len(df) * x) for x in np.cumsum(fr)[:3])
    return {"train": p[:a], "val": p[a:b], "cal": p[b:c], "test": p[c:]}


class FeatureBuilder:
    """Fit on the training rows only. Model-level target encoding uses out-of-fold values on the training rows."""

    def fit(self, df_train, seed=3, folds=5):
        self.brands = [b for b, c in df_train["brand"].value_counts().items() if c >= 60]
        self.prior = float(df_train["log_price"].mean())
        self.engine_med = float(df_train["engine"].median())
        self.brand_te, _ = smooth_te(df_train["brand"], df_train["log_price"], self.prior, 10)
        self.model_te, self.model_n = smooth_te(df_train["model"], df_train["log_price"], self.prior, 5)
        self.name_te, _ = smooth_te(df_train["name"], df_train["log_price"], self.prior, 3)
        # out-of-fold encoding for the training rows (prevents the model feature from memorising its own label)
        rs = np.random.RandomState(seed)
        fold = rs.randint(0, folds, len(df_train))
        oof, oof2 = np.zeros(len(df_train)), np.zeros(len(df_train))
        d = df_train.reset_index(drop=True)
        for f in range(folds):
            tr, te = d[fold != f], d[fold == f]
            bte, _ = smooth_te(tr["brand"], tr["log_price"], self.prior, 10)
            mte, _ = smooth_te(tr["model"], tr["log_price"], self.prior, 5)
            nte, _ = smooth_te(tr["name"], tr["log_price"], self.prior, 3)
            oof[fold == f] = [mte.get(m, bte.get(b, self.prior)) for m, b in zip(te["model"], te["brand"])]
            oof2[fold == f] = [nte.get(nm, mte.get(m, bte.get(b, self.prior))) for nm, m, b in zip(te["name"], te["model"], te["brand"])]
        self._oof, self._oof2 = oof, oof2
        self.names = (["age", "log_km", "owner_rank", "fuel_diesel", "fuel_petrol", "seller_dealer", "seller_trustmark",
                       "automatic", "model_price_index", "engine_litres", "variant_price_index"] + ["brand_" + b for b in self.brands])
        return self

    def row(self, brand, model, age, km, fuel, seller, transmission, owner, name=None, engine=None):
        te = self.model_te.get(model, self.brand_te.get(brand, self.prior))
        vte = self.name_te.get(name, te) if name else te
        eng = self.engine_med if engine is None or engine != engine else engine
        return [age, np.log1p(km), OWNER[owner], fuel == "Diesel", fuel == "Petrol", seller == "Dealer",
                seller == "Trustmark Dealer", transmission == "Automatic", te, eng, vte] + [brand == b for b in self.brands]

    def transform(self, df, train=False):
        X = np.array([self.row(r.brand, r.model, r.age, r.km_driven, r.fuel, r.seller_type, r.transmission, r.owner, r.name, r.engine)
                      for r in df.itertuples()], dtype=float)
        if train:
            X[:, 8], X[:, 10] = self._oof, self._oof2
        return X
