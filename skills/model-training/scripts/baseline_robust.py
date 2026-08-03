#!/usr/bin/env python3
"""
Robust baseline: HistGradientBoostingClassifier with native categoricals,
data-driven regularization, and seed-averaging. Mirrors the strongest known
recipe for the Kaggle-in-Kaggle competition family (Public LB-ranking).

- Detects text columns with a dtype-robust test (object OR pandas StringDtype),
  important under pandas>=3.0 where text is inferred as StringDtype.
- Lets HGB handle categorical + missing values natively (no imputation, no
  LabelEncoder, which would corrupt categorical structure on text columns).
- Applies L2 / min_samples_leaf gates based on feature-to-row ratio.
- Seed-averages K fits when (n_object_cols > 0) or (n_train >= 5000).

Writes submission.csv with columns [row_id, target] ordered as test.csv.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# --- cwd-robust bootstrap (skills may run in a temp dir; data lives in /work) --
def _pick_workdir():
    cwd = os.getcwd()
    for base in dict.fromkeys([cwd, "/work", "/kaggle/working"]):
        if os.path.exists(os.path.join(base, "train.csv")) or os.path.exists(
            os.path.join(base, "sample_submission.csv")
        ):
            return base
    return cwd

os.chdir(_pick_workdir())


def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    sample = pd.read_csv('sample_submission.csv')

    id_cols = [c for c in ['row_id', 'id'] if c in train.columns]
    id_col = id_cols[0] if id_cols else sample.columns[0]
    target = 'target'
    if target not in train.columns:
        # find the binary label column generically
        for c in train.columns:
            if train[c].nunique() == 2 and c != id_col and (
                c == 'target' or train[c].dtype.kind in 'ifb'
            ):
                target = c
                break

    features = [c for c in train.columns if c != id_col and c != target]
    if train[target].isna().any():
        train = train[train[target].notna()]
    X = train[features]
    y = train[target].astype(int)
    X_test = test[features]

    # dtype-robust categorical mask (object OR StringDtype); standardize to str
    X = X.astype({c: 'object' for c in X.columns if X[c].dtype == 'string'})
    cat_mask = X.dtypes == object
    if not X_test.columns.equals(X.columns):
        # align test columns/order to train
        X_test = X_test.reindex(columns=features)
    n_object_cols = int(cat_mask.sum())

    # feature-to-row ratio regularization gates
    n = len(train)
    n_feat = len(features)
    ratio = n_feat / n
    l2 = 1.0 if ratio >= 0.010 else 0.0
    if ratio >= 0.030:
        msl = 70
    elif ratio >= 0.015:
        msl = 50
    else:
        msl = 20

    # seed-averaging gate (OR-gate)
    seed_avg = (n_object_cols > 0) or (n >= 5000)
    seeds = list(range(10)) if seed_avg else [0]

    preds = np.zeros(len(X_test), dtype=np.float64)
    for seed in seeds:
        clf = HistGradientBoostingClassifier(
            categorical_features=cat_mask,
            random_state=seed,
            max_iter=300,
            early_stopping=True,
            l2_regularization=l2,
            min_samples_leaf=msl,
        )
        clf.fit(X, y)
        preds += clf.predict_proba(X_test)[:, 1]
    preds /= len(seeds)

    if len(sample) != len(test):
        # align sample row order to test instead
        out = pd.DataFrame({id_col: test[id_col], target: preds})
    else:
        sample = sample.copy()
        sample[id_col] = test[id_col].astype(sample[id_col].dtype)
        sample[target] = sample[target].astype(float)
        sample[target] = preds
        out = sample

    out.to_csv('submission.csv', index=False)
    n_obj = n_object_cols
    print(f"baseline OK rows={len(test)} feats={n_feat} n_object_cols={n_obj} "
          f"ratio={ratio:.4f} l2={l2} msl={msl} seed_avg={seed_avg} seeds={len(seeds)}")
    print(f"submission.csv written: cols={list(out.columns)} "
          f"pred range=[{preds.min():.3f},{preds.max():.3f}]")


if __name__ == '__main__':
    main()