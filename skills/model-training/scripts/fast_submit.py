#!/usr/bin/env python3
"""
Fast submission: single LGB model with feature engineering.
Runs in <2 minutes. Designed to always complete and leave budget for submit_predictions.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

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
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sub = pd.read_csv("sample_submission.csv")

    id_col = "row_id" if "row_id" in train.columns else ("id" if "id" in train.columns else sub.columns[0])
    target = "target" if "target" in train.columns else None
    if target is None:
        for c in train.columns:
            if c != id_col and train[c].nunique() == 2 and train[c].dtype.kind in "ifb":
                target = c
                break
    pred_col = sub.columns[1] if len(sub.columns) > 1 else target

    if train[target].isna().any():
        train = train[train[target].notna()]

    feats = [c for c in train.columns if c not in (id_col, target)]
    X = train[feats].copy()
    Xt = test[feats].copy()
    y = train[target].astype(int).values

    # Parse ordinal strings to numeric
    cats = []
    for c in feats:
        if pd.api.types.is_numeric_dtype(X[c]):
            continue
        vals = X[c].dropna().astype(str)
        if len(vals) > 0 and vals.str.match("^ord_[0-9]+$").all():
            X[c] = pd.to_numeric(X[c].astype(str).str.slice(4), errors="coerce")
            Xt[c] = pd.to_numeric(Xt[c].astype(str).str.slice(4), errors="coerce")
        else:
            cats.append(c)

    # Target encode categoricals
    global_mean = float(np.mean(y))
    ys = pd.Series(y)
    if len(cats) > 0:
        skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for c in cats:
            xc = X[c].astype(str)
            xtc = Xt[c].astype(str)
            oof = np.full(len(X), global_mean, dtype=float)
            for tr, va in skf_te.split(X, y):
                grp = ys.iloc[tr].groupby(xc.iloc[tr])
                means, counts = grp.mean(), grp.count()
                enc = (counts * means + 10 * global_mean) / (counts + 10)
                oof[va] = xc.iloc[va].map(enc).fillna(global_mean).values
            grp = ys.groupby(xc)
            fmeans, fcounts = grp.mean(), grp.count()
            fenc = (fcounts * fmeans + 10 * global_mean) / (fcounts + 10)
            X[c + "__te"] = oof.astype(float)
            Xt[c + "__te"] = xtc.map(fenc).fillna(global_mean).values
            fc = xc.value_counts()
            X[c + "__freq"] = xc.map(fc).fillna(0).astype(float)
            Xt[c + "__freq"] = xtc.map(fc).fillna(0).astype(float)

    # Quick feature engineering
    numeric_cols = [c for c in X.columns if c not in cats and pd.api.types.is_numeric_dtype(X[c])]
    for c in numeric_cols[:20]:
        miss_pct = X[c].isna().mean()
        if miss_pct > 0.01:
            X[c + "__miss"] = X[c].isna().astype(float)
            Xt[c + "__miss"] = Xt[c].isna().astype(float)

    # Top correlations for interactions
    corrs = {}
    for c in numeric_cols:
        try:
            corrs[c] = abs(np.corrcoef(X[c].fillna(0).values, y)[0, 1])
        except:
            corrs[c] = 0
    top_feats = sorted(corrs.keys(), key=lambda f: -corrs[f])[:8]
    for i in range(min(4, len(top_feats))):
        for j in range(i+1, min(6, len(top_feats))):
            f1, f2 = top_feats[i], top_feats[j]
            X[f"{f1}__x__{f2}"] = X[f1] * X[f2]
            Xt[f"{f1}__x__{f2}"] = Xt[f1] * Xt[f2]

    # Drop string cats for LGB
    keep = [c for c in X.columns if c not in cats]
    X, Xt = X[keep].copy(), Xt[keep].copy()

    # Replace inf and fill remaining NaN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    Xt = Xt.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Train LightGBM with OOF
    import lightgbm as lgbm
    n = len(X)
    oof = np.zeros(n)
    testp = np.zeros(len(Xt))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for tr, va in skf.split(X, y):
        m = lgbm.LGBMClassifier(
            n_estimators=800, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1
        )
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
              callbacks=[lgbm.early_stopping(50, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        testp += m.predict_proba(Xt)[:, 1] / 5

    auc = roc_auc_score(y, oof)
    print(f"RESULT lgb_oof_auc={auc:.5f}")

    # Write submission
    p = np.clip(np.nan_to_num(testp, nan=0.5), 1e-6, 1 - 1e-6)
    out = sub[[id_col]].copy()
    d = pd.DataFrame({id_col: test[id_col].values, pred_col: p})
    out = out.merge(d, on=id_col, how="left")
    out[pred_col] = out[pred_col].fillna(0.5).astype(float)
    out.to_csv("submission.csv", index=False)
    print(f"SUBMISSION submission.csv written rows={len(out)} auc={auc:.5f}")

if __name__ == "__main__":
    main()
