#!/usr/bin/env python3
"""
Balanced pipeline: LGB + HGB with 5-fold CV, feature engineering, and blend.
Fast enough to complete, strong enough to score well.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

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

    # Parse ordinals
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

    # Target encode
    global_mean = float(np.mean(y))
    ys = pd.Series(y)
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

    # Feature engineering
    numeric_cols = [c for c in X.columns if c not in cats and pd.api.types.is_numeric_dtype(X[c])]
    for c in numeric_cols:
        if X[c].isna().mean() > 0.01:
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

    # Drop cats for numeric models
    keep = [c for c in X.columns if c not in cats]
    Xn, Xtn = X[keep].copy(), Xt[keep].copy()
    Xn = Xn.replace([np.inf, -np.inf], np.nan).fillna(0)
    Xtn = Xtn.replace([np.inf, -np.inf], np.nan).fillna(0)

    n = len(Xn)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model 1: LightGBM
    print("Training LGB...")
    import lightgbm as lgbm
    oof_lgb = np.zeros(n)
    test_lgb = np.zeros(len(Xtn))
    for tr, va in skf.split(Xn, y):
        m = lgbm.LGBMClassifier(
            n_estimators=800, learning_rate=0.03, num_leaves=63,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1
        )
        m.fit(Xn.iloc[tr], y[tr], eval_set=[(Xn.iloc[va], y[va])], eval_metric="auc",
              callbacks=[lgbm.early_stopping(50, verbose=False)])
        oof_lgb[va] = m.predict_proba(Xn.iloc[va])[:, 1]
        test_lgb += m.predict_proba(Xtn)[:, 1] / 5
    auc_lgb = roc_auc_score(y, oof_lgb)
    print(f"  LGB OOF AUC: {auc_lgb:.5f}")

    # Model 2: HistGradientBoosting (native categoricals - use original data)
    print("Training HGB...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    # Re-encode categoricals for HGB
    X_hgb = X.copy()
    Xt_hgb = Xt.copy()
    # Keep numeric + target encoded columns only
    hgb_keep = [c for c in X_hgb.columns if c not in cats]
    X_hgb = X_hgb[hgb_keep].replace([np.inf, -np.inf], np.nan)
    Xt_hgb = Xt_hgb[hgb_keep].replace([np.inf, -np.inf], np.nan)

    oof_hgb = np.zeros(n)
    test_hgb = np.zeros(len(Xt_hgb))
    for tr, va in skf.split(X_hgb, y):
        m = HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.03, max_depth=7,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=42, early_stopping=True
        )
        m.fit(X_hgb.iloc[tr].fillna(0), y[tr])
        oof_hgb[va] = m.predict_proba(X_hgb.iloc[va].fillna(0))[:, 1]
        test_hgb += m.predict_proba(Xt_hgb.fillna(0))[:, 1] / 5
    auc_hgb = roc_auc_score(y, oof_hgb)
    print(f"  HGB OOF AUC: {auc_hgb:.5f}")

    # Blend: rank average
    rank_lgb = rankdata(oof_lgb) / n
    rank_hgb = rankdata(oof_hgb) / n
    blend_oof = (rank_lgb + rank_hgb) / 2
    auc_blend = roc_auc_score(y, blend_oof)

    rank_test_lgb = rankdata(test_lgb) / len(Xtn)
    rank_test_hgb = rankdata(test_hgb) / len(Xtn)
    blend_test = (rank_test_lgb + rank_test_hgb) / 2

    print(f"  Blend OOF AUC: {auc_blend:.5f}")

    # Pick best
    results = {
        "lgb": (auc_lgb, test_lgb),
        "hgb": (auc_hgb, test_hgb),
        "blend": (auc_blend, blend_test),
    }
    best_name = max(results, key=lambda k: results[k][0])
    best_auc, best_test = results[best_name]
    print(f"\nBest: {best_name} (AUC={best_auc:.5f})")

    # Write submission
    p = np.clip(np.nan_to_num(best_test, nan=0.5), 1e-6, 1 - 1e-6)
    out = sub[[id_col]].copy()
    d = pd.DataFrame({id_col: test[id_col].values, pred_col: p})
    out = out.merge(d, on=id_col, how="left")
    out[pred_col] = out[pred_col].fillna(0.5).astype(float)
    out.to_csv("submission.csv", index=False)
    print(f"SUBMISSION submission.csv written rows={len(out)}")

if __name__ == "__main__":
    main()
