#!/usr/bin/env python3
"""
Aggressive pipeline: feature engineering + optimized ensemble + stacking.
Goal: push from 0.818 to 0.830+ AUC.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.optimize import minimize

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

def load():
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
    return X, Xt, y, cats, test[id_col], sub, id_col, pred_col


def target_encode(X, Xt, y, cats, smoothing=10.0):
    global_mean = float(np.mean(y))
    ys = pd.Series(y)
    Xn, Xtn = X.copy(), Xt.copy()
    if len(cats) == 0:
        return Xn, Xtn
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for c in cats:
        xc = X[c].astype(str)
        xtc = Xt[c].astype(str)
        oof = np.full(len(X), global_mean, dtype=float)
        for tr, va in skf.split(X, y):
            grp = ys.iloc[tr].groupby(xc.iloc[tr])
            means, counts = grp.mean(), grp.count()
            enc = (counts * means + smoothing * global_mean) / (counts + smoothing)
            oof[va] = xc.iloc[va].map(enc).fillna(global_mean).values
        grp = ys.groupby(xc)
        fmeans, fcounts = grp.mean(), grp.count()
        fenc = (fcounts * fmeans + smoothing * global_mean) / (fcounts + smoothing)
        test_enc = xtc.map(fenc).fillna(global_mean).values
        Xn[c + "__te"] = oof.astype(float)
        Xtn[c + "__te"] = test_enc.astype(float)
        fc = xc.value_counts()
        Xn[c + "__freq"] = xc.map(fc).fillna(0).astype(float)
        Xtn[c + "__freq"] = xtc.map(fc).fillna(0).astype(float)
    return Xn, Xtn


def aggressive_feature_engineering(X, Xt, cats, y):
    """Aggressive feature engineering for maximum AUC."""
    Xn, Xtn = X.copy(), Xt.copy()
    numeric_cols = [c for c in Xn.columns if c not in cats and pd.api.types.is_numeric_dtype(Xn[c])]

    # 1. Missing indicators for ALL columns with any missing
    for c in Xn.columns:
        miss_pct = Xn[c].isna().mean()
        if miss_pct > 0.01:
            Xn[c + "__miss"] = Xn[c].isna().astype(float)
            Xtn[c + "__miss"] = Xt[c].isna().astype(float)

    # 2. Log transforms for skewed numeric features
    for c in numeric_cols[:50]:
        vals = Xn[c].dropna()
        if len(vals) > 0 and vals.min() >= 0:
            Xn[c + "__log"] = np.log1p(Xn[c])
            Xtn[c + "__log"] = np.log1p(Xt[c])

    # 3. Square root transforms
    for c in numeric_cols[:30]:
        vals = Xn[c].dropna()
        if len(vals) > 0 and vals.min() >= 0:
            Xn[c + "__sqrt"] = np.sqrt(Xn[c])
            Xtn[c + "__sqrt"] = np.sqrt(Xt[c])

    # 4. Rank features (robust to outliers)
    for c in numeric_cols[:30]:
        Xn[c + "__rank"] = Xn[c].rank(pct=True)
        Xtn[c + "__rank"] = Xt[c].rank(pct=True)

    # 5. Binned features for top numeric
    for c in numeric_cols[:20]:
        try:
            Xn[c + "__bin10"] = pd.qcut(Xn[c], q=10, labels=False, duplicates='drop')
            Xt[c + "__bin10"] = pd.cut(Xt[c], bins=10, labels=False)
            Xn[c + "__bin10"] = Xn[c + "__bin10"].fillna(0)
            Xt[c + "__bin10"] = Xt[c + "__bin10"].fillna(0)
        except Exception:
            pass

    # 6. Interaction features (top correlations with target)
    if len(numeric_cols) >= 2:
        corrs = {}
        for c in numeric_cols:
            try:
                corrs[c] = abs(np.corrcoef(Xn[c].fillna(0).values, y)[0, 1])
            except Exception:
                corrs[c] = 0
        top_feats = sorted(corrs.keys(), key=lambda f: -corrs[f])[:min(10, len(corrs))]

        for i in range(len(top_feats)):
            for j in range(i+1, min(len(top_feats), i+3)):
                f1, f2 = top_feats[i], top_feats[j]
                Xn[f"{f1}__x__{f2}"] = Xn[f1] * Xn[f2]
                Xt[f"{f1}__x__{f2}"] = Xt[f1] * Xt[f2]
                denom = Xn[f2].replace(0, np.nan)
                Xn[f"{f1}__div__{f2}"] = Xn[f1] / denom
                denom_t = Xt[f2].replace(0, np.nan)
                Xt[f"{f1}__div__{f2}"] = Xt[f1] / denom_t

    # 7. Row-wise aggregations
    if len(numeric_cols) >= 3:
        top_n = numeric_cols[:min(15, len(numeric_cols))]
        Xn["__row_mean"] = Xn[top_n].mean(axis=1)
        Xt["__row_mean"] = Xt[top_n].mean(axis=1)
        Xn["__row_std"] = Xn[top_n].std(axis=1)
        Xt["__row_std"] = Xt[top_n].std(axis=1)
        Xn["__row_max"] = Xn[top_n].max(axis=1)
        Xt["__row_max"] = Xt[top_n].max(axis=1)
        Xn["__row_min"] = Xn[top_n].min(axis=1)
        Xt["__row_min"] = Xt[top_n].min(axis=1)
        Xn["__row_median"] = Xn[top_n].median(axis=1)
        Xt["__row_median"] = Xt[top_n].median(axis=1)
        Xn["__row_skew"] = Xn[top_n].skew(axis=1)
        Xt["__row_skew"] = Xt[top_n].skew(axis=1)

    # 8. Difference features between top correlated pairs
    if len(numeric_cols) >= 2:
        for i in range(min(5, len(top_feats))):
            for j in range(i+1, min(8, len(top_feats))):
                f1, f2 = top_feats[i], top_feats[j]
                Xn[f"{f1}__minus__{f2}"] = Xn[f1] - Xn[f2]
                Xt[f"{f1}__minus__{f2}"] = Xt[f1] - Xt[f2]

    # 9. Target-guided numeric transforms: z-score per feature
    for c in numeric_cols[:30]:
        mu = Xn[c].mean()
        sd = Xn[c].std()
        if sd > 0:
            Xn[c + "__zscore"] = (Xn[c] - mu) / sd
            Xt[c + "__zscore"] = (Xt[c] - mu) / sd

    return Xn, Xt


def safe_folds(nfolds, seed, X, y):
    try:
        return list(StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed).split(X, y))
    except ValueError:
        from sklearn.model_selection import KFold
        return list(KFold(n_splits=nfolds, shuffle=True, random_state=seed).split(X, y))


def proba1(m, X):
    p = m.predict_proba(X)
    if p.shape[1] < 2:
        return np.full(len(X), 1.0 if m.classes_[0] == 1 else 0.0)
    return p[:, 1]


def write_sub(fname, sub, id_col, pred_col, test_ids, preds):
    p = np.asarray(preds, dtype=float)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    d = pd.DataFrame({id_col: test_ids.values, pred_col: p})
    out = sub[[id_col]].copy()
    out = out.merge(d, on=id_col, how="left")
    out[pred_col] = out[pred_col].fillna(0.5).astype(float)
    out.to_csv(fname, index=False)


def fit_lgb(seed, Xa, ya, Xb, yb):
    import lightgbm as lgbm
    m = lgbm.LGBMClassifier(
        n_estimators=1200, learning_rate=0.03, num_leaves=63,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1
    )
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc",
          callbacks=[lgbm.early_stopping(80, verbose=False)])
    return m


def fit_xgb(seed, Xa, ya, Xb, yb):
    import xgboost as xgbm
    m = xgbm.XGBClassifier(
        n_estimators=1200, learning_rate=0.03, max_depth=7,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method="hist", enable_categorical=True, eval_metric="auc",
        early_stopping_rounds=80, random_state=seed, verbosity=0, n_jobs=-1
    )
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    return m


def fit_cat(seed, Xa, ya, Xb, yb, cats):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(
        iterations=1200, learning_rate=0.03, depth=8,
        l2_leaf_reg=3.0, eval_metric="AUC",
        cat_features=cats, early_stopping_rounds=80, random_seed=seed,
        verbose=0, allow_writing_files=False, thread_count=-1
    )
    m.fit(Xa, ya, eval_set=(Xb, yb))
    return m


def fit_hgb(seed, Xa, ya, Xb, yb):
    from sklearn.ensemble import HistGradientBoostingClassifier
    mask = np.zeros(Xa.shape[1], dtype=bool)
    m = HistGradientBoostingClassifier(
        max_iter=1200, learning_rate=0.03, max_depth=7,
        min_samples_leaf=20, l2_regularization=1.0,
        random_state=seed, early_stopping=True, categorical_features=mask
    )
    m.fit(Xa, ya)
    return m


def train_family(fam, X, Xt, y, cats, nfolds=5, seeds=[42, 101, 202]):
    """Train a model family with OOF CV. Returns (oof, test_pred, auc)."""
    n = len(X)
    oof = np.zeros(n)
    testp = np.zeros(len(Xt))
    t0 = time.time()

    if fam == "cat":
        Xm, Xtm = X.copy(), Xt.copy()
    else:
        keep = [c for c in X.columns if c not in cats]
        Xm, Xtm = X[keep].copy(), Xt[keep].copy()

    for seed in seeds:
        for tr, va in safe_folds(nfolds, seed, Xm, y):
            if fam == "lgb":
                m = fit_lgb(seed, Xm.iloc[tr], y[tr], Xm.iloc[va], y[va])
            elif fam == "xgb":
                m = fit_xgb(seed, Xm.iloc[tr], y[tr], Xm.iloc[va], y[va])
            elif fam == "cat":
                m = fit_cat(seed, Xm.iloc[tr], y[tr], Xm.iloc[va], y[va], cats)
            elif fam == "hgb":
                m = fit_hgb(seed, Xm.iloc[tr], y[tr], Xm.iloc[va], y[va])
            oof[va] += proba1(m, Xm.iloc[va]) / len(seeds)
            testp += proba1(m, Xtm) / (nfolds * len(seeds))

    auc = roc_auc_score(y, oof)
    elapsed = time.time() - t0
    print(f"  {fam}: OOF AUC={auc:.5f} ({elapsed:.0f}s, {nfolds}f, {len(seeds)}s)")
    return oof, testp, auc


def optimize_blend(oof_dict, y):
    """Find optimal non-negative weights for blending."""
    names = list(oof_dict.keys())
    n_models = len(names)
    mat = np.column_stack([oof_dict[n] for n in names])

    def obj(w):
        w = np.maximum(w, 0)
        s = w.sum()
        if s == 0:
            return 0
        w = w / s
        pred = mat @ w
        return -roc_auc_score(y, pred)

    best_w, best_auc = None, -1
    for _ in range(20):
        x0 = np.random.dirichlet(np.ones(n_models))
        res = minimize(obj, x0, method="Nelder-Mead", options={"maxiter": 2000})
        w = np.maximum(res.x, 0)
        s = w.sum()
        if s > 0:
            w = w / s
        pred = mat @ w
        auc = roc_auc_score(y, pred)
        if auc > best_auc:
            best_auc = auc
            best_w = w
    return dict(zip(names, best_w.tolist())), best_auc


def main():
    print("=" * 60)
    print("AGGRESSIVE PIPELINE - Maximum AUC")
    print("=" * 60)

    X0, Xt0, y, cats, ids, sub, idc, pc = load()
    print(f"Data: {len(X0)} train, {len(Xt0)} test, {len(cats)} cats")

    # Target encode
    print("\n[1] Target encoding...")
    X, Xt = target_encode(X0, Xt0, y, cats)

    # Aggressive feature engineering
    print("[2] Aggressive feature engineering...")
    X, Xt = aggressive_feature_engineering(X, Xt, cats, y)
    print(f"  Features: {X.shape[1]}")

    # Train all families
    print("\n[3] Training model families...")
    families = ["lgb", "xgb", "hgb", "cat"]
    oof_dict = {}
    test_dict = {}
    aucs = {}

    for fam in families:
        try:
            oof, testp, auc = train_family(fam, X, Xt, y, cats,
                                            nfolds=5, seeds=[42, 101, 202])
            oof_dict[fam] = oof
            test_dict[fam] = testp
            aucs[fam] = auc
            np.save(f"oof_{fam}.npy", oof)
            np.save(f"test_{fam}.npy", testp)
            write_sub(f"sub_{fam}.csv", sub, idc, pc, ids, testp)
        except Exception as e:
            print(f"  {fam} FAILED: {e}")

    if len(oof_dict) < 2:
        print("ERROR: Need at least 2 families")
        return

    # Rank-average blend of all
    print("\n[4] Blending...")
    all_names = list(oof_dict.keys())
    rank_oof = np.zeros(len(y))
    rank_test = np.zeros(len(Xt))
    for n in all_names:
        rank_oof += rankdata(oof_dict[n]) / (len(y) * len(all_names))
        rank_test += rankdata(test_dict[n]) / (len(Xt) * len(all_names))
    rank_auc = roc_auc_score(y, rank_oof)
    print(f"  Rank-avg (all {len(all_names)}): OOF AUC={rank_auc:.5f}")
    write_sub("sub_blend_all.csv", sub, idc, pc, ids, rank_test)

    # Optimized weighted blend
    print("\n[5] Optimized weighted blend...")
    weights, opt_auc = optimize_blend(oof_dict, y)
    print(f"  Optimized blend: OOF AUC={opt_auc:.5f}")
    print(f"  Weights: {weights}")

    weighted_test = np.zeros(len(Xt))
    for n in all_names:
        weighted_test += test_dict[n] * weights[n]
    write_sub("sub_blend_optimized.csv", sub, idc, pc, ids, weighted_test)

    # Ridge stacking
    print("\n[6] Ridge stacking...")
    from sklearn.linear_model import Ridge
    meta_mat = np.column_stack([oof_dict[n] for n in all_names])
    meta_test_mat = np.column_stack([test_dict[n] for n in all_names])

    # OOF for Ridge stacking
    ridge_oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in skf.split(meta_mat, y):
        ridge = Ridge(alpha=100.0)
        ridge.fit(meta_mat[tr], y[tr])
        ridge_oof[va] = ridge.predict(meta_mat[va])
    ridge_auc = roc_auc_score(y, ridge_oof)

    ridge_final = Ridge(alpha=100.0)
    ridge_final.fit(meta_mat, y)
    ridge_test = ridge_final.predict(meta_test_mat)
    write_sub("sub_ridge_stacking.csv", sub, idc, pc, ids, ridge_test)
    print(f"  Ridge stacking: OOF AUC={ridge_auc:.5f}")

    # Also try ridge stacking on rank-transformed predictions
    rank_meta = np.column_stack([rankdata(oof_dict[n]) / len(y) for n in all_names])
    rank_meta_test = np.column_stack([rankdata(test_dict[n]) / len(Xt) for n in all_names])
    ridge_rank_oof = np.zeros(len(y))
    for tr, va in skf.split(rank_meta, y):
        ridge = Ridge(alpha=100.0)
        ridge.fit(rank_meta[tr], y[tr])
        ridge_rank_oof[va] = ridge.predict(rank_meta[va])
    ridge_rank_auc = roc_auc_score(y, ridge_rank_oof)
    ridge_rank_final = Ridge(alpha=100.0)
    ridge_rank_final.fit(rank_meta, y)
    ridge_rank_test = ridge_rank_final.predict(rank_meta_test)
    write_sub("sub_ridge_rank_stacking.csv", sub, idc, pc, ids, ridge_rank_test)
    print(f"  Ridge rank stacking: OOF AUC={ridge_rank_auc:.5f}")

    # Select best submission
    print("\n[7] Best submissions:")
    candidates = {
        "sub_blend_optimized.csv": opt_auc,
        "sub_ridge_stacking.csv": ridge_auc,
        "sub_ridge_rank_stacking.csv": ridge_rank_auc,
        "sub_blend_all.csv": rank_auc,
    }
    for fam in all_names:
        candidates[f"sub_{fam}.csv"] = aucs[fam]

    ranked = sorted(candidates.items(), key=lambda x: -x[1])
    for i, (fname, auc) in enumerate(ranked):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1}. {fname}: OOF AUC={auc:.5f}{marker}")

    # Copy best to submission.csv for easy submission
    best_fname = ranked[0][0]
    import shutil
    shutil.copy(best_fname, "submission.csv")
    print(f"\n  -> submission.csv = {best_fname} (OOF AUC={ranked[0][1]:.5f})")

    print("\n" + "=" * 60)
    print(f"BEST OOF AUC: {ranked[0][1]:.5f}")
    print(f"TARGET: 0.830+ (gap: {0.830 - ranked[0][1]:.5f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
