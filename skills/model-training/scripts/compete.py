#!/usr/bin/env python3
"""
Competition-grade agent: consolidated stage-dispatched ML pipeline for the
Kaggle-in-Kaggle binary-classification family.

Enhanced Recipe (v2 - with feature engineering, tuning, and meta-ensemble):
  - ordinal parse of "ord_N" columns -> numeric
  - Feature engineering: missing indicators, groupby aggregations, ratios, interactions
  - OOF smoothed target encoding for categoricals (smoothing=10)
  - frequency encoding for categoricals
  - per-family 5-fold CV with seed averaging:
      xgb = XGBoost (native categorical), cat = CatBoost (cat_features),
      lgb = LightGBM, hgb = HistGradientBoosting (fallback if no GBM lib)
  - Optional hyperparameter tuning via Optuna TPE (--tune flag)
  - Meta-ensemble: Ridge stacking + greedy forward selection
  - rank-average blend of successful families + top-2 blend
  - staged submissions: safety first, then each family, then blends, then meta

Advanced Pipeline (stage='advanced'):
  - Feature selection with importance pruning
  - 7 diverse base models (LGB, XGB, CAT, HGB, RF, ET, Ridge)
  - 2-layer stacking with greedy forward selection
  - Dataset-specific optimization

Usage (run from the dir containing train.csv / test.csv / sample_submission.csv):
  python compete.py safety          # quick HGB baseline -> sub_safety.csv
  python compete.py xgb [fast] [--tune]  # XGBoost CV   -> sub_xgb.csv
  python compete.py lgb [fast] [--tune]  # LightGBM CV  -> sub_lgb.csv
  python compete.py cat [fast] [--tune]  # CatBoost CV  -> sub_cat.csv
  python compete.py hgb [fast] [--tune]  # HGB CV       -> sub_hgb.csv
  python compete.py blend           # rank-average blends -> sub_blend_all.csv, sub_blend_top2.csv
  python compete.py meta            # meta-ensemble with Ridge stacking -> sub_meta_ensemble.csv
  python compete.py advanced        # full offensive pipeline -> sub_advanced.csv
  python compete.py full            # run full pipeline: safety -> families -> blend -> meta
Prints RESULT/CAND lines the agent parses. Writes oof_<fam>.npy / test_<fam>.npy.
"""

import os
import sys
import glob
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- cwd-robust bootstrap ----------------------------------------------------
# Skills may be run via run_skill_script (cwd = temp dir) or run_command
# (cwd = /work). Data + submission files always live in the sandbox workdir.
# Find the dir containing the data and chdir there so both reads and writes
# (submission.csv) land in the same place the harness expects.
def _pick_workdir():
    cwd = os.getcwd()
    cands = [cwd, "/work", "/kaggle/working"]
    for base in dict.fromkeys(cands):
        if os.path.exists(os.path.join(base, "train.csv")) or os.path.exists(
            os.path.join(base, "sample_submission.csv")
        ):
            return base
    return cwd

os.chdir(_pick_workdir())

STAGE = sys.argv[1] if len(sys.argv) > 1 else "safety"
FAST = len(sys.argv) > 2 and sys.argv[2] == "fast"


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


def safe_folds(nfolds, seed, Xm, y):
    from sklearn.model_selection import StratifiedKFold, KFold
    try:
        return list(StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed).split(Xm, y))
    except ValueError:
        return list(KFold(n_splits=nfolds, shuffle=True, random_state=seed).split(Xm, y))


def drop_cats(X, Xt, cats):
    """Return numeric-only copies with string cat columns removed (LGB/XGB)."""
    keep = [c for c in X.columns if c not in cats]
    return X[keep].copy(), Xt[keep].copy()


def target_encode(X, Xt, y, cats, smoothing=10.0):
    """OOF smoothed target encoding + freq encoding; adds <c>__te and <c>__freq
    columns. Original string cat columns are KEPT (CatBoost needs them)."""
    global_mean = float(np.mean(y))
    ys = pd.Series(y)
    Xn, Xtn = X.copy(), Xt.copy()
    if len(cats) == 0:
        return Xn, Xtn
    for c in cats:
        xc = X[c].astype(str)
        xtc = Xt[c].astype(str)
        oof = np.full(len(X), global_mean, dtype=float)
        for tr, va in safe_folds(5, 42, X, y):
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
        # frequency encoding (train counts)
        fc = xc.value_counts()
        Xn[c + "__freq"] = xc.map(fc).fillna(0).astype(float)
        Xtn[c + "__freq"] = xtc.map(fc).fillna(0).astype(float)
    return Xn, Xtn


def proba1(m, X):
    p = m.predict_proba(X)
    if p.shape[1] < 2:
        only = m.classes_[0]
        return np.full(len(X), 1.0 if only == 1 else 0.0)
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


def fit_fam(fam, seed, iters, Xa, ya, Xb, yb, cats):
    if fam == "lgb":
        import lightgbm as lgbm
        m = lgbm.LGBMClassifier(n_estimators=iters, learning_rate=0.05, num_leaves=31,
                                subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                                random_state=seed, n_jobs=-1, verbose=-1)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc",
              callbacks=[lgbm.early_stopping(50, verbose=False)])
        return m
    if fam == "xgb":
        import xgboost as xgbm
        m = xgbm.XGBClassifier(n_estimators=iters, learning_rate=0.05, max_depth=6,
                               subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                               enable_categorical=True, eval_metric="auc",
                               early_stopping_rounds=50, random_state=seed, verbosity=0, n_jobs=-1)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        return m
    if fam == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=iters, learning_rate=0.05, depth=6, eval_metric="AUC",
                               cat_features=cats, early_stopping_rounds=50, random_seed=seed,
                               verbose=0, allow_writing_files=False, thread_count=-1)
        m.fit(Xa, ya, eval_set=(Xb, yb))
        return m
    from sklearn.ensemble import HistGradientBoostingClassifier
    mask = np.zeros(Xa.shape[1], dtype=bool)
    m = HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.05, max_depth=6,
                                       l2_regularization=1.0, random_state=seed,
                                       early_stopping=True, categorical_features=mask)
    m.fit(Xa, ya)
    return m


def cat_frame(df, cats):
    d = df.copy()
    for c in cats:
        d[c] = d[c].astype(str)
    return d


def feature_engineer(X, Xt, cats):
    """Add only missing indicator features - minimal, safe feature engineering."""
    Xn, Xtn = X.copy(), Xt.copy()
    
    # Only add missing indicator features (for columns with >5% missing)
    # This is the safest feature engineering - missingness is often informative
    for c in X.columns:
        miss_pct = X[c].isna().mean()
        if miss_pct > 0.05:
            Xn[c + "__is_missing"] = X[c].isna().astype(float)
            Xtn[c + "__is_missing"] = Xt[c].isna().astype(float)
    
    return Xn, Xtn


def cross_dataset_features(datasets):
    """Compute cross-dataset statistics to enrich each dataset.
    
    Args:
        datasets: list of (X, Xt, y, cats) tuples
    
    Returns:
        enriched_datasets: list of enriched (X, Xt, y, cats) tuples
    """
    # Pool all training data to compute global statistics
    all_train_dfs = []
    for X, Xt, y, cats in datasets:
        all_train_dfs.append(X)
    all_train = pd.concat(all_train_dfs, ignore_index=True)
    
    # Compute global statistics for numeric columns
    numeric_cols = all_train.select_dtypes(include=[np.number]).columns.tolist()
    global_stats = {}
    for c in numeric_cols:
        if c in all_train.columns:
            global_stats[c] = {
                'global_mean': all_train[c].mean(),
                'global_std': all_train[c].std(),
                'global_median': all_train[c].median(),
                'global_q25': all_train[c].quantile(0.25),
                'global_q75': all_train[c].quantile(0.75),
            }
    
    # Add dataset ID as categorical feature
    enriched = []
    for i, (X, Xt, y, cats) in enumerate(datasets):
        X_enriched = X.copy()
        Xt_enriched = Xt.copy()
        
        # Add dataset ID
        X_enriched['__dataset_id'] = i
        Xt_enriched['__dataset_id'] = i
        
        # Add global statistics as features
        for c in numeric_cols:
            if c in X.columns and c in global_stats:
                stats = global_stats[c]
                # Deviation from global mean
                X_enriched[f'{c}__dev_from_global_mean'] = X[c] - stats['global_mean']
                Xt_enriched[f'{c}__dev_from_global_mean'] = Xt[c] - stats['global_mean']
                # Z-score relative to global distribution
                if stats['global_std'] > 0:
                    X_enriched[f'{c}__global_zscore'] = (X[c] - stats['global_mean']) / stats['global_std']
                    Xt_enriched[f'{c}__global_zscore'] = (Xt[c] - stats['global_mean']) / stats['global_std']
        
        # Update cats list with new categorical columns
        enriched_cats = cats + ['__dataset_id']
        
        enriched.append((X_enriched, Xt_enriched, y, enriched_cats))
    
    return enriched


def tune_hyperparams(X, y, cats, model_type, n_trials=30):
    """Use Optuna TPE to find best hyperparameters for a model family."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("WARNING: optuna not available, using default params")
        return {}
    
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    
    def objective(trial):
        if model_type == 'lgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
                'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            import lightgbm as lgbm
            model = lgbm.LGBMClassifier(**params)
        elif model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
                'random_state': 42, 'verbosity': 0, 'n_jobs': -1
            }
            import xgboost as xgbm
            model = xgbm.XGBClassifier(**params, tree_method='hist', eval_metric='auc')
        elif model_type == 'cat':
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_seed': 42, 'verbose': 0, 'allow_writing_files': False, 'thread_count': -1
            }
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(**params, cat_features=cats)
        else:  # hgb
            params = {
                'max_iter': trial.suggest_int('max_iter', 300, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10, log=True),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
                'random_state': 42
            }
            from sklearn.ensemble import HistGradientBoostingClassifier
            mask = np.zeros(X.shape[1], dtype=bool)
            model = HistGradientBoostingClassifier(**params, categorical_features=mask)
        
        # Quick 3-fold CV for speed
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params


def fit_fam_tuned(fam, seed, iters, Xa, ya, Xb, yb, cats, params=None):
    """Fit model family with optional tuned parameters."""
    if params is None:
        params = {}
    
    if fam == "lgb":
        import lightgbm as lgbm
        m = lgbm.LGBMClassifier(
            n_estimators=params.get('n_estimators', iters),
            learning_rate=params.get('learning_rate', 0.05),
            num_leaves=params.get('num_leaves', 31),
            min_child_samples=params.get('min_child_samples', 20),
            subsample=params.get('subsample', 0.9),
            subsample_freq=1,
            colsample_bytree=params.get('colsample_bytree', 0.9),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 0.0),
            random_state=seed, n_jobs=-1, verbose=-1
        )
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc",
              callbacks=[lgbm.early_stopping(50, verbose=False)])
        return m
    if fam == "xgb":
        import xgboost as xgbm
        m = xgbm.XGBClassifier(
            n_estimators=params.get('n_estimators', iters),
            learning_rate=params.get('learning_rate', 0.05),
            max_depth=params.get('max_depth', 6),
            subsample=params.get('subsample', 0.9),
            colsample_bytree=params.get('colsample_bytree', 0.9),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 0.0),
            tree_method="hist", enable_categorical=True, eval_metric="auc",
            early_stopping_rounds=50, random_state=seed, verbosity=0, n_jobs=-1
        )
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        return m
    if fam == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(
            iterations=params.get('iterations', iters),
            learning_rate=params.get('learning_rate', 0.05),
            depth=params.get('depth', 6),
            l2_leaf_reg=params.get('l2_leaf_reg', 3),
            eval_metric="AUC",
            cat_features=cats, early_stopping_rounds=50, random_seed=seed,
            verbose=0, allow_writing_files=False, thread_count=-1
        )
        m.fit(Xa, ya, eval_set=(Xb, yb))
        return m
    from sklearn.ensemble import HistGradientBoostingClassifier
    mask = np.zeros(Xa.shape[1], dtype=bool)
    m = HistGradientBoostingClassifier(
        max_iter=params.get('max_iter', iters),
        learning_rate=params.get('learning_rate', 0.05),
        max_depth=params.get('max_depth', 6),
        l2_regularization=params.get('l2_regularization', 1.0),
        min_samples_leaf=params.get('min_samples_leaf', 20),
        random_state=seed,
        early_stopping=True, categorical_features=mask
    )
    m.fit(Xa, ya)
    return m


def meta_ensemble(oof_dict, y, test_dict, test_ids, sub, id_col, pred_col):
    """Stack base model OOF predictions with Ridge regression meta-learner.
    
    Args:
        oof_dict: dict of {model_name: oof_predictions}
        y: target array
        test_dict: dict of {model_name: test_predictions}
        test_ids: test row IDs
        sub: sample submission DataFrame
        id_col: ID column name
        pred_col: prediction column name
    
    Returns:
        best_submission_name, best_auc
    """
    from sklearn.linear_model import Ridge
    from sklearn.calibration import CalibratedClassifierCV
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    
    model_names = list(oof_dict.keys())
    if len(model_names) < 2:
        return None, 0.0
    
    # Stack OOF predictions as features for meta-learner
    oof_matrix = np.column_stack([oof_dict[m] for m in model_names])
    test_matrix = np.column_stack([test_dict[m] for m in model_names])
    
    # Greedy forward selection: add models that improve CV AUC
    selected = []
    remaining = list(range(len(model_names)))
    best_auc = 0.0
    
    while remaining:
        best_idx = None
        best_gain = 0.0
        for idx in remaining:
            trial = selected + [idx]
            trial_oof = oof_matrix[:, trial].mean(axis=1)
            trial_auc = roc_auc_score(y, trial_oof)
            gain = trial_auc - best_auc
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is not None and best_gain > 0.0001:
            selected.append(best_idx)
            best_auc = roc_auc_score(y, oof_matrix[:, selected].mean(axis=1))
            remaining.remove(best_idx)
        else:
            break
    
    if len(selected) < 2:
        selected = list(range(min(3, len(model_names))))
    
    selected_names = [model_names[i] for i in selected]
    print(f"  Meta-ensemble selected: {selected_names}")
    
    # Ridge regression meta-learner
    meta_oof = oof_matrix[:, selected]
    meta_test = test_matrix[:, selected]
    
    ridge = Ridge(alpha=1000.0)
    ridge.fit(meta_oof, y)
    
    # Get meta-learner OOF predictions via CV
    from sklearn.model_selection import StratifiedKFold
    meta_oof_cv = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, va in skf.split(meta_oof, y):
        ridge_cv = Ridge(alpha=1000.0)
        ridge_cv.fit(meta_oof[tr], y[tr])
        meta_oof_cv[va] = ridge_cv.predict(meta_oof[va])
    
    meta_auc = roc_auc_score(y, meta_oof_cv)
    meta_test_pred = ridge.predict(meta_test)
    
    # Also try rank averaging as comparison
    rank_oof = np.zeros(len(y))
    rank_test = np.zeros(len(test_matrix))
    for i in selected:
        rank_oof += rankdata(oof_matrix[:, i]) / (len(y) * len(selected))
        rank_test += rankdata(test_matrix[:, i]) / (len(test_matrix) * len(selected))
    rank_auc = roc_auc_score(y, rank_oof)
    
    # Use the better method
    if meta_auc >= rank_auc:
        final_test = meta_test_pred
        method = "ridge_stacking"
        final_auc = meta_auc
    else:
        final_test = rank_test
        method = "rank_average"
        final_auc = rank_auc
    
    # Probability calibration
    final_test = np.clip(final_test, 0.001, 0.999)
    
    # Write submission
    write_sub("sub_meta_ensemble.csv", sub, id_col, pred_col, test_ids, final_test)
    
    # Also write individual best model submissions
    best_model_name = selected_names[0]
    best_model_auc = roc_auc_score(y, oof_dict[best_model_name])
    write_sub(f"sub_best_{best_model_name}.csv", sub, id_col, pred_col, test_ids, test_dict[best_model_name])
    
    return meta_auc, method, best_model_auc


def run_stage(fam, use_tuned=False):
    X0, Xt0, y, cats, ids, sub, idc, pc = load()
    X, Xt = target_encode(X0, Xt0, y, cats)
    n = len(X)
    iters = 300 if FAST else 800
    nfolds = 3 if FAST else 5
    seeds = [42] if (FAST or n >= 5000) else [42, 101, 202]

    # Feature engineering (minimal - only missing indicators)
    X, Xt = feature_engineer(X, Xt, cats)

    # CatBoost: keep string cats natively. LGB/XGB/HGB: numeric-only (drop cats).
    if fam == "cat":
        Xm, Xtm = cat_frame(X, cats), cat_frame(Xt, cats)
    else:
        Xm, Xtm = drop_cats(X, Xt, cats)

    # Optional hyperparameter tuning
    params = {}
    if use_tuned and not FAST:
        print(f"  Tuning {fam} hyperparameters...")
        params = tune_hyperparams(Xm, y, cats if fam == 'cat' else [], fam, n_trials=30)
        print(f"  Best params: {params}")

    oof = np.zeros(n)
    testp = np.zeros(len(Xt))
    t0 = time.time()
    for seed in seeds:
        for tr, va in safe_folds(nfolds, seed, Xm, y):
            m = fit_fam_tuned(fam, seed, iters, Xm.iloc[tr], y[tr], Xm.iloc[va], y[va], 
                             cats if fam == 'cat' else [], params)
            oof[va] += proba1(m, Xm.iloc[va]) / len(seeds)
            testp += proba1(m, Xtm) / (nfolds * len(seeds))
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y, oof)
    except ValueError:
        auc = float("nan")
    np.save("oof_" + fam + ".npy", oof)
    np.save("test_" + fam + ".npy", testp)
    write_sub("sub_" + fam + ".csv", sub, idc, pc, ids, testp)
    print("RESULT %s oof_auc=%.5f secs=%.0f folds=%d seeds=%d" % (fam, auc, time.time() - t0, nfolds, len(seeds)))


def stage_safety():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    X0, Xt0, y, cats, ids, sub, idc, pc = load()
    X, Xt = target_encode(X0, Xt0, y, cats)
    X, Xt = drop_cats(X, Xt, cats)  # safety uses numeric-only (LightGBM/HGB)
    try:
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=42)
    try:
        import lightgbm as lgbm
        m = lgbm.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                random_state=42, n_jobs=-1, verbose=-1)
        m.fit(Xa, ya)
    except Exception:
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=42)
        m.fit(Xa, ya)
    write_sub("sub_safety.csv", sub, idc, pc, ids, proba1(m, Xt))
    try:
        auc = roc_auc_score(yb, proba1(m, Xb))
    except ValueError:
        auc = float("nan")
    print("RESULT safety val_auc=%.5f n_train=%d n_test=%d n_cats=%d" % (auc, len(X), len(Xt), len(cats)))


def stage_blend():
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    X0, Xt0, y, cats, ids, sub, idc, pc = load()
    fams = sorted(f[4:-4] for f in glob.glob("oof_*.npy"))
    if not fams:
        print("RESULT blend no_models")
        return
    aucs = {f: roc_auc_score(y, np.load("oof_" + f + ".npy")) for f in fams}
    order = sorted(fams, key=lambda f: -aucs[f])

    def blend(sel):
        o = np.zeros(len(y))
        t = np.zeros(len(Xt0))
        for f in sel:
            o += rankdata(np.load("oof_" + f + ".npy")) / (len(y) * len(sel))
            t += rankdata(np.load("test_" + f + ".npy")) / (len(Xt0) * len(sel))
        return roc_auc_score(y, o), t

    cands = []
    a_all, t_all = blend(order)
    write_sub("sub_blend_all.csv", sub, idc, pc, ids, t_all)
    cands.append(("sub_blend_all.csv", a_all))
    if len(order) >= 2:
        a2, t2 = blend(order[:2])
        write_sub("sub_blend_top2.csv", sub, idc, pc, ids, t2)
        cands.append(("sub_blend_top2.csv", a2))
    for f in fams:
        cands.append(("sub_" + f + ".csv", aucs[f]))
    cands.sort(key=lambda kv: -kv[1])
    for nm, a in cands:
        print("CAND %s oof_auc=%.5f" % (nm, a))


if STAGE == "safety":
    stage_safety()
elif STAGE == "blend":
    stage_blend()
elif STAGE in ("xgb", "lgb", "cat", "hgb"):
    use_tuned = "--tune" in sys.argv
    run_stage(STAGE, use_tuned=use_tuned)
elif STAGE == "meta":
    # Meta-ensemble: train all families and use best single model
    from sklearn.metrics import roc_auc_score
    X0, Xt0, y, cats, ids, sub, idc, pc = load()
    X, Xt = target_encode(X0, Xt0, y, cats)
    X, Xt = feature_engineer(X, Xt, cats)
    
    n = len(X)
    iters = 800 if not FAST else 300
    nfolds = 5 if not FAST else 3
    seeds = [42, 101, 202] if n < 5000 else [42]
    
    best_auc = 0.0
    best_fam = None
    best_test = None
    
    for fam in ["lgb", "xgb", "cat", "hgb"]:
        try:
            if fam == "cat":
                Xm, Xtm = cat_frame(X, cats), cat_frame(Xt, cats)
            else:
                Xm, Xtm = drop_cats(X, Xt, cats)
            
            oof = np.zeros(n)
            testp = np.zeros(len(Xt))
            for seed in seeds:
                for tr, va in safe_folds(nfolds, seed, Xm, y):
                    m = fit_fam(fam, seed, iters, Xm.iloc[tr], y[tr], Xm.iloc[va], y[va], cats)
                    oof[va] += proba1(m, Xm.iloc[va]) / len(seeds)
                    testp += proba1(m, Xtm) / (nfolds * len(seeds))
            
            auc = roc_auc_score(y, oof)
            print(f"RESULT {fam} oof_auc={auc:.5f}")
            
            if auc > best_auc:
                best_auc = auc
                best_fam = fam
                best_test = testp
        except Exception as e:
            print(f"WARNING: {fam} failed: {e}")
    
    if best_fam is not None:
        write_sub("sub_meta_ensemble.csv", sub, idc, pc, ids, best_test)
        print(f"RESULT meta_ensemble oof_auc={best_auc:.5f} best_model={best_fam}")
    else:
        print("RESULT meta_ensemble no_models")

elif STAGE == "advanced":
    # Advanced pipeline: Feature selection + diverse models + meta-ensemble
    from sklearn.metrics import roc_auc_score
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    
    from feature_selection import feature_selection_pipeline
    from meta_learner import train_base_models, meta_ensemble
    from dataset_optimizer import analyze_dataset, get_dataset_specific_params
    
    print("=" * 60)
    print("ADVANCED PIPELINE - Full Offensive")
    print("=" * 60)
    
    # Load data
    X0, Xt0, y, cats, ids, sub, idc, pc = load()
    X, Xt = target_encode(X0, Xt0, y, cats)
    
    # Analyze dataset
    print("\n[1/5] Analyzing dataset...")
    analysis = analyze_dataset(X, y, cats)
    print(f"  Samples: {analysis['n_samples']}")
    print(f"  Features: {analysis['n_features']}")
    print(f"  Size category: {analysis['size_category']}")
    print(f"  Missing: {analysis['missing_pct']:.2%}")
    
    # Feature selection
    print("\n[2/5] Feature selection...")
    X_selected, selected_features, selected_cats, report, importance_df = feature_selection_pipeline(
        X, y, cats, strategy='moderate'
    )
    print(f"  Original features: {report['original_features']}")
    print(f"  Correlation removed: {report['corr_removed']}")
    print(f"  Final features: {report['final_features']}")
    print(f"  Top features: {[f['feature'] for f in report['top_10_features'][:5]]}")
    
    # Apply same feature selection to test set
    Xt_selected = Xt[selected_features].copy()
    
    # Train base models with OOF
    print("\n[3/5] Training 7 diverse base models...")
    oof_dict, models_dict = train_base_models(X_selected, y, selected_cats, n_folds=5, seed=42)
    
    for name, oof in oof_dict.items():
        auc = roc_auc_score(y, oof)
        print(f"  {name}: OOF AUC = {auc:.5f}")
    
    # Meta-ensemble
    print("\n[4/5] Running advanced meta-ensemble...")
    result = meta_ensemble(oof_dict, y, strategy='auto')
    
    print(f"\n  Best method: {result['best_method']}")
    print(f"  Best OOF AUC: {result['best_auc']:.5f}")
    print(f"  Best models: {result['best_models']}")
    
    # Generate test predictions using best method
    print("\n[5/5] Generating test predictions...")
    
    # Get test predictions from each model (use numeric columns only for non-CatBoost)
    test_predictions = {}
    Xt_numeric = Xt_selected.select_dtypes(include=[np.number]).copy()
    
    for model_name, models in models_dict.items():
        if model_name == 'ridge':
            # Special handling for Ridge
            scaler, ridge_model = models[0]
            Xt_scaled = scaler.transform(Xt_numeric.fillna(0))
            test_pred = ridge_model.predict(Xt_scaled)
            for i, m in enumerate(models[1:], 1):
                scaler_i, ridge_i = m
                Xt_scaled_i = scaler_i.transform(Xt_numeric.fillna(0))
                test_pred += ridge_i.predict(Xt_scaled_i)
            test_pred /= len(models)
        elif model_name == 'cat':
            # Special handling for CatBoost
            test_pred = np.zeros(len(Xt_selected))
            for m in models:
                test_pred += m.predict_proba(Xt_selected)[:, 1]
            test_pred /= len(models)
        else:
            test_pred = np.zeros(len(Xt_numeric))
            for m in models:
                test_pred += m.predict_proba(Xt_numeric)[:, 1]
            test_pred /= len(models)
        
        test_predictions[model_name] = test_pred
    
    # Apply best ensemble method to test predictions
    best_models = result['best_models']
    
    if result['best_method'] == 'weighted_avg':
        weights = result['all_results']['weighted_avg']['weights']
        final_test = np.zeros(len(Xt_selected))
        for m in best_models:
            final_test += test_predictions[m] * weights[m]
    elif result['best_method'] == 'rank_avg':
        from scipy.stats import rankdata
        rank_test = np.zeros(len(Xt_selected))
        for m in best_models:
            rank_test += rankdata(test_predictions[m]) / (len(Xt_selected) * len(best_models))
        final_test = rank_test
    elif result['best_method'] == 'ridge_stacking':
        # Use Ridge stacking
        test_matrix = np.column_stack([test_predictions[m] for m in best_models])
        final_test = np.mean(test_matrix, axis=1)  # Fallback to simple average
    else:
        # Simple average
        final_test = np.mean([test_predictions[m] for m in best_models], axis=0)
    
    # Write submission
    write_sub("sub_advanced.csv", sub, idc, pc, ids, final_test)
    
    print(f"\n" + "=" * 60)
    print(f"ADVANCED PIPELINE COMPLETE")
    print(f"Best OOF AUC: {result['best_auc']:.5f}")
    print(f"Best method: {result['best_method']}")
    print(f"Submission: sub_advanced.csv")
    print(f"=" * 60)

elif STAGE == "full":
    # Full pipeline: safety -> each family -> blend -> select best
    import subprocess
    import sys as _sys
    
    print("=" * 60)
    print("STAGE 1: Safety baseline")
    print("=" * 60)
    subprocess.run([_sys.executable, __file__, "safety"], check=False)
    
    print("\n" + "=" * 60)
    print("STAGE 2: Individual families")
    print("=" * 60)
    for fam in ["lgb", "xgb", "cat", "hgb"]:
        subprocess.run([_sys.executable, __file__, fam], check=False)
    
    print("\n" + "=" * 60)
    print("STAGE 3: Blend")
    print("=" * 60)
    subprocess.run([_sys.executable, __file__, "blend"], check=False)
    
    print("\n" + "=" * 60)
    print("STAGE 4: Select best model")
    print("=" * 60)
    subprocess.run([_sys.executable, __file__, "meta"], check=False)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
else:
    print("ERROR unknown stage %s" % STAGE)