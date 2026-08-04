#!/usr/bin/env python3
"""
Meta-Learner Module - Advanced 2-layer stacking and ensemble techniques

Key techniques:
1. 2-layer stacking with diverse meta-learners
2. Greedy forward selection
3. Weighted averaging by CV score
4. Rank averaging
5. Blend optimization
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import rankdata


def train_base_models(X, y, cats, n_folds=5, seed=42):
    """
    Train diverse base models and return OOF predictions.
    
    Returns:
        oof_dict: dict of {model_name: oof_predictions}
        test_dict: dict of {model_name}: test_predictions (if X_test provided)
        models_dict: dict of {model_name: fitted_models}
    """
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier, 
        HistGradientBoostingClassifier
    )
    from sklearn.linear_model import Ridge
    from scipy.stats import rankdata
    
    try:
        import lightgbm as lgbm
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False
    
    try:
        import xgboost as xgbm
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
    
    try:
        from catboost import CatBoostClassifier
        HAS_CAT = True
    except ImportError:
        HAS_CAT = False
    
    n = len(X)
    oof_dict = {}
    models_dict = {}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Helper to get OOF predictions
    def get_oof(model, X, y, cat_features=None, is_catboost=False):
        oof = np.zeros(n)
        fitted_models = []
        
        # Drop categorical columns for non-CatBoost models
        if not is_catboost:
            X_numeric = X.select_dtypes(include=[np.number]).copy()
        else:
            X_numeric = X.copy()
        
        for tr, va in skf.split(X_numeric, y):
            X_tr, X_va = X_numeric.iloc[tr], X_numeric.iloc[va]
            y_tr, y_va = y[tr], y[va]
            
            if is_catboost and cat_features and HAS_CAT:
                m = model.__class__(**model.get_params())
                m.fit(X_tr, y_tr, cat_features=cat_features)
            else:
                m = model.__class__(**model.get_params())
                m.fit(X_tr, y_tr)
            
            oof[va] = m.predict_proba(X_va)[:, 1]
            fitted_models.append(m)
        
        return oof, fitted_models
    
    # Model 1: LightGBM
    if HAS_LGB:
        lgb_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': seed,
            'n_jobs': -1,
            'verbose': -1
        }
        lgb_model = lgbm.LGBMClassifier(**lgb_params)
        oof_lgb, models_lgb = get_oof(lgb_model, X, y, is_catboost=False)
        oof_dict['lgb'] = oof_lgb
        models_dict['lgb'] = models_lgb
    
    # Model 2: XGBoost
    if HAS_XGB:
        xgb_params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'random_state': seed,
            'verbosity': 0,
            'n_jobs': -1
        }
        xgb_model = xgbm.XGBClassifier(**xgb_params)
        oof_xgb, models_xgb = get_oof(xgb_model, X, y, is_catboost=False)
        oof_dict['xgb'] = oof_xgb
        models_dict['xgb'] = models_xgb
    
    # Model 3: CatBoost
    if HAS_CAT:
        cat_params = {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'random_seed': seed,
            'verbose': 0,
            'allow_writing_files': False
        }
        # CatBoost needs special handling for categorical features
        oof_cat = np.zeros(n)
        models_cat = []
        for tr, va in skf.split(X, y):
            m = CatBoostClassifier(**cat_params)
            m.fit(X.iloc[tr], y[tr], cat_features=cats if cats else None)
            oof_cat[va] = m.predict_proba(X.iloc[va])[:, 1]
            models_cat.append(m)
        oof_dict['cat'] = oof_cat
        models_dict['cat'] = models_cat
    
    # Model 4: HistGradientBoosting
    hgb_params = {
        'max_iter': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'random_state': seed
    }
    hgb_model = HistGradientBoostingClassifier(**hgb_params)
    oof_hgb, models_hgb = get_oof(hgb_model, X, y, is_catboost=False)
    oof_dict['hgb'] = oof_hgb
    models_dict['hgb'] = models_hgb
    
    # Model 5: ExtraTrees
    et_params = {
        'n_estimators': 500,
        'max_depth': 15,
        'random_state': seed,
        'n_jobs': -1
    }
    et_model = ExtraTreesClassifier(**et_params)
    oof_et, models_et = get_oof(et_model, X, y, is_catboost=False)
    oof_dict['et'] = oof_et
    models_dict['et'] = models_et
    
    # Model 6: RandomForest
    rf_params = {
        'n_estimators': 500,
        'max_depth': 15,
        'random_state': seed,
        'n_jobs': -1
    }
    rf_model = RandomForestClassifier(**rf_params)
    oof_rf, models_rf = get_oof(rf_model, X, y, is_catboost=False)
    oof_dict['rf'] = oof_rf
    models_dict['rf'] = models_rf
    
    # Model 7: Ridge (linear baseline)
    from sklearn.preprocessing import StandardScaler
    oof_ridge = np.zeros(n)
    models_ridge = []
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]).fillna(0))
    
    for tr, va in skf.split(X_scaled, y):
        m = Ridge(alpha=1000.0)
        m.fit(X_scaled[tr], y[tr])
        oof_ridge[va] = m.predict(X_scaled[va])
        models_ridge.append((scaler, m))
    oof_dict['ridge'] = oof_ridge
    models_dict['ridge'] = models_ridge
    
    return oof_dict, models_dict


def greedy_forward_selection(oof_dict, y, min_gain=0.0001):
    """
    Greedy forward selection of models for ensemble.
    
    Returns:
        selected: list of selected model names
        scores: dict of {model_name: auc_score}
    """
    model_names = list(oof_dict.keys())
    
    # Calculate individual scores
    scores = {}
    for name in model_names:
        scores[name] = roc_auc_score(y, oof_dict[name])
    
    # Sort by score descending
    sorted_models = sorted(scores.keys(), key=lambda x: -scores[x])
    
    # Greedy selection
    selected = []
    remaining = list(sorted_models)
    best_auc = 0.0
    
    while remaining:
        best_idx = None
        best_gain = 0.0
        
        for idx, name in enumerate(remaining):
            trial = selected + [name]
            trial_oof = np.mean([oof_dict[m] for m in trial], axis=0)
            trial_auc = roc_auc_score(y, trial_oof)
            gain = trial_auc - best_auc
            
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        
        if best_idx is not None and best_gain > min_gain:
            selected.append(remaining[best_idx])
            best_auc = roc_auc_score(y, np.mean([oof_dict[m] for m in selected], axis=0))
            remaining.pop(best_idx)
        else:
            break
    
    return selected, scores


def optimize_blend_weights(oof_dict, y, method='nelder-mead'):
    """
    Optimize blend weights using scipy optimization.
    
    Returns:
        weights: dict of {model_name: weight}
        blend_auc: AUC of optimized blend
    """
    model_names = list(oof_dict.keys())
    n_models = len(model_names)
    
    oof_matrix = np.column_stack([oof_dict[m] for m in model_names])
    
    def objective(weights):
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n_models) / n_models
        blend = np.average(oof_matrix, axis=1, weights=weights)
        return -roc_auc_score(y, blend)
    
    # Try multiple starting points
    best_weights = None
    best_auc = 0.0
    
    for _ in range(10):
        x0 = np.random.dirichlet(np.ones(n_models))
        result = minimize(objective, x0, method=method, 
                         options={'maxiter': 1000})
        
        weights = np.maximum(result.x, 0)
        weights = weights / weights.sum()
        blend = np.average(oof_matrix, axis=1, weights=weights)
        auc = roc_auc_score(y, blend)
        
        if auc > best_auc:
            best_auc = auc
            best_weights = weights
    
    weights_dict = {name: w for name, w in zip(model_names, best_weights)}
    
    return weights_dict, best_auc


def rank_average_blend(oof_dict, y):
    """
    Rank average blend of predictions.
    
    Returns:
        blend_oof: rank-averaged OOF predictions
        blend_auc: AUC of rank blend
    """
    model_names = list(oof_dict.keys())
    n_models = len(model_names)
    
    rank_oof = np.zeros(len(y))
    for name in model_names:
        rank_oof += rankdata(oof_dict[name]) / (len(y) * n_models)
    
    blend_auc = roc_auc_score(y, rank_oof)
    
    return rank_oof, blend_auc


def meta_ensemble(oof_dict, y, strategy='auto'):
    """
    Advanced meta-ensemble with 2-layer stacking.
    
    Args:
        oof_dict: dict of {model_name: oof_predictions}
        y: target variable
        strategy: 'auto', 'stacking', 'blending', or 'ranking'
    
    Returns:
        result: dict with ensemble results
    """
    model_names = list(oof_dict.keys())
    n_models = len(model_names)
    
    # Step 1: Greedy forward selection
    selected, individual_scores = greedy_forward_selection(oof_dict, y)
    
    if len(selected) < 2:
        selected = model_names[:min(3, n_models)]
    
    # Step 2: Try different ensemble methods
    results = {}
    
    # Method 1: Simple average of selected models
    simple_oof = np.mean([oof_dict[m] for m in selected], axis=0)
    results['simple_avg'] = {
        'oof': simple_oof,
        'auc': roc_auc_score(y, simple_oof),
        'models': selected
    }
    
    # Method 2: Weighted average
    weights, weights_auc = optimize_blend_weights(
        {m: oof_dict[m] for m in selected}, y
    )
    weighted_oof = np.zeros(len(y))
    for m in selected:
        weighted_oof += oof_dict[m] * weights[m]
    results['weighted_avg'] = {
        'oof': weighted_oof,
        'auc': weights_auc,
        'weights': weights,
        'models': selected
    }
    
    # Method 3: Rank average
    rank_oof, rank_auc = rank_average_blend(
        {m: oof_dict[m] for m in selected}, y
    )
    results['rank_avg'] = {
        'oof': rank_oof,
        'auc': rank_auc,
        'models': selected
    }
    
    # Method 4: Ridge stacking (Layer 2)
    meta_oof_matrix = np.column_stack([oof_dict[m] for m in selected])
    
    # OOF predictions for meta-learner
    ridge_meta_oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for tr, va in skf.split(meta_oof_matrix, y):
        ridge = Ridge(alpha=1000.0)
        ridge.fit(meta_oof_matrix[tr], y[tr])
        ridge_meta_oof[va] = ridge.predict(meta_oof_matrix[va])
    
    results['ridge_stacking'] = {
        'oof': ridge_meta_oof,
        'auc': roc_auc_score(y, ridge_meta_oof),
        'models': selected
    }
    
    # Select best method
    best_method = max(results.keys(), key=lambda k: results[k]['auc'])
    best_result = results[best_method]
    
    return {
        'best_method': best_method,
        'best_oof': best_result['oof'],
        'best_auc': best_result['auc'],
        'best_models': best_result['models'],
        'all_results': results,
        'individual_scores': individual_scores
    }
