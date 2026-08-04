#!/usr/bin/env python3
"""
Dataset Optimizer Module - Per-dataset hyperparameter tuning and model selection

Key techniques:
1. Optuna TPE Bayesian optimization
2. Dataset-specific hyperparameters
3. Dataset-specific feature selection
4. Dataset-specific model weights
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score


def analyze_dataset(X, y, cats):
    """
    Analyze dataset characteristics to guide optimization.
    
    Returns:
        analysis: dict with dataset characteristics
    """
    n_samples = len(X)
    n_features = X.shape[1]
    n_numeric = len([c for c in X.columns if c not in cats and pd.api.types.is_numeric_dtype(X[c])])
    n_categorical = len(cats)
    
    # Missing values
    missing_pct = X.isna().mean().mean()
    
    # Class balance
    class_balance = np.min(np.bincount(y)) / np.max(np.bincount(y))
    
    # Feature-to-sample ratio
    ratio = n_features / n_samples
    
    # Dataset size category
    if n_samples < 1000:
        size_category = 'tiny'
    elif n_samples < 5000:
        size_category = 'small'
    elif n_samples < 20000:
        size_category = 'medium'
    else:
        size_category = 'large'
    
    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'n_numeric': n_numeric,
        'n_categorical': n_categorical,
        'missing_pct': missing_pct,
        'class_balance': class_balance,
        'ratio': ratio,
        'size_category': size_category
    }


def get_dataset_specific_params(analysis, model_type):
    """
    Get dataset-specific hyperparameters based on analysis.
    
    Args:
        analysis: dict from analyze_dataset()
        model_type: 'lgb', 'xgb', 'cat', 'hgb', 'rf', 'et'
    
    Returns:
        params: dict of hyperparameters
    """
    n = analysis['n_samples']
    ratio = analysis['ratio']
    missing = analysis['missing_pct']
    
    # Base parameters
    if model_type == 'lgb':
        params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        # Adjust for dataset size
        if n < 1000:
            params['n_estimators'] = 200
            params['num_leaves'] = 15
            params['min_child_samples'] = 10
            params['reg_alpha'] = 1.0
            params['reg_lambda'] = 1.0
        elif n < 5000:
            params['n_estimators'] = 300
            params['num_leaves'] = 20
            params['min_child_samples'] = 15
        elif n > 20000:
            params['n_estimators'] = 1000
            params['num_leaves'] = 50
            params['min_child_samples'] = 30
        
        # Adjust for feature ratio
        if ratio > 0.05:
            params['colsample_bytree'] = 0.6
            params['reg_alpha'] = 1.0
            params['reg_lambda'] = 1.0
    
    elif model_type == 'xgb':
        params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }
        
        if n < 1000:
            params['n_estimators'] = 200
            params['max_depth'] = 4
            params['min_child_weight'] = 5
            params['reg_alpha'] = 1.0
        elif n > 20000:
            params['n_estimators'] = 1000
            params['max_depth'] = 8
        
        if ratio > 0.05:
            params['colsample_bytree'] = 0.6
            params['max_depth'] = 4
    
    elif model_type == 'cat':
        params = {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'random_seed': 42
        }
        
        if n < 1000:
            params['iterations'] = 200
            params['depth'] = 4
            params['l2_leaf_reg'] = 10.0
        elif n > 20000:
            params['iterations'] = 1000
            params['depth'] = 8
    
    elif model_type == 'hgb':
        params = {
            'max_iter': 500,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_samples_leaf': 20,
            'l2_regularization': 1.0
        }
        
        if n < 1000:
            params['max_iter'] = 200
            params['max_depth'] = 4
            params['min_samples_leaf'] = 10
            params['l2_regularization'] = 10.0
        elif n > 20000:
            params['max_iter'] = 1000
            params['max_depth'] = 8
    
    elif model_type == 'rf':
        params = {
            'n_estimators': 500,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        
        if n < 1000:
            params['n_estimators'] = 200
            params['max_depth'] = 10
            params['min_samples_split'] = 10
            params['min_samples_leaf'] = 5
        elif n > 20000:
            params['n_estimators'] = 1000
            params['max_depth'] = 20
    
    elif model_type == 'et':
        params = {
            'n_estimators': 500,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        
        if n < 1000:
            params['n_estimators'] = 200
            params['max_depth'] = 10
        elif n > 20000:
            params['n_estimators'] = 1000
            params['max_depth'] = 20
    
    else:
        params = {}
    
    return params


def tune_with_optuna(X, y, cats, model_type, n_trials=30, timeout=300):
    """
    Tune hyperparameters using Optuna TPE.
    
    Args:
        X: Training features
        y: Target variable
        cats: Categorical columns
        model_type: 'lgb', 'xgb', 'cat', 'hgb'
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
    
    Returns:
        best_params: dict of best hyperparameters
        best_score: best CV score
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("WARNING: optuna not available, using dataset-specific params")
        analysis = analyze_dataset(X, y, cats)
        return get_dataset_specific_params(analysis, model_type), 0.0
    
    def objective(trial):
        if model_type == 'lgb':
            try:
                import lightgbm as lgbm
            except ImportError:
                return 0.0
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
                'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            model = lgbm.LGBMClassifier(**params)
        
        elif model_type == 'xgb':
            try:
                import xgboost as xgbm
            except ImportError:
                return 0.0
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
                'random_state': 42, 'verbosity': 0, 'n_jobs': -1
            }
            model = xgbm.XGBClassifier(**params, tree_method='hist', eval_metric='auc')
        
        elif model_type == 'cat':
            try:
                from catboost import CatBoostClassifier
            except ImportError:
                return 0.0
            
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_seed': 42, 'verbose': 0, 'allow_writing_files': False
            }
            model = CatBoostClassifier(**params, cat_features=cats if cats else None)
        
        elif model_type == 'hgb':
            from sklearn.ensemble import HistGradientBoostingClassifier
            
            params = {
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
                'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10, log=True),
                'random_state': 42
            }
            model = HistGradientBoostingClassifier(**params)
        
        else:
            return 0.0
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params, study.best_value


def optimize_dataset(X, y, cats, model_types=['lgb', 'xgb', 'hgb', 'rf', 'et'], 
                     use_optuna=True, n_trials=20):
    """
    Optimize all models for a specific dataset.
    
    Returns:
        optimized_params: dict of {model_type: best_params}
        analysis: dict with dataset characteristics
    """
    analysis = analyze_dataset(X, y, cats)
    
    optimized_params = {}
    
    for model_type in model_types:
        if use_optuna:
            try:
                best_params, best_score = tune_with_optuna(
                    X, y, cats, model_type, n_trials=n_trials
                )
                optimized_params[model_type] = best_params
                print(f"  {model_type}: CV AUC = {best_score:.5f}")
            except Exception as e:
                print(f"  {model_type}: Optuna failed ({e}), using dataset-specific params")
                optimized_params[model_type] = get_dataset_specific_params(analysis, model_type)
        else:
            optimized_params[model_type] = get_dataset_specific_params(analysis, model_type)
    
    return optimized_params, analysis
