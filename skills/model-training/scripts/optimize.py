#!/usr/bin/env python3
"""
Hyperparameter Optimization Script using Optuna
"""

import sys
import json
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# --- cwd-robust bootstrap (skills may run in a temp dir; data lives in /work) --
def _pick_workdir():
    _cwd = os.getcwd()
    for _base in dict.fromkeys([_cwd, "/work", "/kaggle/working"]):
        if os.path.exists(os.path.join(_base, "train.csv")) or os.path.exists(
            os.path.join(_base, "sample_submission.csv")
        ):
            return _base
    return _cwd

os.chdir(_pick_workdir())


try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def prepare_data(train_path, target_col):
    """Load and prepare data"""
    train_df = pd.read_csv(train_path)
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]

    # Drop id-like high-cardinality string columns (e.g. row_id)
    n = len(X)
    drop_cols = [
        col for col in X.columns
        if X[col].dtype in ('object', 'string', 'category')
        and X[col].nunique() / n > 0.9
    ]
    if drop_cols:
        X = X.drop(columns=drop_cols)

    cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
    
    return X, y


def objective_lgb(trial, X, y):
    """LightGBM objective"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }
    model = lgb.LGBMClassifier(**params)
    return cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring='roc_auc', n_jobs=1).mean()


def objective_xgb(trial, X, y):
    """XGBoost objective"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
        'eval_metric': 'auc'
    }
    model = xgb.XGBClassifier(**params)
    return cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring='roc_auc', n_jobs=1).mean()


def objective_rf(trial, X, y):
    """Random Forest objective"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    return cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring='roc_auc', n_jobs=1).mean()


def objective_histgb(trial, X, y):
    """HistGradientBoosting objective"""
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10, log=True),
        'class_weight': 'balanced',
        'random_state': 42
    }
    model = HistGradientBoostingClassifier(**params)
    return cross_val_score(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                          scoring='roc_auc', n_jobs=1).mean()


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else 'train.csv'
    target_col = sys.argv[2] if len(sys.argv) > 2 else 'target'
    model_name = sys.argv[3] if len(sys.argv) > 3 else 'lightgbm'
    n_trials = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    
    print(f"Loading data from {train_path}...")
    X, y = prepare_data(train_path, target_col)
    print(f"Data shape: {X.shape}")
    
    objectives = {
        'lightgbm': objective_lgb if HAS_LGB else None,
        'xgboost': objective_xgb if HAS_XGB else None,
        'rf': objective_rf,
        'histgb': objective_histgb
    }
    
    if model_name not in objectives or objectives[model_name] is None:
        print(f"Model {model_name} not available. Available: {[k for k, v in objectives.items() if v]}")
        return
    
    print(f"Optimizing {model_name} with {n_trials} trials...")
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objectives[model_name](trial, X, y), n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest trial:")
    print(f"  Value: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'model': model_name
    }
    
    with open(f'optuna_{model_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to optuna_{model_name}_results.json")


if __name__ == '__main__':
    main()