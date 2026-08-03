#!/usr/bin/env python3
"""
Model Training Script for Binary Classification
Supports multiple algorithms with cross-validation
"""

import sys
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

# Try to import gradient boosting libraries
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


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


def get_models(random_state=42):
    """Get dictionary of model pipelines"""
    models = {
        'logistic': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000, 
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
        'rf': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        'histgb': HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=10,
            learning_rate=0.1,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=random_state
        )
    }
    
    if HAS_LGB:
        models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1
        )
    
    if HAS_XGB:
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric='auc'
        )
    
    if HAS_CAT:
        models['catboost'] = cb.CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            class_weights=[1, 1],
            random_seed=random_state,
            verbose=False,
            thread_count=-1
        )
    
    return models


def prepare_data(train_path, test_path, target_col):
    """Load and prepare data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate features and target
    X = train_df.drop(columns=[target_col])
    y = train_df[target_col]
    X_test = test_df.copy()

    # DROP id-like identifier columns (e.g. row_id, id): near-unique string
    # columns. Label-encoding them either leaks test identities or crashes on
    # unseen labels. Numeric high-cardinality columns are real features - kept.
    n = len(X)
    drop_cols = [
        col for col in X.columns
        if col in X_test.columns
        and X_test[col].dtype in ('object', 'string', 'category')   # string-ish
        and X[col].dtype in ('object', 'string', 'category')
        and X[col].nunique() / n > 0.9                              # near-unique
    ]
    if drop_cols:
        print(f"Dropping id-like columns: {drop_cols}")
        X = X.drop(columns=drop_cols)
        X_test = X_test.drop(columns=drop_cols)

    # Handle categorical columns (encode on union of train+test so no unseen
    # labels can appear; missing values are filled first and treated as a token).
    cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns
    for col in cat_cols:
        combined = pd.concat([X[col], X_test[col]]).astype(str).fillna('__NA__')
        le = LabelEncoder()
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str).fillna('__NA__'))
        if col in X_test.columns:
            X_test[col] = le.transform(X_test[col].astype(str).fillna('__NA__'))

    # Handle missing values in numeric features
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in ['float64', 'int64']:
                fill_val = X[col].median()
            else:
                fill_val = X[col].mode()[0] if not X[col].mode().empty else 0
            X[col] = X[col].fillna(fill_val)
            if col in X_test.columns:
                X_test[col] = X_test[col].fillna(fill_val)

    return X, y, X_test


def train_and_evaluate(X, y, models, n_folds=5, random_state=42):
    """Train models with cross-validation"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = {}
    oof_predictions = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            # Cross-validation predictions.
            # IMPORTANT: n_jobs must be 1 here - models already use n_jobs=-1 internally,
            # and joblib outer-parallelism + threaded GBMs (LightGBM/XGBoost) deadlock.
            oof_pred = cross_val_predict(
                model, X, y, 
                cv=skf, 
                method='predict_proba',
                n_jobs=1
            )[:, 1]
            
            auc = roc_auc_score(y, oof_pred)
            results[name] = {
                'cv_auc': float(auc),
                'cv_auc_std': 0.0  # Could compute per-fold std
            }
            oof_predictions[name] = oof_pred
            print(f"  {name}: AUC = {auc:.4f}")
            
            # Fit on full data for later use
            model.fit(X, y)
            
        except Exception as e:
            print(f"  {name} failed: {e}")
            results[name] = {'error': str(e)}
    
    return results, oof_predictions


def predict_test(models, X_test):
    """Generate test predictions from all models"""
    test_preds = {}
    for name, model in models.items():
        try:
            if hasattr(model, 'predict_proba'):
                test_preds[name] = model.predict_proba(X_test)[:, 1]
            else:
                test_preds[name] = model.predict(X_test)
        except Exception as e:
            print(f"  {name} prediction failed: {e}")
            test_preds[name] = None
    return test_preds


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else 'train.csv'
    test_path = sys.argv[2] if len(sys.argv) > 2 else 'test.csv'
    target_col = sys.argv[3] if len(sys.argv) > 3 else 'target'
    output_dir = sys.argv[4] if len(sys.argv) > 4 else '.'
    
    print("Loading data...")
    X, y, X_test = prepare_data(train_path, test_path, target_col)
    print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    models = get_models()
    print(f"Available models: {list(models.keys())}")
    
    print("\nTraining with cross-validation...")
    results, oof_preds = train_and_evaluate(X, y, models)
    
    print("\nGenerating test predictions...")
    test_preds = predict_test(models, X_test)
    
    # Save results
    output = {
        'cv_results': results,
        'oof_predictions': {k: v.tolist() for k, v in oof_preds.items()},
        'test_predictions': {k: v.tolist() if v is not None else None for k, v in test_preds.items()},
        'best_model': max(results, key=lambda k: results[k].get('cv_auc', 0)) if results else None
    }
    
    with open(f'{output_dir}/training_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Save models
    for name, model in models.items():
        with open(f'{output_dir}/model_{name}.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    print("\nTraining complete!")
    print(f"Best model: {output['best_model']}")
    if output['best_model']:
        print(f"Best CV AUC: {results[output['best_model']]['cv_auc']:.4f}")
    
    return output


if __name__ == '__main__':
    main()