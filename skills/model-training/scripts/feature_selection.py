#!/usr/bin/env python3
"""
Feature Selection Module - Intelligent feature pruning and selection
for maximum model performance.

Key techniques:
1. Feature importance pruning (remove bottom 20%)
2. Correlation-based removal (remove highly correlated features)
3. Mutual information selection
4. Permutation importance validation
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold


def get_feature_importance(X, y, cats, n_folds=3, seed=42):
    """
    Get feature importance using multiple methods and average them.
    
    Returns:
        importance_df: DataFrame with feature names and importance scores
    """
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    
    numeric_cols = [c for c in X.columns if c not in cats and pd.api.types.is_numeric_dtype(X[c])]
    
    # Method 1: Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    rf.fit(X[numeric_cols].fillna(0), y)
    rf_importance = pd.Series(rf.feature_importances_, index=numeric_cols)
    
    # Method 2: Extra Trees importance
    et = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    et.fit(X[numeric_cols].fillna(0), y)
    et_importance = pd.Series(et.feature_importances_, index=numeric_cols)
    
    # Method 3: Mutual Information
    mi_scores = mutual_info_classif(X[numeric_cols].fillna(0), y, random_state=seed)
    mi_importance = pd.Series(mi_scores, index=numeric_cols)
    
    # Normalize and average
    rf_norm = rf_importance / rf_importance.max() if rf_importance.max() > 0 else rf_importance
    et_norm = et_importance / et_importance.max() if et_importance.max() > 0 else et_importance
    mi_norm = mi_importance / mi_importance.max() if mi_importance.max() > 0 else mi_importance
    
    avg_importance = (rf_norm + et_norm + mi_norm) / 3
    
    importance_df = pd.DataFrame({
        'feature': numeric_cols,
        'rf_importance': rf_importance.values,
        'et_importance': et_importance.values,
        'mi_importance': mi_importance.values,
        'avg_importance': avg_importance.values
    }).sort_values('avg_importance', ascending=False)
    
    return importance_df


def remove_correlated_features(X, threshold=0.95):
    """
    Remove highly correlated features to reduce redundancy.
    
    Args:
        X: DataFrame with features
        threshold: correlation threshold (default 0.95)
    
    Returns:
        X_reduced: DataFrame with reduced features
        removed: list of removed feature names
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return X, []
    
    corr_matrix = X[numeric_cols].corr().abs()
    
    # Get upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation > threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Remove from DataFrame
    X_reduced = X.drop(columns=to_drop, errors='ignore')
    
    return X_reduced, to_drop


def select_features(X, y, cats, importance_threshold=0.2, max_features=50):
    """
    Select the best features using importance pruning.
    
    Args:
        X: Training features
        y: Target variable
        cats: Categorical columns
        importance_threshold: Minimum importance (as percentile)
        max_features: Maximum number of features to keep
    
    Returns:
        X_selected: DataFrame with selected features
        selected_features: list of selected feature names
        importance_df: Feature importance DataFrame
    """
    # Get feature importance
    importance_df = get_feature_importance(X, y, cats)
    
    # Select features above threshold
    min_importance = importance_df['avg_importance'].quantile(importance_threshold)
    important_features = importance_df[importance_df['avg_importance'] >= min_importance]['feature'].tolist()
    
    # Limit to max_features
    if len(important_features) > max_features:
        top_features = importance_df.head(max_features)['feature'].tolist()
        important_features = [f for f in important_features if f in top_features]
    
    # Always keep categorical columns
    keep_features = important_features + [c for c in cats if c in X.columns]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in keep_features:
        if f not in seen:
            seen.add(f)
            unique_features.append(f)
    
    X_selected = X[unique_features].copy()
    
    return X_selected, unique_features, importance_df


def feature_selection_pipeline(X, y, cats, strategy='moderate'):
    """
    Complete feature selection pipeline.
    
    Args:
        X: Training features
        y: Target variable
        cats: Categorical columns
        strategy: 'conservative', 'moderate', or 'aggressive'
    
    Returns:
        X_selected: DataFrame with selected features
        selected_features: list of selected feature names
        report: dict with selection statistics
    """
    original_features = X.shape[1]
    
    # Strategy settings
    settings = {
        'conservative': {'importance_threshold': 0.3, 'max_features': 60, 'corr_threshold': 0.98},
        'moderate': {'importance_threshold': 0.2, 'max_features': 50, 'corr_threshold': 0.95},
        'aggressive': {'importance_threshold': 0.1, 'max_features': 40, 'corr_threshold': 0.90}
    }
    
    config = settings.get(strategy, settings['moderate'])
    
    # Step 1: Remove highly correlated features
    X_reduced, corr_removed = remove_correlated_features(X, config['corr_threshold'])
    
    # Step 2: Select features by importance
    X_selected, selected_features, importance_df = select_features(
        X_reduced, y, cats,
        importance_threshold=config['importance_threshold'],
        max_features=config['max_features']
    )
    
    # Update cats list to only include selected categorical columns
    selected_cats = [c for c in cats if c in selected_features]
    
    report = {
        'original_features': original_features,
        'corr_removed': len(corr_removed),
        'importance_pruned': len(selected_features),
        'final_features': X_selected.shape[1],
        'selected_cats': len(selected_cats),
        'selected_numeric': X_selected.shape[1] - len(selected_cats),
        'top_10_features': importance_df.head(10)[['feature', 'avg_importance']].to_dict('records')
    }
    
    return X_selected, selected_features, selected_cats, report, importance_df
