#!/usr/bin/env python3
"""
Advanced Data Profiling Script
Generates detailed statistical profiles for each feature
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats


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


def profile_numeric(series, name):
    """Profile a numeric feature"""
    clean = series.dropna()
    if len(clean) == 0:
        return {'type': 'numeric', 'error': 'All values missing'}
    
    return {
        'type': 'numeric',
        'count': int(series.count()),
        'missing': int(series.isnull().sum()),
        'mean': float(clean.mean()),
        'std': float(clean.std()),
        'min': float(clean.min()),
        'max': float(clean.max()),
        'median': float(clean.median()),
        'skewness': float(stats.skew(clean)) if len(clean) > 2 else 0,
        'kurtosis': float(stats.kurtosis(clean)) if len(clean) > 3 else 0,
        'q25': float(clean.quantile(0.25)),
        'q75': float(clean.quantile(0.75)),
        'iqr': float(clean.quantile(0.75) - clean.quantile(0.25)),
        'zeros_pct': float((clean == 0).sum() / len(clean) * 100),
        'negative_pct': float((clean < 0).sum() / len(clean) * 100),
        'outlier_count': int(((clean < (clean.quantile(0.25) - 1.5 * (clean.quantile(0.75) - clean.quantile(0.25)))) | 
                               (clean > (clean.quantile(0.75) + 1.5 * (clean.quantile(0.75) - clean.quantile(0.25))))).sum())
    }


def profile_categorical(series, name, max_categories=50):
    """Profile a categorical feature"""
    clean = series.dropna()
    if len(clean) == 0:
        return {'type': 'categorical', 'error': 'All values missing'}
    
    value_counts = clean.value_counts()
    top_categories = value_counts.head(max_categories)
    
    return {
        'type': 'categorical',
        'count': int(series.count()),
        'missing': int(series.isnull().sum()),
        'n_unique': int(series.nunique()),
        'top_categories': top_categories.to_dict(),
        'entropy': float(stats.entropy(value_counts.values)) if len(value_counts) > 1 else 0,
        'most_common_pct': float(value_counts.iloc[0] / len(clean) * 100) if len(value_counts) > 0 else 0
    }


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else 'train.csv'
    target_col = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(train_path).exists():
        print(f"File not found: {train_path}")
        return
    
    df = pd.read_csv(train_path)
    
    if target_col is None:
        # Try to guess target
        for col in df.columns:
            if df[col].nunique() == 2:
                target_col = col
                break
    
    profiles = {}
    
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        
        if pd.api.types.is_numeric_dtype(series):
            profiles[col] = profile_numeric(series, col)
        else:
            profiles[col] = profile_categorical(series, col)
    
    # Save profiles
    with open('feature_profiles.json', 'w') as f:
        json.dump(profiles, f, indent=2, default=str)
    
    print(f"Profiled {len(profiles)} features")
    print("Results saved to feature_profiles.json")


if __name__ == '__main__':
    main()