#!/usr/bin/env python3
"""
Data Exploration Script for Binary Classification
Run with: python scripts/explore.py [train_path] [test_path] [sample_submission_path]
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path


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


def detect_feature_types(df, target_col=None):
    """Detect feature types: numeric, categorical, datetime, binary, constant"""
    feature_info = {}
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        n_unique = series.nunique()
        n_missing = series.isnull().sum()
        pct_missing = n_missing / len(series) * 100
        
        # Check if constant
        if n_unique <= 1:
            ftype = 'constant'
        # Check if binary
        elif n_unique == 2:
            ftype = 'binary'
        # Check if numeric
        elif pd.api.types.is_numeric_dtype(series):
            if n_unique < 20 and n_unique / len(series) < 0.05:
                ftype = 'categorical_numeric'
            else:
                ftype = 'numeric'
        # Check if datetime
        elif pd.api.types.is_datetime64_any_dtype(series):
            ftype = 'datetime'
        # Check if categorical
        else:
            if n_unique / len(series) < 0.05:
                ftype = 'categorical_low_cardinality'
            elif n_unique / len(series) < 0.5:
                ftype = 'categorical_medium_cardinality'
            else:
                ftype = 'categorical_high_cardinality'
        
        feature_info[col] = {
            'type': ftype,
            'n_unique': int(n_unique),
            'n_missing': int(n_missing),
            'pct_missing': round(pct_missing, 2),
            'dtype': str(series.dtype)
        }
    return feature_info


def analyze_target(df, target_col):
    """Analyze target variable distribution"""
    target_series = df[target_col]
    value_counts = target_series.value_counts()
    return {
        'distribution': value_counts.to_dict(),
        'class_balance': round(value_counts.min() / value_counts.max(), 4),
        'n_classes': len(value_counts),
        'missing': int(target_series.isnull().sum())
    }


def compute_correlations(df, target_col, feature_info):
    """Compute correlations with target for numeric features"""
    numeric_cols = [col for col, info in feature_info.items() 
                    if info['type'] in ('numeric', 'categorical_numeric', 'binary')]
    
    if not numeric_cols or target_col not in df.columns:
        return {}
    
    # Include target for correlation
    corr_df = df[numeric_cols + [target_col]].corr()
    target_corrs = corr_df[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    
    return {
        'top_positive': target_corrs[target_corrs > 0].head(10).to_dict(),
        'top_negative': target_corrs[target_corrs < 0].head(10).to_dict(),
        'all': target_corrs.to_dict()
    }


def main():
    # Default paths
    train_path = sys.argv[1] if len(sys.argv) > 1 else 'train.csv'
    test_path = sys.argv[2] if len(sys.argv) > 2 else 'test.csv'
    sample_path = sys.argv[3] if len(sys.argv) > 3 else 'sample_submission.csv'
    
    output = {
        'train': {},
        'test': {},
        'sample_submission': {}
    }
    
    # Load train
    if Path(train_path).exists():
        train_df = pd.read_csv(train_path)
        output['train']['shape'] = list(train_df.shape)
        output['train']['columns'] = list(train_df.columns)
        output['train']['dtypes'] = {k: str(v) for k, v in train_df.dtypes.to_dict().items()}
        output['train']['memory_usage_mb'] = round(train_df.memory_usage(deep=True).sum() / 1024**2, 2)
        output['train']['head'] = train_df.head(3).to_dict(orient='records')
        
        # Try to identify target column
        target_col = None
        for col in train_df.columns:
            if train_df[col].nunique() == 2 and train_df[col].dtype in ['int64', 'int32', 'float64', 'object']:
                target_col = col
                break
        
        if target_col:
            output['train']['target_column'] = target_col
            output['train']['target_analysis'] = analyze_target(train_df, target_col)
            output['train']['feature_types'] = detect_feature_types(train_df, target_col)
            output['train']['correlations'] = compute_correlations(train_df, target_col, output['train']['feature_types'])
            
            # Missing value summary
            missing = train_df.isnull().sum()
            output['train']['missing_summary'] = missing[missing > 0].to_dict()
            
            # Descriptive stats for numeric
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                output['train']['numeric_stats'] = train_df[numeric_cols].describe().to_dict()
    
    # Load test
    if Path(test_path).exists():
        test_df = pd.read_csv(test_path)
        output['test']['shape'] = list(test_df.shape)
        output['test']['columns'] = list(test_df.columns)
        output['test']['dtypes'] = {k: str(v) for k, v in test_df.dtypes.to_dict().items()}
        output['test']['memory_usage_mb'] = round(test_df.memory_usage(deep=True).sum() / 1024**2, 2)
        output['test']['head'] = test_df.head(3).to_dict(orient='records')
        output['test']['feature_types'] = detect_feature_types(test_df)
        missing = test_df.isnull().sum()
        output['test']['missing_summary'] = missing[missing > 0].to_dict()
    
    # Load sample submission
    if Path(sample_path).exists():
        sample_df = pd.read_csv(sample_path)
        output['sample_submission']['shape'] = list(sample_df.shape)
        output['sample_submission']['columns'] = list(sample_df.columns)
        output['sample_submission']['head'] = sample_df.head(3).to_dict(orient='records')
    
    # Save exploration results
    with open('exploration_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Print summary
    print("=" * 60)
    print("DATA EXPLORATION SUMMARY")
    print("=" * 60)
    
    if 'train' in output and output['train']:
        print(f"\nTrain: {output['train'].get('shape', 'N/A')}")
        print(f"Target: {output['train'].get('target_column', 'Unknown')}")
        if 'target_analysis' in output['train']:
            ta = output['train']['target_analysis']
            print(f"  Classes: {ta['distribution']}")
            print(f"  Balance: {ta['class_balance']:.4f}")
        print(f"  Features: {len(output['train'].get('feature_types', {}))}")
        
        # Feature type summary
        ftypes = output['train'].get('feature_types', {})
        type_counts = {}
        for info in ftypes.values():
            type_counts[info['type']] = type_counts.get(info['type'], 0) + 1
        print(f"  Feature types: {type_counts}")
        
        if 'missing_summary' in output['train'] and output['train']['missing_summary']:
            print(f"  Missing values in: {list(output['train']['missing_summary'].keys())}")
    
    if 'test' in output and output['test']:
        print(f"\nTest: {output['test'].get('shape', 'N/A')}")
    
    print("\nFull results saved to exploration_results.json")
    
    return output


if __name__ == '__main__':
    main()