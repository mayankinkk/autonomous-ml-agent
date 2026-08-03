#!/usr/bin/env python3
"""
Submission Script - Create and validate submissions
"""

import sys
import json
import os
import pandas as pd
import numpy as np
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


def validate_submission(submission_path, sample_path):
    """Validate submission format matches sample"""
    sub = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)
    
    errors = []
    warnings = []
    
    # Check shape
    if sub.shape != sample.shape:
        errors.append(f"Shape mismatch: submission {sub.shape} vs sample {sample.shape}")
    
    # Check columns
    if list(sub.columns) != list(sample.columns):
        errors.append(f"Column mismatch: {list(sub.columns)} vs {list(sample.columns)}")
    
    # Check row count
    if len(sub) != len(sample):
        errors.append(f"Row count mismatch: {len(sub)} vs {len(sample)}")
    
    # Check for missing values
    if sub.isnull().any().any():
        errors.append("Submission contains NaN values")
    
    # Check prediction range (should be probabilities 0-1)
    pred_col = sub.columns[1] if len(sub.columns) > 1 else sub.columns[0]
    preds = sub[pred_col]
    if preds.min() < 0 or preds.max() > 1:
        warnings.append(f"Predictions outside [0,1] range: min={preds.min():.4f}, max={preds.max():.4f}")
    
    # Check for constant predictions
    if preds.nunique() == 1:
        warnings.append("All predictions are the same value")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'stats': {
            'mean': float(preds.mean()),
            'std': float(preds.std()),
            'min': float(preds.min()),
            'max': float(preds.max()),
            'n_unique': int(preds.nunique())
        }
    }


def create_submission(predictions, sample_path, output_path, id_col=None):
    """Create submission file from predictions"""
    sample = pd.read_csv(sample_path)
    
    if id_col is None:
        id_col = sample.columns[0]
    pred_col = sample.columns[1] if len(sample.columns) > 1 else sample.columns[0]
    
    # Ensure predictions match sample length
    if len(predictions) != len(sample):
        raise ValueError(f"Predictions length {len(predictions)} != sample length {len(sample)}")
    
    submission = sample.copy()
    submission[pred_col] = submission[pred_col].astype(float)
    submission[pred_col] = predictions
    
    # Ensure ID column matches
    if not submission[id_col].equals(sample[id_col]):
        print("Warning: ID column order differs from sample")
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return submission


def main():
    predictions_path = sys.argv[1] if len(sys.argv) > 1 else 'ensemble_predictions.json'
    sample_path = sys.argv[2] if len(sys.argv) > 2 else 'sample_submission.csv'
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'submission.csv'
    pred_key = sys.argv[4] if len(sys.argv) > 4 else 'weighted_ensemble'
    
    # Load predictions
    with open(predictions_path, 'r') as f:
        data = json.load(f)
    
    if pred_key not in data:
        print(f"Key '{pred_key}' not found. Available: {list(data.keys())}")
        return
    
    predictions = data[pred_key]
    
    # Create submission
    submission = create_submission(predictions, sample_path, output_path)
    
    # Validate
    validation = validate_submission(output_path, sample_path)
    
    print("\nValidation Results:")
    print(f"  Valid: {validation['valid']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    print(f"  Stats: {validation['stats']}")
    
    # Save validation report
    with open('submission_validation.json', 'w') as f:
        json.dump(validation, f, indent=2)
    
    if not validation['valid']:
        sys.exit(1)


if __name__ == '__main__':
    main()