#!/usr/bin/env python3
"""
Ensemble Script - Combine predictions from multiple models
"""

import sys
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize


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


def load_predictions(results_path='training_results.json'):
    """Load OOF and test predictions"""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data


def optimize_weights(oof_preds, y_true, method='nelder-mead'):
    """Find optimal ensemble weights"""
    model_names = list(oof_preds.keys())
    n_models = len(model_names)
    
    # Filter out None predictions
    valid_models = {k: v for k, v in oof_preds.items() if v is not None}
    model_names = list(valid_models.keys())
    preds_matrix = np.column_stack([valid_models[k] for k in model_names])
    
    def objective(weights):
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
        ensemble_pred = np.average(preds_matrix, axis=1, weights=weights)
        return -roc_auc_score(y_true, ensemble_pred)
    
    # Initial guess: equal weights
    x0 = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    
    result = minimize(objective, x0, method=method, bounds=bounds,
                     options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    optimal_weights = np.maximum(result.x, 0)
    optimal_weights = optimal_weights / optimal_weights.sum() if optimal_weights.sum() > 0 else x0
    
    return dict(zip(model_names, optimal_weights.tolist()))


def rank_average(predictions):
    """Rank averaging ensemble, normalized to [0, 1]"""
    preds_df = pd.DataFrame(predictions)
    ranks = preds_df.rank()
    avg_ranks = ranks.mean(axis=1).values
    # Normalize ranks to [0,1] so results are valid probabilities
    r_min, r_max = avg_ranks.min(), avg_ranks.max()
    if r_max > r_min:
        avg_ranks = (avg_ranks - r_min) / (r_max - r_min)
    return avg_ranks


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else 'training_results.json'
    target_col = sys.argv[2] if len(sys.argv) > 2 else 'target'
    train_path = sys.argv[3] if len(sys.argv) > 3 else 'train.csv'
    output_path = sys.argv[4] if len(sys.argv) > 4 else 'ensemble_predictions.csv'
    
    print("Loading training results...")
    data = load_predictions(results_path)
    
    oof_preds = data.get('oof_predictions', {})
    test_preds = data.get('test_predictions', {})
    
    if not oof_preds:
        print("No OOF predictions found!")
        return
    
    # Load true labels for weight optimization
    train_df = pd.read_csv(train_path)
    y_true = train_df[target_col].values
    
    print("Optimizing ensemble weights...")
    weights = optimize_weights(oof_preds, y_true)
    print(f"Optimal weights: {weights}")
    
    # Apply weights to test predictions
    valid_test = {k: v for k, v in test_preds.items() if v is not None}
    model_names = list(valid_test.keys())
    test_matrix = np.column_stack([valid_test[k] for k in model_names])
    
    weight_array = np.array([weights.get(k, 0) for k in model_names])
    weight_array = weight_array / weight_array.sum() if weight_array.sum() > 0 else np.ones(len(model_names)) / len(model_names)
    
    ensemble_test = np.average(test_matrix, axis=1, weights=weight_array)
    
    # Also create rank average
    rank_avg_test = rank_average(valid_test)
    
    # Create simple average
    simple_avg_test = test_matrix.mean(axis=1)
    
    # Save ensemble predictions
    output = {
        'weights': weights,
        'weighted_ensemble': ensemble_test.tolist(),
        'rank_average': rank_avg_test.tolist(),
        'simple_average': simple_avg_test.tolist(),
        'individual': {k: v for k, v in valid_test.items()}
    }
    
    with open(output_path.replace('.csv', '.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Save CSV for submission.
    # sample_submission prediction column is often integer-typed; cast to float
    # before assigning probabilities, or pandas raises a dtype TypeError.
    sample_sub = pd.read_csv('sample_submission.csv')
    submission = sample_sub.copy()
    pred_col = submission.columns[1]  # Assume second column is prediction
    submission[pred_col] = submission[pred_col].astype(float)
    submission[pred_col] = ensemble_test
    submission.to_csv(output_path, index=False)
    
    print(f"\nEnsemble predictions saved to {output_path}")
    print(f"Weighted ensemble stats: mean={ensemble_test.mean():.4f}, std={ensemble_test.std():.4f}")
    print(f"Rank average stats: mean={rank_avg_test.mean():.4f}, std={rank_avg_test.std():.4f}")
    print(f"Simple average stats: mean={simple_avg_test.mean():.4f}, std={simple_avg_test.std():.4f}")


if __name__ == '__main__':
    main()