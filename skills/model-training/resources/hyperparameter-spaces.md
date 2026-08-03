# Hyperparameter Search Spaces

## Optuna Search Spaces

### LightGBM
```python
{
    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
    'max_depth': trial.suggest_int('max_depth', 3, 12),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
    'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),
    'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10, log=True)
}
```

### XGBoost
```python
{
    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
    'max_depth': trial.suggest_int('max_depth', 3, 10),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    'gamma': trial.suggest_float('gamma', 0, 5),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2)
}
```

### CatBoost
```python
{
    'iterations': trial.suggest_int('iterations', 200, 500),
    'depth': trial.suggest_int('depth', 4, 10),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
    'border_count': trial.suggest_int('border_count', 32, 255),
    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
    'random_strength': trial.suggest_float('random_strength', 1e-3, 10, log=True)
}
```

### Random Forest
```python
{
    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
    'max_depth': trial.suggest_int('max_depth', 5, 25),
    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
    'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
    'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0, 0.01)
}
```

### HistGradientBoosting
```python
{
    'max_iter': trial.suggest_int('max_iter', 100, 500),
    'max_depth': trial.suggest_int('max_depth', 3, 15),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
    'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10, log=True),
    'max_bins': trial.suggest_int('max_bins', 64, 255),
    'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 31, 127)
}
```

### Logistic Regression
```python
{
    'C': trial.suggest_float('C', 1e-3, 10, log=True),
    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
    'l1_ratio': trial.suggest_float('l1_ratio', 0, 1)  # for elasticnet
}
```

## Optimization Budget

| Scenario | Trials per Model | Total Time |
|----------|------------------|------------|
| Quick (15 min) | 10-20 | ~5 min |
| Standard (30 min) | 30-50 | ~15 min |
| Thorough (45 min) | 50-100 | ~25 min |

## Tips for Competition

1. **Start with default params** - often work well
2. **Optimize top 2 models only** - diminishing returns
3. **Use early stopping** - saves time
4. **Cache best params** - reuse across similar datasets
5. **Prune unpromising trials** - Optuna does this automatically