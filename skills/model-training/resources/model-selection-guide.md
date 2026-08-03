# Model Selection Guide for Binary Classification

## Recommended Models (Priority Order)

### 1. LightGBM (Best Overall)
- Fast training, excellent performance
- Handles categorical features natively
- Good with missing values
- **Default choice for this competition**

### 2. XGBoost (Robust Alternative)
- Very stable, well-tested
- Excellent for structured data
- Slightly slower than LightGBM

### 3. CatBoost (Best for Categorical)
- Native categorical handling
- Ordered boosting reduces overfitting
- Good default parameters

### 4. HistGradientBoosting (Sklearn Native)
- No external dependencies
- Fast, histogram-based
- Good baseline

### 5. Random Forest (Baseline)
- Robust, hard to overfit
- Good for feature importance
- Slower inference

### 6. Logistic Regression (Linear Baseline)
- Fast, interpretable
- Good for linearly separable data
- Use with StandardScaler

## Model Selection Strategy

1. **Start with**: LightGBM + HistGradientBoosting + Logistic Regression
2. **If time permits**: Add XGBoost, CatBoost, Random Forest
3. **For ensemble**: Use 3-5 diverse models
4. **Avoid**: Deep learning (tabular data, small datasets), SVM (slow)

## Hyperparameter Priorities

### LightGBM
- `n_estimators`: 100-500 (early stopping)
- `learning_rate`: 0.01-0.1
- `num_leaves`: 31-127
- `max_depth`: -1 (no limit) or 5-15
- `min_child_samples`: 20-50

### XGBoost
- `n_estimators`: 100-500
- `max_depth`: 4-8
- `learning_rate`: 0.01-0.1
- `subsample`: 0.7-0.9
- `colsample_bytree`: 0.7-0.9

### CatBoost
- `iterations`: 200-500
- `depth`: 4-8
- `learning_rate`: 0.01-0.1
- `l2_leaf_reg`: 1-10

## For This Competition Family

Datasets share common characteristics:
- Similar feature distributions
- Consistent target balance
- Same preprocessing pipeline works
- Model rankings transfer across datasets

**Strategy**: Optimize on first dataset, reuse params for subsequent.