# Ensemble Strategies for Binary Classification

## Ensemble Methods

### 1. Weighted Average (Recommended)
- Optimize weights on OOF predictions using AUC
- Use scipy.optimize or simple grid search
- Weights >= 0, sum to 1

```python
from scipy.optimize import minimize

def optimize_weights(oof_preds, y_true):
    n = len(oof_preds)
    def objective(w):
        w = np.maximum(w, 0)
        w = w / w.sum()
        return -roc_auc_score(y_true, np.average(oof_preds, axis=1, weights=w))
    
    result = minimize(objective, np.ones(n)/n, bounds=[(0,1)]*n)
    return np.maximum(result.x, 0)
```

### 2. Rank Average
- Convert predictions to ranks
- Average ranks
- Robust to outliers, works well for diverse models

```python
ranks = pd.DataFrame(preds).rank()
ensemble = ranks.mean(axis=1)
```

### 3. Simple Average
- Equal weights
- Good baseline, often hard to beat
- Use when models are similar quality

### 4. Stacking (Meta-learner)
- Train meta-model on OOF predictions
- Use Logistic Regression or simple NN
- Risk of overfitting with small data

```python
from sklearn.linear_model import LogisticRegression
meta = LogisticRegression()
meta.fit(oof_preds.T, y_true)
final_pred = meta.predict_proba(test_preds.T)[:, 1]
```

### 5. Blending (Holdout)
- Split train into train/val
- Train base models on train
- Meta-model on val predictions
- Less data for base models

## Practical Strategy for This Competition

### Phase 1: Quick Ensemble (First 30 min)
1. Train 3-4 diverse models (LightGBM, HistGB, Logistic, RF)
2. Simple average predictions
3. Submit, check public score

### Phase 2: Weighted Ensemble (30-45 min)
1. Optimize weights on OOF
2. Try rank average
3. Submit best variant

### Phase 3: Advanced (45-55 min)
1. Stacking with Logistic Regression
2. Try different model subsets
3. Select top 2 submissions

## Model Diversity Checklist

Ensure models are diverse:
- [ ] Different algorithms (tree vs linear)
- [ ] Different feature subsets (if feasible)
- [ ] Different random seeds
- [ ] Different preprocessing (scaled vs unscaled)

## Weight Optimization Tips

1. **Use OOF predictions** - no data leakage
2. **Constrain weights >= 0** - negative weights unstable
3. **Normalize to sum=1** - interpretable
4. **Check correlation** - high corr = less diversity benefit
5. **Min weight threshold** - drop models with weight < 0.05

## Submission Strategy

1. **Submit diverse ensembles**: weighted, rank, simple
2. **Track public scores** - guide weight adjustments
3. **Don't overfit public LB** - trust CV more
4. **Final selection**: best CV + best public combo

## Common Pitfalls

- Using test predictions for weight optimization (leakage)
- Including failed/poor models in ensemble
- Over-optimizing on small validation set
- Ignoring model correlation
- Submitting too many similar predictions