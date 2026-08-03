# Submission Format Validation

## Required Format

The submission CSV must exactly match `sample_submission.csv`:

```csv
id,target
0,0.5
1,0.3
2,0.7
...
```

## Validation Rules

### 1. Structure
- Same number of rows as test.csv
- Same column names and order
- First column: ID (matches test.csv order)
- Second column: Predicted probability for positive class

### 2. Values
- Predictions must be in range [0, 1]
- No NaN, Inf, or null values
- Numeric type (float)
- Not all identical (degenerate)

### 3. Order
- Row order must match test.csv exactly
- ID column must match test.csv ID column
- No sorting or reordering

## Validation Code

```python
def validate(submission_path, sample_path, test_path=None):
    sub = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)
    
    errors = []
    
    # Shape
    if sub.shape != sample.shape:
        errors.append(f"Shape: {sub.shape} vs {sample.shape}")
    
    # Columns
    if list(sub.columns) != list(sample.columns):
        errors.append(f"Columns: {list(sub.columns)} vs {list(sample.columns)}")
    
    # ID column
    id_col = sample.columns[0]
    if not sub[id_col].equals(sample[id_col]):
        errors.append(f"ID column mismatch")
    
    if test_path:
        test = pd.read_csv(test_path)
        if not sub[id_col].equals(test[id_col]):
            errors.append(f"ID column doesn't match test.csv")
    
    # Predictions
    pred_col = sample.columns[1]
    preds = sub[pred_col]
    
    if preds.isnull().any():
        errors.append("NaN in predictions")
    
    if preds.min() < 0 or preds.max() > 1:
        errors.append(f"Predictions out of range: [{preds.min()}, {preds.max()}]")
    
    if preds.nunique() == 1:
        errors.append("All predictions identical")
    
    return len(errors) == 0, errors
```

## Quick Validation Checklist

Before every submission:
- [ ] `submission.shape == sample.shape`
- [ ] `list(submission.columns) == list(sample.columns)`
- [ ] `submission[id_col].equals(sample[id_col])`
- [ ] `submission[pred_col].between(0, 1).all()`
- [ ] `not submission[pred_col].isnull().any()`
- [ ] `submission[pred_col].nunique() > 1`

## Automation

Run validation automatically:
```bash
python skills/submission-management/scripts/submit.py \
  ensemble_predictions.json \
  sample_submission.csv \
  submission.csv \
  weighted_ensemble
```

This validates and saves `submission_validation.json` with results.