# Feature Type Detection Guide

## Type Categories

### Numeric
- **Continuous**: Float values, wide range (age, price, temperature)
- **Discrete**: Integer counts (number of items, frequency)
- **Categorical encoded as numeric**: Zip codes, IDs with numeric dtype

### Categorical
- **Low cardinality** (< 20 unique): One-hot encoding works well
- **Medium cardinality** (20-100): Target encoding, frequency encoding
- **High cardinality** (> 100): Hashing, embedding, or drop

### Binary
- Two unique values (0/1, True/False, Yes/No, M/F)
- Treat as numeric (0/1) or categorical

### Datetime
- Parse to datetime
- Extract: year, month, day, dayofweek, hour, quarter, is_weekend
- Cyclical encoding for month/day/hour

### Constant / Quasi-constant
- Single unique value: Drop
- > 99% same value: Consider dropping

## Detection Heuristics

```python
def detect_type(series):
    n_unique = series.nunique()
    n_total = len(series)
    unique_ratio = n_unique / n_total
    
    if n_unique <= 1:
        return 'constant'
    elif n_unique == 2:
        return 'binary'
    elif pd.api.types.is_numeric_dtype(series):
        if unique_ratio < 0.05 and n_unique < 20:
            return 'categorical_numeric'
        return 'numeric'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    else:
        if unique_ratio < 0.01:
            return 'categorical_low_cardinality'
        elif unique_ratio < 0.1:
            return 'categorical_medium_cardinality'
        else:
            return 'categorical_high_cardinality'
```

## Preprocessing by Type

| Type | Missing | Encoding | Scaling |
|------|---------|----------|---------|
| Numeric | Median/Mean | None | StandardScaler/MinMax |
| Binary | Mode | 0/1 | Optional |
| Categorical Low | Mode | OneHot/Ordinal | None |
| Categorical Medium | Mode | Target/Frequency | None |
| Categorical High | Mode | Hashing/Embedding | None |
| Datetime | Mode/Forward fill | Extract features | None |

## Competition-Specific Tips

For this competition family:
1. Run detection on first dataset, cache patterns
2. Apply same logic to subsequent datasets
3. Watch for consistent feature naming (e.g., "feat_0", "feat_1")
4. Target often named "target", "label", or "y"