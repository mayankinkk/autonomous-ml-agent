---
name: data-exploration
description: Load, profile, and understand binary-classification datasets. Detects the target column, feature types, missing values, class balance, and feature-target relationships so the agent can design an effective preprocessing and modeling pipeline. Saves exploration_results.json and feature_profiles.json for later steps.
---

# Data Exploration Skill

Use this skill first, immediately after listing the working directory. It produces the insight needed to preprocess and model correctly.

## Available Scripts

### `scripts/explore.py`
Reads `train.csv`, `test.csv`, and `sample_submission.csv` from the working directory and writes `exploration_results.json`.

Usage:
```
python scripts/explore.py train.csv test.csv sample_submission.csv
```

What it reports:
- Train/test shape, dtypes, memory usage, head rows
- Detected target column and class distribution / balance
- Per-feature type (`numeric`, `categorical_low_cardinality`, `categorical_medium_cardinality`, `categorical_high_cardinality`, `binary`, `constant`, `datetime`)
- Missing-value summary per column
- Feature-target correlations (for numeric/binary features)

### `scripts/profile.py`
Reads `train.csv` plus an optional target column name; writes `feature_profiles.json`.

```
python scripts/profile.py train.csv [target_col]
```

What it adds:
- Per-feature skewness, kurtosis, IQR, outlier count, zero/negative percent
- Categorical entropy, top categories, most-common percent

## Workflow Guidance
1. Run `scripts/explore.py` first and read `exploration_results.json`.
2. Identify the target column (often `target`, `label`, or `y`).
3. Note class imbalance — pass `class_weight='balanced'` or `scale_pos_weight` to models.
4. Note which features are constant/quasi-constant (drop them) and which have missing values.
5. Decide encoding: one-hot or target-encode low-cardinality, drop or hash high-cardinality, treat binary as 0/1 numeric.
6. If needed, run `scripts/profile.py` for skew (log-transform) and outlier checks.

Keep stdout summaries compact to conserve token budget.