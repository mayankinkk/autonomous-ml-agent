# EDA Guide for Binary Classification

## Quick Start Checklist

1. **Load & Inspect**
   - Shape, dtypes, memory usage
   - First/last rows
   - Column names and meanings

2. **Target Analysis**
   - Class distribution (imbalance check)
   - Target type (binary 0/1, -1/1, etc.)
   - Missing values in target

3. **Feature Analysis**
   - Feature types: numeric, categorical, datetime, binary
   - Missing value patterns
   - Cardinality of categoricals
   - Constant/quasi-constant features

4. **Relationships**
   - Feature-target correlations (numeric)
   - Feature-target associations (categorical)
   - Feature-feature correlations (multicollinearity)

5. **Data Quality Issues**
   - Outliers in numeric features
   - Rare categories
   - Data leakage indicators
   - Inconsistent formats

## For This Competition Family

Datasets share a common data-generating process. Look for:
- Similar feature naming patterns
- Consistent target encoding
- Shared feature types across datasets
- Similar class imbalance ratios
- Common preprocessing needs

## Recommended Visualizations (if time permits)
- Target distribution bar chart
- Missing value heatmap
- Correlation heatmap (top 20 features)
- Feature importance from quick model
- Class balance per categorical feature

## Red Flags
- Target leakage (features derived from target)
- Extreme class imbalance (< 1% minority)
- High cardinality categoricals (> 1000 unique)
- Temporal leakage (future data in features)
- Duplicate rows