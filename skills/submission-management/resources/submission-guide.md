# Submission Guide for Kaggle Autonomous Agent Competition

> **Competition limits (binding):** maximum **5 submissions per day** and at most **2 Final Submissions**. The harness may allow more (e.g. 30), but the official Competition-Specific Rules (2.1, 2.2) are the hard ceiling. Always respect them. Use single-account entry only; use only competition-provided data (CC BY 4.0); do not share or transmit competition data outside the sandbox.

## Submission Workflow

### 1. Create Predictions
- Generate test predictions from trained models
- Save as JSON with multiple ensemble variants
- Keys: 'weighted_ensemble', 'rank_average', 'simple_average', 'individual'

### 2. Format Submission
- Match sample_submission.csv exactly
- Same row order, same column names
- Prediction column should be probabilities [0, 1]
- ID column must match sample

### 3. Validate Before Submitting
```python
# Checklist:
- [ ] Shape matches sample
- [ ] Columns match sample
- [ ] No NaN values
- [ ] Predictions in [0, 1]
- [ ] Not all same value
- [ ] ID column order matches
```

### 4. Submit
- Use `submit_predictions` tool
- Track submission ID returned
- Log model, ensemble type, CV score

### 5. Track Public Scores
- After scoring, note public AUC
- Update submission history
- Use for model selection

## Budget Management

### Time (60 min)
- 0-10 min: Data exploration
- 10-35 min: Model training + optimization
- 35-50 min: Ensembling + submissions
- 50-60 min: Final selection

### Submissions (5 max per day, official)
- Official competition cap is **5 submissions per day** (harness may allow more — do not exceed 5).
- Reserve 1-2 for final selection among your **2 Final Submissions**.
- Submit diverse, meaningfully-different approaches; don't waste submissions on minor variations.

### Cost ($2.00)
- Track via `get_status`
- Prefer cheaper models if budget tight
- gemini-3.1-flash-lite for simple tasks
- gemini-3.5-flash for complex reasoning

## Submission Strategy

### Early Submissions (Exploration)
1. Simple average of base models
2. Best single model (LightGBM)
3. Rank average

### Mid Submissions (Optimization)
4. Weighted ensemble (optimized)
5. Different model subsets
6. Stacking variant

### Late Submissions (Refinement)
7. Best weighted + best rank combo
8. Feature-engineered variant
9. Different random seeds

### Final Selection
- Pick top 2 by public score
- If no public scores, use CV
- Diversify: different ensemble methods

## Common Mistakes

1. **Wrong format**: ID column mismatch, wrong column names
2. **Data leakage**: Using test info in training
3. **Overfitting public LB**: Trust CV more
4. **Wasting submissions**: Submitting near-duplicates
5. **Ignoring budgets**: Running out of time/submissions/cost
6. **No tracking**: Can't select best at the end

## Tools to Use

- `submit_predictions`: Submit CSV file
- `get_status`: Check budgets, scores
- `select_submission`: Choose final 2
- `run_skill_script`: Run submission scripts