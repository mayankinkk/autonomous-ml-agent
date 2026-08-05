---
name: model-training
description: >-
  Train and cross-validate binary-classification models (LightGBM, XGBoost,
  CatBoost, HistGradientBoosting, Random Forest, Logistic Regression), optimize
  hyperparameters with Optuna, and build weighted/rank/simple ensemble predictions.
  Writes training_results.json, model_*.pkl, and ensemble_predictions.csv.
  IMPORTANT engine note: scripts use n_jobs=1 for outer cross-validation to avoid
  joblib/thread deadlocks with gradient boosting.
---

# Model Training Skill

Use this skill after data exploration, once the target column and feature types are known.

## Available Scripts

### `scripts/aggressive.py` (HIGHEST AUC - run this FIRST)
Maximum AUC pipeline with aggressive feature engineering (log/sqrt/rank transforms, interactions, row aggregates, z-scores, binned features), all 4 model families (LGB/XGB/CAT/HGB) with 5-fold CV and 3 seeds each, optimized weighted blending via scipy, and ridge stacking. Writes `submission.csv` (best), `sub_blend_optimized.csv`, `sub_ridge_stacking.csv`, `sub_ridge_rank_stacking.csv`, plus per-family `sub_*.csv`.

```
python skills/model-training/scripts/aggressive.py
```

### `scripts/compete.py` (fallback - if aggressive.py fails)
Stage-dispatched implementation of the strongest known recipe for this family. One script, staged execution:

```
python scripts/compete.py safety            # quick LightGBM/HGB holdout baseline, writes sub_safety.csv
python scripts/compete.py <fam> [fast]      # per-family OOF CV: xgb | lgb | cat | hgb
python scripts/compete.py blend             # rank-average blend of saved OOF/test preds, writes sub_blend_*.csv
```

- `safety`: quick `train_test_split` (stratified, fallback unstratified) LightGBM (fallback HGB) model, writes `sub_safety.csv` immediately so you always have a valid submission early.
- `<fam>`: OOF cross-validation (3 folds fast / 5 folds full) with target encoding + frequency encoding, per-family: `xgb` (XGBoost), `lgb` (LightGBM), `cat` (CatBoost - uses native categoricals, keeps string cols), `hgb` (HistGradientBoostingClassifier - native categoricals, numeric-only). Writes `oof_<fam>.npy`, `test_<fam>.npy`, `sub_<fam>.csv`. Use `fast` when n_train >= 5000 or time is tight.
- `blend`: loads all `oof_*.npy`/`test_*.npy`, rank-averages [0,1], scores each on OOF AUC, writes `sub_blend_all.csv` and `sub_blend_top2.csv`.
- Target encoding: OOF smoothed (smoothing=10) + frequency encoding, computed with the SAME fold scheme used for training (no leakage). String cat columns are DROPPED for lgb/xgb/hgb and kept only for catboost. `ord_N` strings are parsed to numeric. Rows with NaN target are dropped.
- Validated locally across 16 real family datasets: mean private AUC 0.7995 vs 0.7976 for baseline_robust (12/16 wins). On Kaggle with catboost+xgboost available it can only improve.

Workflow: run `safety` first (always a valid submission), then the available families, then `blend`. Prefer the best-scoring blend submission. Falls back to `scripts/baseline_robust.py` if `compete.py` fails for any reason.

### `scripts/baseline_robust.py` (fallback - only if compete.py fails)
Single reliable pass: `HistGradientBoostingClassifier` with a **dtype-robust native categorical mask** (object OR pandas `StringDtype` - critical under pandas>=3.0 where text columns are `StringDtype`), no LabelEncoder, no imputation (HGB handles missing natively), data-driven `l2`/`min_samples_leaf` gates from the feature-to-row ratio, and **K=10 seed-averaging** when `(n_object_cols > 0) or (n_train >= 5000)`. Reads `train.csv`/`test.csv`/`sample_submission.csv` from the working dir and writes `submission.csv` (`[row_id, target]`, test row order). Validated ~0.97 AUC on some family datasets.

```
python scripts/baseline_robust.py
```

### `scripts/train.py`
Trains all available models with cross-validation, computes OOF AUC for each, generates test predictions. Skips libraries that are not installed. Writes `training_results.json`, `model_<name>.pkl`.

```
python scripts/train.py train.csv test.csv <target_col> <output_dir>
```

### `scripts/optimize.py`
Uses Optuna (TPE) to search hyperparameters for one model. Writes `optuna_<model>_results.json`.

```
python scripts/optimize.py train.csv <target_col> <model> <n_trials>
```

Supported models: `lightgbm`, `xgboost`, `rf`, `histgb`. Use a modest number of trials (e.g., 20-40) to respect the time budget.

### `scripts/ensemble.py`
Loads OOF and test predictions from `training_results.json`, optimizes blend weights on OOF AUC (scipy Nelder-Mead), and also computes rank-average and simple-average ensembles. Writes `ensemble_predictions.json` and `ensemble_predictions.csv` (weighted ensemble). The `rank_average` key is normalized to [0,1].

```
python scripts/ensemble.py training_results.json <target_col> train.csv ensemble_predictions.csv
```

## Workflow Guidance
1. Run `scripts/train.py` to get baseline CV AUC across all models.
2. If the dataset is informative and time allows, run `scripts/optimize.py` on the 1-2 best gradient-boosting models.
3. Run `scripts/ensemble.py` to combine the best diverse models (tree + linear adds robustness).
4. Prefer `weighted_ensemble` for submissions; it is optimized on OOF AUC.
5. Use out-of-fold predictions for any model selection — never the test predictions — to avoid leakage.

## Engineering Notes
- Cross-validation is configured with `n_jobs=1` in the outer fold loop while each model uses its own internal threads. Do NOT switch these to `n_jobs=-1`; that combination deadlocks (joblib + threaded LightGBM/XGBoost) and would hang the session.
- Default several gradient-boosting params are already balanced/sanitized for binary classification.