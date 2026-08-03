---
name: submission-management
description: Validate, submit, and track predictions, then select the final submissions for the leaderboard. Ensures every submission matches sample_submission.csv exactly (shape, columns, row order, range 0-1, no NaN) and logs CV/public scores into submission_history.json so the best two can be chosen at session end.
---

# Submission Management Skill

Use this skill after generating predictions and again at the very end of the session to pick the final submissions.

## Available Scripts

### `scripts/submit.py`
Validates a prediction vector against `sample_submission.csv`, writes a formatted `submission.csv`, and reports whether it is `valid`. Fails (exit 1) if the format is wrong.

```
python scripts/submit.py ensemble_predictions.json sample_submission.csv submission.csv <pred_key>
```

Use `pred_key=weighted_ensemble` (or `rank_average` / `simple_average`).

### `scripts/track.py`
Maintains `submission_history.json` and prints the running leaderboard of tracked submissions.

```
python scripts/track.py add <model_name> <ensemble_type> <cv_auc> <file>
python scripts/track.py show
python scripts/track.py update <public_auc> ...
python scripts/track.py best
```

Track every submission you make along with its CV AUC, then record its public AUC after scoring.

### `scripts/select.py`
Chooses the top-N submissions by public score (falling back to CV AUC) and writes `final_selection.json`.

```
python scripts/select.py submission_history.json [n]
```

Prints the exact arguments to pass to the `select_submission` tool.

## Workflow Guidance
1. Before any `submit_predictions` call, run `scripts/submit.py` and confirm `valid: True`.
2. Log each submission with `scripts/track.py add ...` including the CV AUC.
3. After each public score is returned by `get_status`, call `scripts/track.py update` to record it.
4. Near session end (or when time/submissions are low), run `scripts/select.py` to identify the best two and pass their IDs to `select_submission`.
5. Respect the official competition cap of **5 submissions per day** (the harness may allow more) and select at most **2 Final Submissions**. Do not exceed these.

## Validation Rules
- Submission shape must equal `sample_submission.csv`; column names and order must match.
- Row order must match `test.csv` (do not sort/reorder).
- Predictions values in [0, 1], no NaN, not all identical.

## Budget Awareness
- Session: 60 minutes, up to 30 `submit_predictions`, $2.00 LLM spend (per-session harness limits).
- Competition cap (binding even if harness allows more): max **5 submissions/day**, at most **2 Final Submissions**. Stop once a limit is reached.
- Keep LLM context usage low by passing compact data summaries through `get_status`.
- Stop early and finalize the best two submissions when low on time or budget.