#!/usr/bin/env python3
"""
Final Submission Selection Script
"""

import sys
import json
import os
from pathlib import Path


# --- cwd-robust bootstrap (skills may run in a temp dir; data lives in /work) --
def _pick_workdir():
    _cwd = os.getcwd()
    for _base in dict.fromkeys([_cwd, "/work", "/kaggle/working"]):
        if os.path.exists(os.path.join(_base, "train.csv")) or os.path.exists(
            os.path.join(_base, "sample_submission.csv")
        ):
            return _base
    return _cwd

os.chdir(_pick_workdir())


def select_final_submissions(history_path='submission_history.json', n=2):
    """Select final submissions for leaderboard"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    subs = history['submissions']
    
    # Prefer public scores, fallback to CV
    with_public = [s for s in subs if 'public_score' in s]
    
    if with_public:
        best = sorted(with_public, key=lambda x: x['public_score'], reverse=True)[:n]
        score_type = 'public_score'
    else:
        with_cv = [s for s in subs if 'cv_score' in s]
        best = sorted(with_cv, key=lambda x: x['cv_score'], reverse=True)[:n]
        score_type = 'cv_score'
    
    selected = []
    for i, sub in enumerate(best):
        selected.append({
            'rank': i + 1,
            'submission_id': sub['id'],
            'score': sub[score_type],
            'score_type': score_type,
            'model': sub.get('model'),
            'ensemble_type': sub.get('ensemble_type'),
            'file': sub.get('file')
        })
    
    return selected


def main():
    history_path = sys.argv[1] if len(sys.argv) > 1 else 'submission_history.json'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    selected = select_final_submissions(history_path, n)
    
    print("=" * 60)
    print(f"FINAL SELECTION (Top {n})")
    print("=" * 60)
    
    for sel in selected:
        print(f"\nRank {sel['rank']}:")
        print(f"  Submission #{sel['submission_id']}")
        print(f"  {sel['score_type'].upper()} AUC: {sel['score']:.4f}")
        print(f"  Model: {sel['model']}")
        print(f"  Ensemble: {sel['ensemble_type']}")
        print(f"  File: {sel['file']}")
    
    # Save selection
    with open('final_selection.json', 'w') as f:
        json.dump(selected, f, indent=2)
    
    print(f"\nSelection saved to final_selection.json")
    
    # Print command for select_submission tool
    sub_ids = [str(s['submission_id']) for s in selected]
    print(f"\nTo select in competition: select_submission {','.join(sub_ids)}")


if __name__ == '__main__':
    main()