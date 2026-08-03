#!/usr/bin/env python3
"""
Submission Tracking Script - Track submissions and scores
"""

import sys
import json
import os
import pandas as pd
from datetime import datetime
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


TRACKING_FILE = 'submission_history.json'


def load_history():
    """Load submission history"""
    if Path(TRACKING_FILE).exists():
        with open(TRACKING_FILE, 'r') as f:
            return json.load(f)
    return {'submissions': [], 'best_public': None, 'best_cv': None}


def save_history(history):
    """Save submission history"""
    with open(TRACKING_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def add_submission(history, submission_info):
    """Add a submission to history"""
    submission_info['timestamp'] = datetime.now().isoformat()
    submission_info['id'] = len(history['submissions']) + 1
    history['submissions'].append(submission_info)
    return history


def update_scores(history, public_scores):
    """Update public scores for recent submissions"""
    # This would be called after getting scores from the competition
    # For now, just log that scores were received
    for i, score in enumerate(public_scores):
        if i < len(history['submissions']):
            history['submissions'][-(i+1)]['public_score'] = score
    return history


def get_best_submissions(history, n=2):
    """Get best submissions by public score, fallback to CV"""
    subs = history['submissions']
    
    # Filter submissions with public scores
    with_public = [s for s in subs if 'public_score' in s]
    
    if with_public:
        # Sort by public score descending
        best = sorted(with_public, key=lambda x: x['public_score'], reverse=True)[:n]
    else:
        # Fallback to CV score
        with_cv = [s for s in subs if 'cv_score' in s]
        best = sorted(with_cv, key=lambda x: x['cv_score'], reverse=True)[:n]
    
    return best


def print_history(history):
    """Print submission history"""
    print("=" * 80)
    print("SUBMISSION HISTORY")
    print("=" * 80)
    
    for sub in history['submissions']:
        print(f"\n#{sub['id']} - {sub['timestamp']}")
        print(f"  Model: {sub.get('model', 'N/A')}")
        print(f"  Ensemble: {sub.get('ensemble_type', 'N/A')}")
        if 'cv_score' in sub:
            print(f"  CV AUC: {sub['cv_score']:.4f}")
        if 'public_score' in sub:
            print(f"  Public AUC: {sub['public_score']:.4f}")
        if 'notes' in sub:
            print(f"  Notes: {sub['notes']}")
    
    best = get_best_submissions(history, 5)
    print("\n" + "=" * 80)
    print("TOP SUBMISSIONS")
    print("=" * 80)
    for i, sub in enumerate(best):
        score = sub.get('public_score', sub.get('cv_score', 0))
        score_type = 'Public' if 'public_score' in sub else 'CV'
        print(f"  {i+1}. #{sub['id']} - {score_type} AUC: {score:.4f} ({sub.get('ensemble_type', 'N/A')})")


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'show'
    
    history = load_history()
    
    if action == 'show':
        print_history(history)
    
    elif action == 'add':
        # Add submission from command line args
        model = sys.argv[2] if len(sys.argv) > 2 else 'unknown'
        ensemble = sys.argv[3] if len(sys.argv) > 3 else 'unknown'
        cv_score = float(sys.argv[4]) if len(sys.argv) > 4 else None
        
        info = {
            'model': model,
            'ensemble_type': ensemble,
            'cv_score': cv_score,
            'file': sys.argv[5] if len(sys.argv) > 5 else 'submission.csv'
        }
        
        history = add_submission(history, info)
        save_history(history)
        print(f"Added submission #{info['id']}")
    
    elif action == 'update':
        # Update with public scores (would come from competition API)
        print("Public score update would be called after competition scoring")
        # Example: python track.py update 0.85 0.83 0.87
        scores = [float(x) for x in sys.argv[2:]]
        history = update_scores(history, scores)
        save_history(history)
        print(f"Updated {len(scores)} submissions with public scores")
    
    elif action == 'best':
        best = get_best_submissions(history, 2)
        print("Best submissions for final selection:")
        for i, sub in enumerate(best):
            score = sub.get('public_score', sub.get('cv_score', 0))
            print(f"  {i+1}. Submission #{sub['id']}: {score:.4f}")
        
        # Save best submission IDs
        with open('best_submissions.json', 'w') as f:
            json.dump([s['id'] for s in best], f)
    
    else:
        print(f"Unknown action: {action}")
        print("Actions: show, add, update, best")


if __name__ == '__main__':
    main()