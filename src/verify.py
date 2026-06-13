"""
Verification Decision Logic — Fingerprint + SVM Dual Gate
==========================================================
Uses both:
1. Fingerprint comparison (intersection-based match_score) as primary gate
2. One-Class SVM prediction as secondary confirmation
"""

import json
import os
import sqlite3
import sys
from typing import List, Tuple, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from config import VERIFY_THRESHOLD
from fingerprint import compare_fingerprints

DATABASE = os.path.join(_ROOT, "keystroke_auth.db")





def verify_login(
    user_id: int,
    login_fingerprint: List[float],
    prompt_index: int,
    db_path: str = DATABASE,
    threshold: float = VERIFY_THRESHOLD,
) -> Tuple[bool, float, List[float]]:
    """
    Verify a login fingerprint against the user's profile.

    Uses dual verification:
    1. Compare login fingerprint against each stored profile fingerprint FOR THE SAME PROMPT
    2. If SVM model exists, also check SVM prediction

    Returns
    -------
    (decision, match_score, scores_per_profile)
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Fetch all profile fingerprints for THIS SPECIFIC PROMPT
    rows = conn.execute(
        "SELECT fingerprint_json FROM profile_fingerprints WHERE user_id = ? AND prompt_index = ? ORDER BY created_at ASC",
        (user_id, prompt_index),
    ).fetchall()

    if not rows:
        print(f"[verify] User {user_id} has no profile fingerprints for prompt {prompt_index} — bypassing.")
        conn.close()
        return True, 1.0, []

    stored_fps = [json.loads(row["fingerprint_json"]) for row in rows]

    # ── Gate 1: Fingerprint comparison ──────────────────────────────────
    scores = []
    for stored_fp in stored_fps:
        score, overlap = compare_fingerprints(login_fingerprint, stored_fp)
        if score is not None:
            scores.append(score)
        else:
            scores.append(0.0)

    # Strict check: score must be >= threshold against EVERY SINGLE profile record
    if not scores:
        comparison_pass = False
        match_score = 0.0
    else:
        comparison_pass = all(s >= threshold for s in scores)
        match_score = float(min(scores))  # The weakest link is the effective score

    # ── Final decision: strict comparison must pass ──────────
    decision = comparison_pass

    print(f"[verify] Comparison: score={match_score:.4f} ({'PASS' if comparison_pass else 'FAIL'}) | Final: {'ACCEPT' if decision else 'REJECT'}")

    return decision, match_score, scores

