"""
config.py — Centralized configuration constants.
All tunable parameters live here; never hardcode them inline.
"""

# ── Fingerprint comparison threshold ──────────────────────────────────────────
# match_score from compare_fingerprints must be >= this to accept login.
# Threshold set to 0.90 based on cosine similarity metric.
VERIFY_THRESHOLD = 0.90


# ── Profile growth cap ────────────────────────────────────────────────────────
MAX_PROFILE_SIZE = 20

# ── One-Class SVM parameters ─────────────────────────────────────────────────
SVM_NU = 0.15  # upper bound on outlier fraction (lower = stricter)

# ── Enrollment ────────────────────────────────────────────────────────────────
ENROLLMENT_REPS = 3  # how many times each prompt is typed during registration
