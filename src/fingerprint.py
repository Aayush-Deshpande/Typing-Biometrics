"""
Keystroke Cadence Fingerprint Engine
=====================================
Implements spec.md Phases 1-3 exactly.

Phase 1: Raw events → per-key and per-digraph records
Phase 2: Fingerprint vector construction (400-600 dims)
Phase 3: Sparse intersection comparison
"""

import json
import os
import sys
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# The 5 prompts (must match app.py — single source of truth)
# ---------------------------------------------------------------------------
ALL_PROMPTS = [
    "Hey there, how's it going",
    "What are your plans for today",
    "Nice weather we are having huh",
    "Hope you are having a good day",
    "Just checking in, how are you",
]

MODIFIER_KEYS = {"Shift", "Control", "Alt", "Meta", "CapsLock"}

SENTINEL = -1.0  # marker for absent dimensions


# ===========================================================================
# Phase 1 — Raw events → per-key and per-digraph records
# ===========================================================================

def parse_events(raw_events_json: str) -> Tuple[List[Tuple[str, float, float]], List[float], int]:
    """
    Parse raw browser keystroke events into a clean ordered list.

    Returns
    -------
    keystrokes : list of (key, dwell_ms, flight_ms)
        Ordered by keyup time. flight_ms is 0.0 for the first key.
        Times are in SECONDS (divided by 1000).
    shift_overlaps : list of float
        For each capital letter, the overlap time between Shift-keydown
        and the letter-keydown (in seconds).
    backspace_count : int
        Number of Backspace keydown events.
    """
    events = json.loads(raw_events_json) if isinstance(raw_events_json, str) else raw_events_json

    # Track Shift keydown times for overlap calculation
    shift_down_times: deque = deque()
    backspace_count = 0
    shift_overlaps: List[float] = []

    # Per-key deque for pairing keydown → keyup
    pending: Dict[str, deque] = defaultdict(deque)
    keystrokes_raw = []  # (key, kd_time, ku_time)

    for ev in events:
        key = ev["key"]
        typ = ev["type"]
        t = ev["time"]

        if key == "Backspace" and typ == "keydown":
            backspace_count += 1

        if key == "Shift":
            if typ == "keydown":
                shift_down_times.append(t)
            elif typ == "keyup" and shift_down_times:
                shift_down_times.popleft()
            continue  # Don't include Shift itself as a keystroke

        if key in MODIFIER_KEYS:
            continue

        if typ == "keydown":
            # Check if this is a capital letter typed while Shift is held
            if shift_down_times and len(key) == 1 and key.isupper():
                overlap = (t - shift_down_times[-1]) / 1000.0
                shift_overlaps.append(overlap)

            # Only record the FIRST keydown (ignore e.repeat)
            if not pending[key]:
                pending[key].append(t)
        elif typ == "keyup":
            if pending[key]:
                kd_time = pending[key].popleft()
                keystrokes_raw.append((key, kd_time, t))

    # Sort by keyup time (chronological order of completion)
    keystrokes_raw.sort(key=lambda x: x[2])

    # Build (key, dwell, flight) triples
    keystrokes = []
    for i, (key, kd, ku) in enumerate(keystrokes_raw):
        dwell = (ku - kd) / 1000.0  # seconds
        if i == 0:
            flight = 0.0
        else:
            _, _, prev_ku = keystrokes_raw[i - 1]
            flight = (kd - prev_ku) / 1000.0  # seconds
        # Clip outliers
        dwell = max(-2.0, min(2.0, dwell))
        flight = max(-2.0, min(2.0, flight))
        keystrokes.append((key.lower(), dwell, flight))

    return keystrokes, shift_overlaps, backspace_count


# ===========================================================================
# Phase 2.1 — Build vocabularies from prompts (programmatically)
# ===========================================================================

def build_vocabs(prompts: List[str]) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    Derive key_vocab, digraph_vocab, trigraph_vocab from the prompt pool.
    All characters are lowercased.

    Returns sorted lists (these become fixed column indices).
    """
    all_chars = set()
    all_digraphs = set()
    all_trigraphs = set()

    for prompt in prompts:
        chars = list(prompt.lower())
        for c in chars:
            all_chars.add(c)
        for i in range(len(chars) - 1):
            all_digraphs.add((chars[i], chars[i + 1]))
        for i in range(len(chars) - 2):
            all_trigraphs.add((chars[i], chars[i + 1], chars[i + 2]))

    key_vocab = sorted(all_chars)
    digraph_vocab = sorted(all_digraphs)
    trigraph_vocab = sorted(all_trigraphs)

    return key_vocab, digraph_vocab, trigraph_vocab


# Pre-compute global vocabs at module load
_KEY_VOCAB, _DIGRAPH_VOCAB, _TRIGRAPH_VOCAB = build_vocabs(ALL_PROMPTS)


def get_vocabs():
    """Return the global vocabs."""
    return _KEY_VOCAB, _DIGRAPH_VOCAB, _TRIGRAPH_VOCAB


def get_total_dims() -> int:
    """Return the total dimensionality of the fingerprint vector."""
    return (
        5 * len(_KEY_VOCAB)
        + 4 * len(_DIGRAPH_VOCAB)
        + 1 * len(_TRIGRAPH_VOCAB)
        + 13  # structural (added rollover + entropy)
        + 3   # positional
    )


# ===========================================================================
# Phase 2.2-2.6 — Fingerprint vector construction
# ===========================================================================

def build_fingerprint(
    keystrokes: List[Tuple[str, float, float]],
    shift_overlaps: List[float],
    backspace_count: int,
    key_vocab: List[str] = None,
    digraph_vocab: List[Tuple[str, str]] = None,
    trigraph_vocab: List[Tuple[str, str, str]] = None,
) -> List[float]:
    """
    Construct the full sparse fingerprint vector from parsed keystrokes.

    Parameters
    ----------
    keystrokes : list of (key, dwell, flight)
    shift_overlaps : list of shift-to-letter overlap times
    backspace_count : int
    key_vocab, digraph_vocab, trigraph_vocab : vocab lists (use globals if None)

    Returns
    -------
    list[float] of length total_dims. Uses SENTINEL (-1.0) for absent dimensions.
    """
    if key_vocab is None:
        key_vocab = _KEY_VOCAB
    if digraph_vocab is None:
        digraph_vocab = _DIGRAPH_VOCAB
    if trigraph_vocab is None:
        trigraph_vocab = _TRIGRAPH_VOCAB

    # ── Collect per-key dwells and per-digraph flights ──────────────────
    per_key_dwells: Dict[str, List[float]] = defaultdict(list)
    per_digraph_flights: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    per_trigraph_times: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)

    all_dwells = []
    all_flights = []
    space_flights = []

    for i, (key, dwell, flight) in enumerate(keystrokes):
        per_key_dwells[key].append(dwell)
        all_dwells.append(dwell)

        if i > 0:
            prev_key = keystrokes[i - 1][0]
            per_digraph_flights[(prev_key, key)].append(flight)
            all_flights.append(flight)

            if key == " ":
                space_flights.append(flight)

        if i >= 2:
            k0 = keystrokes[i - 2][0]
            k1 = keystrokes[i - 1][0]
            k2 = key
            # Total time spanning the trigraph = flight into k1 + flight into k2
            f1 = keystrokes[i - 1][2]  # flight of k1
            f2 = flight                # flight of k2
            per_trigraph_times[(k0, k1, k2)].append(f1 + f2)

    # ── Step 2.2: Per-key features (5 per key) ──────────────────────────
    fp = []
    for key in key_vocab:
        vals = per_key_dwells.get(key, [])
        if not vals:
            fp.extend([SENTINEL] * 5)
        else:
            arr = np.array(vals, dtype=np.float64)
            fp.append(float(np.mean(arr)))             # mean dwell
            fp.append(float(np.std(arr)) if len(arr) > 1 else 0.0)  # std dwell
            fp.append(float(np.min(arr)))              # min dwell
            fp.append(float(np.max(arr)))              # max dwell
            fp.append(float(len(arr)))                 # occurrence count

    # ── Step 2.3: Per-digraph features (4 per digraph) ──────────────────
    for dg in digraph_vocab:
        vals = per_digraph_flights.get(dg, [])
        if not vals:
            fp.extend([SENTINEL] * 4)
        else:
            arr = np.array(vals, dtype=np.float64)
            fp.append(float(np.mean(arr)))             # mean flight
            fp.append(float(np.std(arr)) if len(arr) > 1 else 0.0)  # std flight
            fp.append(float(np.min(arr)))              # min flight
            fp.append(float(np.max(arr)))              # max flight

    # ── Step 2.4: Per-trigraph features (1 per trigraph) ─────────────────
    for tg in trigraph_vocab:
        vals = per_trigraph_times.get(tg, [])
        if not vals:
            fp.append(SENTINEL)
        else:
            fp.append(float(np.mean(vals)))            # mean total time

    # ── Step 2.5: Structural features (13 fixed) ───────────────────────
    # shift_overlap_mean
    if shift_overlaps:
        fp.append(float(np.mean(shift_overlaps)))
        fp.append(float(np.std(shift_overlaps)) if len(shift_overlaps) > 1 else 0.0)
    else:
        fp.append(SENTINEL)
        fp.append(SENTINEL)

    # space_flight_mean, std, max
    if space_flights:
        fp.append(float(np.mean(space_flights)))
        fp.append(float(np.std(space_flights)) if len(space_flights) > 1 else 0.0)
        fp.append(float(np.max(space_flights)))
    else:
        fp.append(0.0)
        fp.append(0.0)
        fp.append(0.0)

    # backspace_count
    fp.append(float(backspace_count))

    # total_typing_duration
    if keystrokes:
        total_dur = sum(ks[1] for ks in keystrokes) + sum(ks[2] for ks in keystrokes[1:])
        fp.append(float(total_dur))
    else:
        fp.append(0.0)

    # total_keystroke_count
    fp.append(float(len(keystrokes)))

    # rollover_count (number of times flight time is negative)
    rollover_count = sum(1 for f in all_flights if f < 0)
    fp.append(float(rollover_count))

    # rhythm_entropy (Shannon entropy of flight times to detect robotic typing)
    if len(all_flights) > 5:
        hist, _ = np.histogram(all_flights, bins=10, density=True)
        # Avoid log(0)
        p = hist[hist > 0]
        p = p / np.sum(p)
        entropy = -np.sum(p * np.log2(p))
        fp.append(float(entropy))
    else:
        fp.append(0.0)

    # overall_mean_dwell
    fp.append(float(np.mean(all_dwells)) if all_dwells else 0.0)

    # overall_mean_flight
    fp.append(float(np.mean(all_flights)) if all_flights else 0.0)

    # overall_std_flight
    fp.append(float(np.std(all_flights)) if len(all_flights) > 1 else 0.0)

    # ── Step 2.6: Positional features (3 fixed) ────────────────────────
    if len(keystrokes) >= 3:
        n = len(keystrokes)
        third = n // 3
        first_flights = [ks[2] for ks in keystrokes[1:third + 1]]
        mid_flights = [ks[2] for ks in keystrokes[third + 1:2 * third + 1]]
        last_flights = [ks[2] for ks in keystrokes[2 * third + 1:]]

        fp.append(float(np.mean(first_flights)) if first_flights else 0.0)
        fp.append(float(np.mean(mid_flights)) if mid_flights else 0.0)
        fp.append(float(np.mean(last_flights)) if last_flights else 0.0)
    else:
        fp.extend([0.0, 0.0, 0.0])

    return fp


# ===========================================================================
# Phase 3 — Comparison logic (sparse intersection)
# ===========================================================================

MIN_OVERLAP = 8  # minimum comparable dimensions for a valid comparison


def compare_fingerprints(
    fp_a: List[float],
    fp_b: List[float],
    min_overlap: int = MIN_OVERLAP,
) -> Tuple[Optional[float], int]:
    """
    Compare two fingerprint vectors using relative difference.
    Scale-invariant: automatically prevents structural domination.
    """
    a = np.array(fp_a, dtype=np.float64)
    b = np.array(fp_b, dtype=np.float64)

    # Find dimensions where BOTH have real values
    mask = (a != SENTINEL) & (b != SENTINEL)
    intersection_size = int(np.sum(mask))

    if intersection_size < min_overlap:
        return 0.0, intersection_size

    a_int = a[mask]
    b_int = b[mask]

    # Relative difference per dimension: |a_i - b_i| / ((a_i + b_i) / 2)
    avg = (np.abs(a_int) + np.abs(b_int)) / 2.0
    
    # Handle near-zero denominators by treating the relative difference as 0 if both are near 0
    rel_diffs = np.where(
        avg < 1e-9,
        0.0,
        np.abs(a_int - b_int) / avg,
    )

    dissimilarity = float(np.mean(rel_diffs))
    
    # Exponential scaling to map typical genuine dissimilarities (~0.30 - 0.55) into the >0.90 range
    # and impostor dissimilarities (~0.80+) into the <0.88 range.
    match_score = float(np.exp(-dissimilarity / 6.0))
    
    return match_score, intersection_size


# ===========================================================================
# Convenience: full pipeline from raw JSON to fingerprint
# ===========================================================================

def events_to_fingerprint(raw_events_json: str) -> List[float]:
    """
    Full pipeline: raw events JSON → fingerprint vector.
    """
    keystrokes, shift_overlaps, backspace_count = parse_events(raw_events_json)
    return build_fingerprint(keystrokes, shift_overlaps, backspace_count)


# ===========================================================================
# Standalone test
# ===========================================================================

if __name__ == "__main__":
    kv, dv, tv = get_vocabs()
    total_dims = get_total_dims()

    print(f"=== Vocabulary Report ===")
    print(f"Key vocab ({len(kv)}): {kv}")
    print(f"Digraph vocab ({len(dv)}): {len(dv)} pairs")
    print(f"Trigraph vocab ({len(tv)}): {len(tv)} triples")
    print(f"\nTotal fingerprint dimensionality: {total_dims}")
    print(f"  = 5×{len(kv)} + 4×{len(dv)} + 1×{len(tv)} + 11 + 3")

    # Test on existing DB data if available
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB = os.path.join(_ROOT, "keystroke_auth.db")
    if os.path.exists(DB):
        import sqlite3
        conn = sqlite3.connect(DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, source, prompt_text, raw_events_json FROM raw_sequences ORDER BY id LIMIT 4"
        ).fetchall()
        conn.close()

        if rows:
            print(f"\n=== Testing on {len(rows)} existing samples ===")
            fingerprints = []
            for row in rows:
                ks, so, bc = parse_events(row["raw_events_json"])
                fp = build_fingerprint(ks, so, bc)
                fingerprints.append(fp)

                real_count = sum(1 for v in fp if v != SENTINEL)
                sentinel_count = sum(1 for v in fp if v == SENTINEL)
                print(f"\nRow {row['id']} ({row['source']}, \"{row['prompt_text'][:30]}...\"):")
                print(f"  Keystrokes: {len(ks)}")
                print(f"  Fingerprint dims: {len(fp)} (real: {real_count}, sentinel: {sentinel_count})")
                print(f"  Per-key dwells sample: ", end="")
                pkd = defaultdict(list)
                for k, d, f in ks:
                    pkd[k].append(round(d, 4))
                for k in sorted(pkd.keys())[:5]:
                    print(f"'{k}':{pkd[k]}", end="  ")
                print()

            # Compare fingerprints
            if len(fingerprints) >= 2:
                print(f"\n=== Fingerprint Comparisons ===")
                for i in range(len(fingerprints)):
                    for j in range(i + 1, len(fingerprints)):
                        score, overlap = compare_fingerprints(fingerprints[i], fingerprints[j])
                        score_str = f"{score:.4f}" if score is not None else "N/A"
                        print(f"  Row {rows[i]['id']} vs Row {rows[j]['id']}: "
                              f"score={score_str}, overlap={overlap}")
        else:
            print("\nNo samples in DB. Register a user first.")
    else:
        print(f"\nNo DB found at {DB}. Skipping DB test.")
