# LOCKED SPEC — Keystroke Cadence Fingerprint Verification

## (Do not deviate. Do not "simplify" to averages. Do not propose alternatives mid-build.)

This replaces the embedding/LSTM approach AND the 4-average approach. Both were wrong for different reasons:

- LSTM/CMU approach: domain mismatch, training data wrong shape, kept giving meaningless similarity scores
- 4-average approach: too coarse, can't distinguish people, "checks typing speed not typing pattern"

This spec uses **per-key and per-digraph statistical fingerprints** — high-dimensional, interpretable, no neural network, no external dataset needed. This is the FINAL architecture. Build exactly this.

---

## Core Concept

Instead of one global number per metric, compute **one set of stats per individual key AND per consecutive key-pair (digraph)** that actually appeared in the typed sentence. A person's "fingerprint" is the collection of all these per-unit stats — typically 40-80 dimensions depending on sentence length/variety.

Two people typing "Hey there, how's it going" might have near-identical overall WPM, but their dwell time on "h" specifically, and their flight time for "th", "he", "ow", "space→h" etc. will differ — THAT'S the signal.

---

## Phase 1 — Raw events → per-key and per-digraph records

Input: same raw `{key, type, time}` array from Stage 1 (unchanged — frontend capture stays as-is).

1. Parse into a clean ordered list of `(key, dwell_time, flight_time_from_previous_key)` — same pairing logic as before (handle `e.repeat`: if multiple keydowns occur before a keyup for the same key, IGNORE the repeats, only use the first keydown + the eventual keyup).
2. From this ordered list, build TWO dictionaries:
   - `per_key_dwells = {key: [list of dwell times for this key in this sample]}`
     - e.g. if "e" appears 3 times in the sentence, this has 3 values
   - `per_digraph_flights = {(prev_key, curr_key): [list of flight times for this pair]}`
     - e.g. "he" appearing twice → 2 values under `('h','e')`

**DoD:** print both dictionaries for one real sample (the existing 4 sequences from Stage 1/2 testing) — confirm keys/digraphs match the actual sentence text, and lists have realistic millisecond values.

---

## Phase 2 — Fingerprint vector construction (EXHAUSTIVE — generate ALL of these, not a sample)

**Do not pick "example" features. Generate every feature in every category below, programmatically, for the full vocabulary derived from the 5 prompts. If a category yields 70+ columns, that's correct — build all of them.**

### Step 2.1 — Build vocabularies from the 5 prompts (programmatically, not by hand)

- `key_vocab` = sorted set of every unique character across all 5 prompts (including space)
- `digraph_vocab` = sorted set of every unique consecutive 2-char pair across all 5 prompts
- `trigraph_vocab` = sorted set of every unique consecutive 3-char pair across all 5 prompts
- Print all three vocab sizes and lists.

### Step 2.2 — For EVERY key in `key_vocab`, compute (if key present in sample, else -1):

- mean dwell time
- std dwell time (use 0 if only 1 occurrence)
- min dwell time
- max dwell time
- occurrence count in this sample

→ that's 5 columns × len(key_vocab). Do this for ALL keys, no exceptions.

### Step 2.3 — For EVERY digraph in `digraph_vocab`, compute (if present, else -1):

- mean flight time
- std flight time
- min flight time
- max flight time

→ 4 columns × len(digraph_vocab). ALL digraphs.

### Step 2.4 — For EVERY trigraph in `trigraph_vocab`, compute (if present, else -1):

- mean total time (sum of the 2 flights spanning the trigraph)

→ 1 column × len(trigraph_vocab). ALL trigraphs.

### Step 2.5 — Structural features (always computed, never -1):

- shift_overlap_mean: for every capital letter typed, time between Shift-keydown and the letter's keydown (overlap duration) — mean across all capitals in sample. If no capitals, use -1.
- shift_overlap_std: same, std dev
- space_flight_mean: mean flight time specifically INTO spacebar (word-boundary pause) — mean across all words
- space_flight_std: std of the above
- space_flight_max: max word-boundary pause (catches "thinking pauses")
- backspace_count: number of backspace keydowns in this sample
- total_typing_duration: time from first keydown to last keyup
- total_keystroke_count: total number of keys pressed (including repeats/backspace)
- overall_mean_dwell: mean dwell across ALL keys in this sample (not per-key — this IS a global stat, kept as ONE of many features, not the whole picture)
- overall_mean_flight: mean flight across ALL transitions in this sample
- overall_std_flight: std of all flight times in this sample

→ 11 fixed columns, always real values (no -1).

### Step 2.6 — Positional features (first/last thirds of the sentence):

- first_third_mean_flight: mean flight time for the first 1/3 of keystrokes
- middle_third_mean_flight: mean flight time for middle 1/3
- last_third_mean_flight: mean flight time for last 1/3
- (captures "starts slow, speeds up" or "fatigues toward the end" patterns — fatigue/warmup signatures)

→ 3 columns, always real.

### Total dimensionality

`5*len(key_vocab) + 4*len(digraph_vocab) + 1*len(trigraph_vocab) + 11 + 3`

For 5 casual sentences (~20-25 chars each, ~25 unique chars, ~60-80 digraphs, ~80-100 trigraphs), expect roughly **400-600 total dimensions**. This is correct and intentional — do not reduce it. Print the exact final dimensionality.

**DoD:** print exact vocab sizes, exact total dimension count, and for one real sample print: how many of the dimensions are real values vs -1 sentinels (most will be -1 for any single sentence — that's expected, since one sentence only contains a fraction of the full vocab).

---

## Phase 3 — Comparison logic (NO neural network)

Since each sentence has a DIFFERENT subset of populated dimensions (sparse), direct Euclidean/cosine distance on the full vector is wrong (sentinels would dominate). Instead:

1. Write `compare_fingerprints(fp_a, fp_b)`:
   - Find the INTERSECTION of dimensions where BOTH vectors have real values (not -1) — call this the "comparable dimensions"
   - If intersection is too small (< 8 dimensions, configurable constant), return a special "insufficient overlap" result rather than a misleading score
   - On the intersection dimensions only: compute normalized absolute difference per dimension: `|a_i - b_i| / ((a_i + b_i)/2)` (relative difference, handles the fact dwell~0.1s and flight~0.3s are different scales)
   - Average these relative differences → `dissimilarity_score` (lower = more similar)
   - Convert to a match score: `match_score = 1 / (1 + dissimilarity_score)` (bounded 0-1, higher = better match)
2. Print this comparison for: your 3 enrollment fingerprints vs your 1 login fingerprint (all same person, different sentences) — expect moderate-to-high match scores BUT with realistic variance (not 0.99 across the board like before — different sentences = different intersection dims = naturally different scores).

**DoD:** print intersection size and match score for all enrollment-vs-login pairs. Scores should NOT all be identical (that was the red flag last time) — variance here is expected and correct, since different sentence pairs share different digraphs.

---

## Phase 4 — Profile & threshold (carries over Stage 3 concepts, simplified)

1. `profile_fingerprints` table: same structure as `profile_embeddings` but storing these ~80-100 dim sparse vectors instead of 16-dim dense embeddings.
2. On login: compute `compare_fingerprints(login_fp, enrollment_fp)` for EACH stored profile fingerprint, take the MAX match score.
3. Threshold: start at `match_score >= 0.6`, but THIS MUST BE CALIBRATED empirically (Phase 5), not assumed.
4. Online refinement (grow to 20, evict oldest) — same as original Stage 3 C3, unchanged.

---

## Phase 5 — Calibration (mandatory, do not skip)

1. With 5-6 real users (same requirement as before):
   - Genuine: each user logs in 3+ times → match scores against their own profile
   - Impostor: cross-check each user's login fingerprint against OTHER users' profiles
2. Print genuine score distribution vs impostor score distribution.
3. **This is the actual proof the system works.** If genuine and impostor scores don't separate here, the issue is either (a) vocabulary too small/prompts too similar — widen the prompt pool, or (b) the relative-difference formula needs adjustment — but do NOT add back a neural network. Debug within this framework.
4. Pick threshold at the separation point, document with real numbers.

---

## What gets DELETED from the existing codebase

- `models/encoder.h5`, `encoder_finetuned.h5`, all CMU-related files (`cmu_features.py`, `train_encoder.py`)
- `embeddings` / `profile_embeddings` tables (replaced by `profile_fingerprints`)
- Any TensorFlow/Keras dependency — this system needs only numpy/pandas/sklearn

## What STAYS unchanged

- Stage 1 frontend capture (JS keydown/keyup, `performance.now()`)
- Flask routes, SQLite `users`/`raw_sequences` tables, dashboard structure
- Prompt pool (5 sentences)
- Stage 3's clustering concept (DBSCAN) CAN still apply on top of fingerprints in Phase 4 if useful — optional, not required for v1

---

## Build order (strict)

1. Phase 1 → 2 → 3 on EXISTING 4 samples (you, real data already captured) — get DoD passing before touching the DB/Flask integration
2. Phase 4 — wire into Flask, fresh DB
3. Get 5-6 users to register/login (you'll need to ask people)
4. Phase 5 — calibrate, report real numbers

**No model files. No "let me try a different approach." This vector + comparison function IS the model.**
