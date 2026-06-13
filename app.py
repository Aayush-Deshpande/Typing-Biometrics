import os
import json
import random
import sqlite3
import sys

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, g, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash

# -- Fingerprint engine + SVM verification
_SRC = os.path.join(os.path.abspath(os.path.dirname(__file__)), "src")
sys.path.insert(0, _SRC)

from fingerprint import events_to_fingerprint, compare_fingerprints
from verify import verify_login
from config import VERIFY_THRESHOLD, MAX_PROFILE_SIZE, ENROLLMENT_REPS

EMBEDDINGS_ENABLED = True
VERIFY_ENABLED = True

# -- Stage 4: continuous auth session demo globals
ACS_ALPHA        = 0.6   # EMA smoothing factor (increased for faster hijack response)
DEMO_PROMPTS     = [
    "Hey there, how's it going",
    "What are your plans for today",
    "Nice weather we are having huh",
    "Hope you are having a good day",
    "Just checking in, how are you",
    "Tell me something interesting today",
    "How has your week been going so far",
    "What did you get up to this weekend",
]
GENUINE_PROMPTS  = 4     # first N prompts are genuine; rest are impostor simulation
_CMU_IMPOSTOR_EMB = None  # lazily loaded CMU impostor embedding (cached globally)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

# ── Jinja2 extras ──────────────────────────────────────────────────────────

@app.template_filter("fromjson_len")
def fromjson_len(value: str) -> int:
    """Return the length of a JSON-encoded array (used in dashboard table)."""
    try:
        return len(json.loads(value))
    except (json.JSONDecodeError, TypeError):
        return 0

# Make Python's enumerate available inside Jinja2 templates
app.jinja_env.globals["enumerate"] = enumerate

DATABASE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "keystroke_auth.db")

# ---------------------------------------------------------------------------
# Hardcoded prompt pools (spec §Pages & Flow)
# ---------------------------------------------------------------------------
BASE_PROMPTS = [
    "Hey there, how's it going",
    "What are your plans for today",
    "Nice weather we are having huh",
    "Hope you are having a good day",
    "Just checking in, how are you",
]

# Enrollment: each prompt repeated ENROLLMENT_REPS times = 15 total samples
ENROLLMENT_PROMPTS = BASE_PROMPTS * ENROLLMENT_REPS

LOGIN_PROMPTS = BASE_PROMPTS

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Return a per-request SQLite connection stored on Flask's g object."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create tables if they don't exist yet, and run lightweight migrations."""
    db = get_db()
    db.execute('PRAGMA journal_mode=WAL;')
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS raw_sequences (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL,
            source           TEXT NOT NULL,
            prompt_index     INTEGER,
            prompt_text      TEXT,
            raw_events_json  TEXT NOT NULL,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS profile_fingerprints (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL,
            fingerprint_json TEXT NOT NULL,
            source           TEXT NOT NULL,
            prompt_index     INTEGER NOT NULL,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS user_models (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL UNIQUE,
            model_blob  BLOB NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    db.commit()

    # -- Schema migration: add behavioral columns to raw_sequences if missing
    for col, coltype in [("behavioral_decision", "INTEGER"), ("behavioral_score", "REAL")]:
        try:
            db.execute(f"ALTER TABLE raw_sequences ADD COLUMN {col} {coltype}")
            db.commit()
        except Exception:
            pass  # Column already exists -- safe to ignore


# Run DB init exactly once when the app starts, not on every request
with app.app_context():
    init_db()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


# ── Register ──────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html", prompts=ENROLLMENT_PROMPTS)

    # ── POST ──
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    # Each prompt's events come in as a JSON string in hidden fields
    # named  events_0, events_1, events_2
    events_list = []
    for i in range(len(ENROLLMENT_PROMPTS)):
        raw = request.form.get(f"events_{i}", "[]")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
        events_list.append(parsed)

    for ev in events_list:
        if len(ev) < 15:
            flash("Keystroke sequence too short. Please type the prompts manually (no auto-fill).", "error")
            return render_template("register.html", prompts=ENROLLMENT_PROMPTS)

    if not username or not password:
        flash("Username and password are required.", "error")
        return render_template("register.html", prompts=ENROLLMENT_PROMPTS)

    db = get_db()

    # Check for duplicate username
    existing = db.execute(
        "SELECT id FROM users WHERE username = ?", (username,)
    ).fetchone()
    if existing:
        flash("Username already taken.", "error")
        return render_template("register.html", prompts=ENROLLMENT_PROMPTS)

    # Insert user
    password_hash = generate_password_hash(password)
    cursor = db.execute(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        (username, password_hash),
    )
    db.commit()
    user_id = cursor.lastrowid

    # Compute fingerprints for all enrollment samples
    fingerprints = []
    for idx, (prompt_text_item, events) in enumerate(
        zip(ENROLLMENT_PROMPTS, events_list)
    ):
        db.execute(
            """INSERT INTO raw_sequences
               (user_id, source, prompt_index, prompt_text, raw_events_json)
               VALUES (?, 'enrollment', ?, ?, ?)""",
            (user_id, idx, prompt_text_item, json.dumps(events)),
        )
        db.commit()

        try:
            fp = events_to_fingerprint(json.dumps(events))
            fingerprints.append(fp)
            db.execute(
                "INSERT INTO profile_fingerprints (user_id, fingerprint_json, source, prompt_index) VALUES (?, ?, 'enrollment', ?)",
                (user_id, json.dumps(fp), idx % len(LOGIN_PROMPTS)),
            )
            db.commit()
        except Exception as exc:
            app.logger.warning(f"Fingerprint failed for enrollment sample {idx}: {exc}")


    flash("Registration successful! Please log in.", "success")
    return redirect(url_for("login"))


# ── Login ─────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        prompt_index = random.randrange(len(LOGIN_PROMPTS))
        prompt_text = LOGIN_PROMPTS[prompt_index]
        return render_template(
            "login.html",
            prompt_text=prompt_text,
            prompt_index=prompt_index,
        )

    # ── POST ──
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    try:
        prompt_index = int(request.form.get("prompt_index", 0))
    except (ValueError, TypeError):
        prompt_index = 0
    prompt_text = request.form.get("prompt_text", "")
    raw_events = request.form.get("events_login", "[]")

    try:
        events = json.loads(raw_events)
    except json.JSONDecodeError:
        events = []

    if len(events) < 15:
        flash("Keystroke sequence too short. Please type the prompt manually (no auto-fill).", "error")
        new_index = random.randrange(len(LOGIN_PROMPTS))
        return render_template(
            "login.html",
            prompt_text=LOGIN_PROMPTS[new_index],
            prompt_index=new_index,
        )

    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()

    # First-factor check
    if user is None or not check_password_hash(user["password_hash"], password):
        flash("Invalid username or password.", "error")
        # Do NOT store any row on failed auth
        new_index = random.randrange(len(LOGIN_PROMPTS))
        return render_template(
            "login.html",
            prompt_text=LOGIN_PROMPTS[new_index],
            prompt_index=new_index,
        )

    # Password correct — store the login keystroke sequence
    cursor3 = db.execute(
        """INSERT INTO raw_sequences
           (user_id, source, prompt_index, prompt_text, raw_events_json)
           VALUES (?, 'login', ?, ?, ?)""",
        (user["id"], prompt_index, prompt_text, json.dumps(events)),
    )
    db.commit()
    login_seq_id = cursor3.lastrowid

    # Compute login fingerprint
    login_fp = None
    try:
        login_fp = events_to_fingerprint(json.dumps(events))
    except Exception as exc:
        app.logger.warning(f"Fingerprint failed for login seq {login_seq_id}: {exc}")

    # Behavioral verification
    behavioral_decision = None
    behavioral_score    = None
    if login_fp is not None:
        try:
            b_decision, b_score, _ = verify_login(
                user["id"], login_fp, prompt_index=prompt_index, threshold=VERIFY_THRESHOLD
            )
            behavioral_decision = int(b_decision)
            behavioral_score    = round(b_score, 6)

            db.execute(
                "UPDATE raw_sequences SET behavioral_decision=?, behavioral_score=? WHERE id=?",
                (behavioral_decision, behavioral_score, login_seq_id),
            )
            db.commit()

            print(f"[Login] Behavioral check for '{username}': "
                  f"score={behavioral_score:.4f} threshold={VERIFY_THRESHOLD} "
                  f"decision={'ACCEPT' if b_decision else 'REJECT'}")

            if not b_decision:
                flash(f"Biometric verification failed (Score: {behavioral_score:.4f}). Access Denied.", "error")
                new_index = random.randrange(len(LOGIN_PROMPTS))
                return render_template(
                    "login.html",
                    prompt_text=LOGIN_PROMPTS[new_index],
                    prompt_index=new_index,
                )

            # Online profile refinement — grow profile on accepted logins
            try:
                current_count = db.execute(
                    "SELECT COUNT(*) FROM profile_fingerprints WHERE user_id = ?",
                    (user["id"],),
                ).fetchone()[0]

                if current_count >= MAX_PROFILE_SIZE:
                    oldest = db.execute(
                        """SELECT id FROM profile_fingerprints
                           WHERE user_id = ? AND source = 'login_accepted'
                           ORDER BY created_at ASC LIMIT 1""",
                        (user["id"],),
                    ).fetchone()
                    if oldest:
                        db.execute("DELETE FROM profile_fingerprints WHERE id = ?", (oldest["id"],))
                        db.commit()

                db.execute(
                    "INSERT INTO profile_fingerprints (user_id, fingerprint_json, source, prompt_index) VALUES (?, ?, 'login_accepted', ?)",
                    (user["id"], json.dumps(login_fp), prompt_index),
                )
                db.commit()

            except Exception as exc:
                app.logger.warning(f"Profile growth failed: {exc}")

        except Exception as exc:
            app.logger.warning(f"Behavioral verification failed: {exc}")

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return redirect(url_for("dashboard"))


# ── Dashboard ─────────────────────────────────────────────────────────────

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    db = get_db()
    # Pull this user's most recent raw_sequences rows for display
    rows = db.execute(
        """SELECT source, prompt_index, prompt_text, raw_events_json, created_at
           FROM raw_sequences
           WHERE user_id = ?
           ORDER BY created_at DESC
           LIMIT 10""",
        (session["user_id"],),
    ).fetchall()

    # Compute basic time-delta stats for the most recent row (sanity check)
    sample_stats = None
    if rows:
        latest_events = json.loads(rows[0]["raw_events_json"])
        if len(latest_events) >= 2:
            times = [e["time"] for e in latest_events]
            deltas = [round(times[i + 1] - times[i], 3) for i in range(len(times) - 1)]
            sample_stats = {
                "n_events": len(latest_events),
                "deltas_ms": deltas[:10],  # first 10 gaps
                "min_delta": round(min(deltas), 3),
                "max_delta": round(max(deltas), 3),
            }

    # DoD counts — real DB values, not hardcoded
    enrollment_count = db.execute(
        "SELECT COUNT(*) FROM raw_sequences WHERE user_id = ? AND source = 'enrollment'",
        (session["user_id"],),
    ).fetchone()[0]

    login_count = db.execute(
        "SELECT COUNT(*) FROM raw_sequences WHERE user_id = ? AND source = 'login'",
        (session["user_id"],),
    ).fetchone()[0]

    # ── Stage 2: fingerprint data ──────────────────────────────────────────
    latest_fingerprint = None   # list of floats for most recent sequence
    cosine_sims       = []     # list of {prompt_index, sim, prompt_text}
    stage2_ready      = EMBEDDINGS_ENABLED

    if EMBEDDINGS_ENABLED:
        # Latest fingerprint (any source)
        latest_fp_row = db.execute(
            """SELECT fingerprint_json, source
               FROM profile_fingerprints
               WHERE user_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (session["user_id"],),
        ).fetchone()
        if latest_fp_row:
            latest_fingerprint = [round(v, 3) for v in json.loads(latest_fp_row["fingerprint_json"])]

        # Enrollment fingerprints vs latest fingerprint
        enroll_fp_rows = db.execute(
            """SELECT fingerprint_json, source
               FROM profile_fingerprints
               WHERE user_id = ? AND source = 'enrollment'
               ORDER BY created_at ASC""",
            (session["user_id"],),
        ).fetchall()

        if latest_fp_row and enroll_fp_rows:
            login_vec = json.loads(latest_fp_row["fingerprint_json"])
            for idx, er in enumerate(enroll_fp_rows):
                enroll_vec = json.loads(er["fingerprint_json"])
                score, overlap = compare_fingerprints(login_vec, enroll_vec)
                sim = score if score is not None else 0.0
                cosine_sims.append({
                    "prompt_index": idx,
                    "prompt_text":  f"Enrollment Sample {idx+1}",
                    "sim":          round(sim, 4),
                })
            # Print to terminal
            print(f"[Dashboard] Fingerprint similarities for {session['username']}:")
            for cs in cosine_sims:
                print(f"  {cs['prompt_text']}: {cs['sim']:.4f}")

    # -- Stage 3: verification + profile data ----------------------------------
    verify_result   = None   # {score, decision, threshold} for latest login
    profile_count   = 0
    profile_log     = []     # recent profile_fingerprints entries
    stage3_ready    = VERIFY_ENABLED

    if VERIFY_ENABLED:
        # Latest login's behavioral result
        beh_row = db.execute(
            """SELECT behavioral_score, behavioral_decision
               FROM raw_sequences
               WHERE user_id = ? AND source = 'login'
                 AND behavioral_score IS NOT NULL
               ORDER BY created_at DESC LIMIT 1""",
            (session["user_id"],),
        ).fetchone()
        if beh_row:
            verify_result = {
                "score":     round(beh_row["behavioral_score"], 4),
                "decision":  bool(beh_row["behavioral_decision"]),
                "threshold": VERIFY_THRESHOLD,
            }

        profile_count = db.execute(
            "SELECT COUNT(*) FROM profile_fingerprints WHERE user_id = ?",
            (session["user_id"],)
        ).fetchone()[0]

        profile_log = db.execute(
            """SELECT source, created_at FROM profile_fingerprints
               WHERE user_id = ? ORDER BY created_at DESC LIMIT 5""",
            (session["user_id"],),
        ).fetchall()
        profile_log = [{"source": r["source"], "created_at": r["created_at"]} for r in profile_log]

    return render_template(
        "dashboard.html",
        username=session["username"],
        rows=rows,
        sample_stats=sample_stats,
        enrollment_count=enrollment_count,
        login_count=login_count,
        stage2_ready=stage2_ready,
        latest_embedding=latest_fingerprint,
        cosine_sims=cosine_sims,
        stage3_ready=stage3_ready,
        stage4_ready=stage3_ready,
        verify_result=verify_result,
        profile_count=profile_count,
        max_profile_size=MAX_PROFILE_SIZE,
        profile_log=profile_log,
        acs_state=session.get("acs_state"),
    )


# ── Logout ────────────────────────────────────────────────────────────────

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("login"))


# -- About -----------------------------------------------------------------------

@app.route("/about")
def about():
    return render_template("about.html")


# -- Session Demo (D1) -----------------------------------------------------------

def _get_synthetic_impostor_fingerprint():
    """
    Generate a random 530-dimensional synthetic fingerprint to act as an impostor.
    Simulates someone with a totally different typing cadence.
    """
    import numpy as np
    from fingerprint import SENTINEL, get_total_dims

    total_dims = get_total_dims()
    fp = np.full(total_dims, SENTINEL, dtype=np.float64)
    # Populate ~30% of dimensions with random realistic values
    mask = np.random.rand(total_dims) > 0.7
    fp[mask] = np.random.normal(loc=0.15, scale=0.08, size=np.sum(mask))
    fp = np.clip(fp, 0.0, 2.0)
    return fp.tolist()


@app.route("/session_demo")
def session_demo():
    if "user_id" not in session:
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    # D2 guard: must have enrollment data
    db = get_db()
    enroll_count = db.execute(
        "SELECT COUNT(*) FROM raw_sequences WHERE user_id=? AND source='enrollment'",
        (session["user_id"],),
    ).fetchone()[0]
    if enroll_count < 3:
        flash("Complete enrollment (register) before using the session demo.", "error")
        return redirect(url_for("dashboard"))

    # Reset ACS state for this demo session
    session["acs_state"] = {
        "acs": 1.0,
        "history": [],
        "consecutive_below": 0,
        "locked": False,
        "lockout_prompt": None,
    }
    session.modified = True

    return render_template(
        "session_demo.html",
        prompts=DEMO_PROMPTS,
        genuine_count=GENUINE_PROMPTS,
        threshold=VERIFY_THRESHOLD,
        verify_enabled=VERIFY_ENABLED,
    )


@app.route("/session_demo/submit", methods=["POST"])
def session_demo_submit():
    """AJAX endpoint: receives typed events for one prompt, returns ACS update."""
    if "user_id" not in session:
        return jsonify({"error": "not logged in"}), 401

    data           = request.get_json(force=True) or {}
    prompt_num     = int(data.get("prompt_num", 0))    # 0-indexed
    raw_events     = data.get("events", [])
    impostor_mode  = data.get("impostor_mode", False)  # True for prompts >= GENUINE_PROMPTS

    # D2 guard: empty events
    if not impostor_mode and len(raw_events) < 15:
        return jsonify({"error": "no_events", "message": "Keystroke sequence too short. Please type manually."}), 400

    score = VERIFY_THRESHOLD   # safe default

    if VERIFY_ENABLED:
        try:
            if impostor_mode:
                # Inject synthetic impostor fingerprint instead of computing from typed events
                imp_fp = _get_synthetic_impostor_fingerprint()
                _, score, _ = verify_login(session["user_id"], imp_fp, prompt_index=prompt_num % len(LOGIN_PROMPTS))
            else:
                login_fp = events_to_fingerprint(json.dumps(raw_events))
                _, score, _ = verify_login(session["user_id"], login_fp, prompt_index=prompt_num % len(LOGIN_PROMPTS))
        except Exception as exc:
            app.logger.warning(f"[SessionDemo] verification failed: {exc}")
            score = 0.0

    # EMA update
    state = session.get("acs_state", {
        "acs": 1.0, "history": [], "consecutive_below": 0,
        "locked": False, "lockout_prompt": None,
    })

    acs = ACS_ALPHA * score + (1 - ACS_ALPHA) * state["acs"]
    below = acs < VERIFY_THRESHOLD
    consecutive_below = (state["consecutive_below"] + 1) if below else 0
    just_locked = (not state["locked"]) and consecutive_below >= 2
    locked = state["locked"] or just_locked

    state["acs"]               = acs
    state["consecutive_below"] = consecutive_below
    state["locked"]            = locked
    if just_locked:
        state["lockout_prompt"] = prompt_num
    state["history"].append({
        "prompt_num": prompt_num,
        "score":      round(score, 4),
        "acs":        round(acs, 4),
        "impostor":   impostor_mode,
    })
    session["acs_state"] = state
    session.modified = True

    prompts_after_injection = None
    if just_locked and state["lockout_prompt"] is not None:
        prompts_after_injection = state["lockout_prompt"] - GENUINE_PROMPTS + 1

    if just_locked:
        app.logger.info(
            f"[SessionDemo] LOCKOUT for user '{session.get('username')}' "
            f"at prompt {prompt_num} — {prompts_after_injection} prompt(s) after impostor injection"
        )

    return jsonify({
        "score":                   round(score, 4),
        "acs":                     round(acs, 4),
        "locked":                  locked,
        "just_locked":             just_locked,
        "threshold":               VERIFY_THRESHOLD,
        "prompt_num":              prompt_num,
        "prompts_after_injection": prompts_after_injection,
        "history":                 state["history"],
    })


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # use_reloader=False: TensorFlow modifies site-package timestamps during import,
    # causing Flask's watchdog to infinitely reload when debug=True. Disabling the
    # reloader prevents this -- manually restart the server after code changes.
    app.run(debug=True, port=5000, use_reloader=False)
