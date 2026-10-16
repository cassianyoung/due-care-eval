#!/usr/bin/env python3
"""schema.py : creates the results database's tables and views on a connection.

`BASE_SCHEMA` holds the excuse test's raw readings (`artefacts`, `banks`, `cells`,
`orderings`); `duty_db.SCHEMA` holds the duty test's. Every derived quantity is a view,
defined once in `results_derive.VIEWS` and `duty_db.VIEWS`.

    import sqlite3
    from db import schema
    con = sqlite3.connect("results.sqlite")
    schema.create(con)
"""
import sqlite3

try:
    from . import duty_db, results_derive
except ImportError:                      # run as a plain script from inside db/
    import duty_db
    import results_derive

BASE_SCHEMA = """
CREATE TABLE artefacts(
  id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, sha256 TEXT NOT NULL,
  run TEXT NOT NULL, card TEXT, arm TEXT, delivery TEXT, readout TEXT,
  model_id TEXT, revision TEXT, adapter_digest TEXT, env TEXT,
  banked_at TEXT, bank_name TEXT, bank_sha256 TEXT,
  adapter_family TEXT NOT NULL, superseded_by TEXT, enters_tables INTEGER NOT NULL,
  -- THE LOAD PROVENANCE: two artefacts read from one memoised model load record the
  -- same load_seconds and name each other in load_shared_with, so "these two frames
  -- came from one load" is assertable from SQL.
  load_seconds REAL, load_shared_with TEXT,
  -- `run` is the session label a result is filed under; `log_run` is the label of
  -- the call log that produced it. They usually agree; where they differ, anything
  -- that reads the call log keys on `log_run`.
  log_run TEXT);
CREATE TABLE banks(
  id INTEGER PRIMARY KEY, bank_name TEXT NOT NULL, bank_sha256 TEXT NOT NULL,
  row_uid TEXT NOT NULL, family TEXT, role TEXT, text TEXT NOT NULL,
  text_sha256 TEXT NOT NULL, UNIQUE(bank_name, row_uid));
CREATE TABLE cells(
  id INTEGER PRIMARY KEY, artefact_id INTEGER NOT NULL REFERENCES artefacts(id),
  family TEXT, role TEXT NOT NULL, question TEXT NOT NULL, frame TEXT,
  ev REAL, ev_sd REAL, answer_mass REAL, mass_0_3 REAL, p_yes REAL,
  probs TEXT, n_draws INTEGER, n_reasoning INTEGER, n_void INTEGER,
  reasoning_share REAL, thin INTEGER, void INTEGER,
  -- ALL THREE READINGS ARE CARRIED, NOT ONLY THE PROMOTED ONE, so a reader sees what
  -- was excluded instead of inferring it from a NULL: `ev` is the promoted reading,
  -- and the reasoning-only and all-draws readings sit beside it.
  ev_reasoning_only REAL, ev_all_draws REAL,
  -- The CONFIGURATION, resolved from the reasoning setting sent on the wire, NEVER
  -- from the observed share: the share says what happened, not what was asked for,
  -- and keying on it would let an off arm that reasoned on some draws switch reading
  -- cell by cell.
  reasoning_configured TEXT,          -- on | off | logprob | unknown
  ev_source TEXT,                     -- which reading `ev` above came from
  reasoned_despite_off INTEGER,       -- flags the cell; never changes its reading
  -- WHERE mass_0_3 CAME FROM. A sum over a log-prob distribution and a share of sampled
  -- draws are different measurements of the same quantity, and a column that holds both
  -- without saying which invites them to be averaged.
  mass_0_3_source TEXT,
  reasoning_only_source TEXT,         -- how ev_reasoning_only / n_reasoning were got
  answer_mass_source TEXT,            -- banked, or derived as 1 - void_rate
  row_text_sha256 TEXT, prompt_sha256_16 TEXT,
  -- ── A BANKED BAND, FOR ARMS WITH NO PER-ORDERING ROWS ─────────────────────────
  -- The `band` view computes 2.33 x AVG(sd) from `orderings`. A reading that banks
  -- per-arm aggregates over its permutations rather than the permutations themselves
  -- has no `orderings` rows, so its own band is carried here WITH ITS SOURCE: the
  -- quantity exists and is visibly not the view's construction.
  banked_band REAL, banked_band_source TEXT);
CREATE TABLE orderings(
  id INTEGER PRIMARY KEY, artefact_id INTEGER NOT NULL REFERENCES artefacts(id),
  family TEXT, role TEXT NOT NULL, question TEXT NOT NULL, frame TEXT NOT NULL,
  ev REAL, answer_mass REAL, void INTEGER);
"""


def all_views():
    """Every exported view as {name: CREATE VIEW sql}, excuse test first."""
    v = dict(results_derive.VIEWS)
    v.update(duty_db.VIEWS)
    return v


def create(con):
    """Create the base tables, the duty tables and every view on a fresh connection.

    Views are created in dependency-safe order: each pass creates every view whose
    dependencies already exist, and the loop refuses if a pass makes no progress."""
    con.executescript(BASE_SCHEMA)
    con.executescript(duty_db.SCHEMA)
    pending = all_views()
    while pending:
        made, last_err = [], None
        for name, sql in pending.items():
            try:
                con.execute(sql)
                made.append(name)
            except sqlite3.OperationalError as exc:
                if "no such table" not in str(exc):
                    raise
                last_err = exc
        if not made:
            raise RuntimeError(f"views cannot be created: {sorted(pending)}; "
                               f"last error {last_err}")
        for name in made:
            del pending[name]
    con.commit()
    return con
