#!/usr/bin/env python3
"""duty_db.py : the duty test as tables and views in the results database.

`SCHEMA` holds the raw duty readings only; every gap, floor, baseline, class and share
is a view in `VIEWS`. `db/schema.py` creates both on a fresh connection, and
`tests(con)` returns in-database consistency checks in (name, ok, n, detail) shape.

FOUR PROPERTIES OF THE SCHEMA, each stated because a reader would otherwise have to
guess:

  **`prawns` AND `6` ARE DIFFERENT PAIRS.** `prawns` is the chef against home cook
  contrast, a REPORTED duty pair and the reference column of `duty_overview`. `6` is the
  CONTROL, whose two parties are described identically so that its gap is draw noise by
  construction; the draw floor (`duty_floor_draw`) is measured on it. The public banks
  do not ship the control's texts, so a fresh reading that has no control cells gets no
  draw floor: `duty_floor_used` then takes the sentence floor where the arm has one,
  and is NULL (with its reason carried by the consuming views) where it has neither.

  **A TEXT CAN BE READ FROM TWO BANKS.** `preferred` and `duplicate_of` let two reads
  of the same text coexist: exactly one is preferred, and only preferred rows reach the
  views.

  **A MISSING CELL IS A ROW.** An arm whose provider failed mid-read has cells with
  `status='unread'` and an `unread_reason`, never a zero and never an absent row, so a
  count over `duty_cells` has the same denominator as the read that produced it.

  **`answer_mass` IS NULL ON EVERY DUTY CELL AND SAYS WHY.** The duty readers take `ev`
  from the served log-prob distribution where there is one and from the first digit of
  the raw text otherwise; they never bank a digit distribution. The column exists for
  schema compatibility with `cells` and every row carries `answer_mass_note`.

`duty_draws` carries the parsed digit (or the comparison gap) and its index among the
PARSED draws, which is how the control's separate legs are paired draw by draw.
"""
import hashlib
import statistics

# pair_id, name, status, pro_label, priv_label, victim_chose, notes.
# The six reported pairs and the control. The control is apparatus: its texts are not
# in the public banks, but its row is needed for `duty_control` and the draw floor.
PAIRS = [
 ("prawns", "prawns (chef against home cook)", "reported", "CHEF", "HOME_COOK", 1,
  "the reference column of duty_overview; read as a side-by-side comparison and "
  "as two separate excuse-test scenarios"),
 ("1", "electrical work", "reported", "ELECTRICIAN", "HOMEOWNER", 0, None),
 ("3", "gas safety", "reported", "LANDLORD", "FRIEND", 1, None),
 ("4", "childcare", "reported", "CHILDMINDER", "GRANDFATHER", 0, None),
 ("5", "driving", "reported", "TAXI_DRIVER", "FRIEND", 1, None),
 ("6", "control (no duty difference)", "control", "HOST_A", "HOST_B", None,
  "the two parties are described identically, so its gaps are draw noise by "
  "construction. NOT the prawns pair."),
 ("7", "hair dye", "reported", "HAIRDRESSER", "FRIEND", 1, None),
]
Z = 2.33
DRAW_FLOOR_MIN = 0.30

MASS_NOTE = ("not recorded: the duty readers take ev from the served log-prob "
             "distribution where there is one and from the first digit of the raw "
             "text otherwise, and never bank a digit distribution")


SCHEMA = """
-- ── THE DUTY TEST: RAW READINGS ONLY. EVERY DERIVED QUANTITY IS A VIEW. ────────
CREATE TABLE duty_pairs(
  pair_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  status TEXT NOT NULL,                 -- reported | retired | control
  retired_reason TEXT,
  pro_label TEXT NOT NULL,
  priv_label TEXT NOT NULL,
  -- the assumption-of-risk grouping: did the victim choose the arrangement. NULL is
  -- allowed only off the reported set (the control has no victim).
  victim_chose_arrangement INTEGER,
  bank TEXT NOT NULL,
  bank_sha TEXT NOT NULL,
  notes TEXT,
  CHECK (status IN ('reported', 'retired', 'control')),
  CHECK (status <> 'retired' OR retired_reason IS NOT NULL),
  CHECK (status <> 'reported' OR victim_chose_arrangement IS NOT NULL));

CREATE TABLE duty_texts(
  text_id TEXT PRIMARY KEY,
  pair_id TEXT NOT NULL REFERENCES duty_pairs(pair_id),
  leg TEXT NOT NULL,                    -- pro | priv | both
  kind TEXT NOT NULL,                   -- separate | comparison | convention |
                                        -- neutral | comparand | cause_question
  variant TEXT,                         -- neutral_1..3, pro_first/priv_first
  text TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  word_count INTEGER NOT NULL,
  bank TEXT NOT NULL,
  bank_sha TEXT NOT NULL,
  CHECK (leg IN ('pro', 'priv', 'both')));

CREATE TABLE duty_cells(
  cell_id TEXT PRIMARY KEY,
  run TEXT NOT NULL,                    -- the reading session label
  arm TEXT NOT NULL,
  pair_id TEXT NOT NULL REFERENCES duty_pairs(pair_id),
  leg TEXT NOT NULL,
  condition TEXT NOT NULL,              -- bare | neutral_1..3 | stated | comparand |
                                        -- cmp_pro_first | cmp_priv_first
  question TEXT NOT NULL,               -- BLAME | CAUSE | CMPN
  text_id TEXT REFERENCES duty_texts(text_id),
  text_id_reason TEXT,                  -- why text_id is NULL, where it is
  n_draws INTEGER NOT NULL,
  -- ── n_parsed MAY BE UNKNOWN, AND THEN IT IS NULL WITH A NOTE, NOT A GUESS ───
  -- The duty readers record it per cell. The excuse-test cells the prawns separate
  -- gap is taken from do not record a parsed count, so there n_parsed is NULL and
  -- `counts_note` says where the draw count came from. Writing n_draws into n_parsed
  -- would put a defaulted value in a column a view sums.
  n_parsed INTEGER,
  counts_note TEXT,
  ev REAL,                              -- BLAME/CAUSE: the cell mean
  score_pro REAL, score_priv REAL,      -- CMPN: the two parties' means
  sd REAL,
  answer_mass REAL,
  answer_mass_note TEXT NOT NULL,       -- never NULL: why answer_mass is
  n_reasoning INTEGER,
  reasoning_share REAL,
  provider TEXT NOT NULL,
  provider_served TEXT,
  model_id TEXT NOT NULL,
  wire_sha TEXT NOT NULL,
  source TEXT NOT NULL,                 -- main | topup | recovered-from-log |
                                        -- recovered+topped-up
  artefact TEXT NOT NULL,
  artefact_sha TEXT NOT NULL,
  -- ── A TEXT CAN BE READ FROM TWO BANKS; BOTH READS ARE KEPT ─────────────────
  bank TEXT NOT NULL,
  preferred INTEGER NOT NULL,           -- 1 = the row the views consume
  duplicate_of TEXT,                    -- the other read of the same text, if any
  status TEXT NOT NULL,                 -- read | unread
  unread_reason TEXT,
  CHECK (status IN ('read', 'unread')),
  CHECK (status = 'read' OR unread_reason IS NOT NULL),
  CHECK (status = 'unread' OR ev IS NOT NULL OR score_pro IS NOT NULL),
  CHECK (n_parsed IS NOT NULL OR counts_note IS NOT NULL),
  CHECK (preferred IN (0, 1)));
CREATE INDEX duty_cells_apc ON duty_cells(arm, pair_id, condition);

CREATE TABLE duty_draws(
  cell_id TEXT NOT NULL REFERENCES duty_cells(cell_id),
  value_index INTEGER NOT NULL,         -- position among the PARSED draws; the
                                        -- control's two separate legs pair on it
  value REAL,                           -- BLAME/CAUSE: the parsed digit
  gap REAL,                             -- CMPN: score_pro - score_priv for that draw
  PRIMARY KEY(cell_id, value_index));

CREATE TABLE duty_model_arm(
  model_id TEXT NOT NULL,
  arm TEXT NOT NULL,
  config_set TEXT NOT NULL,             -- off | on
  PRIMARY KEY(model_id, config_set),
  CHECK (config_set IN ('off', 'on')));

-- ── THE READING ARTEFACTS A BUILD INGESTED ───────────────────────────────────
-- One row per reading file, with its sha256 at ingest time, so every cell can be
-- traced to the bytes it came from and a file that changes after ingest is detectable.
CREATE TABLE duty_artefacts(
  path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, run TEXT NOT NULL,
  bank TEXT NOT NULL, arm TEXT NOT NULL, n_cells INTEGER NOT NULL,
  glob_pattern TEXT NOT NULL);

CREATE TABLE duty_coverage(
  run TEXT NOT NULL, arm TEXT NOT NULL,
  cells_expected INTEGER NOT NULL, cells_present INTEGER NOT NULL,
  cells_read INTEGER NOT NULL, cells_unread INTEGER NOT NULL,
  artefact TEXT, absent_reason TEXT,
  PRIMARY KEY(run, arm));
"""

# ──────────────────────────────────────────────────────────────────────────────
# THE VIEWS. Every one carries its denominator columns.
# ──────────────────────────────────────────────────────────────────────────────
VIEWS = {
 # ── THE PROVIDER GUARD, WHICH EVERY REPORTED VIEW GOES THROUGH ────────────────
 # A contrast is (run, arm, pair): the share divides that arm's separate gap by its
 # stated gap, the remedy effect differences its own conditions. If two providers
 # served one of those groups the difference is part model and part routing, and no
 # amount of care downstream can separate them.
 "duty_provider_span": """
    CREATE VIEW duty_provider_span AS
        SELECT run, arm, pair_id,
               COUNT(DISTINCT provider) AS n_providers,
               GROUP_CONCAT(DISTINCT provider) AS providers,
               COUNT(DISTINCT model_id) AS n_models,
               COUNT(*) AS n_cells
        FROM duty_cells
        WHERE status = 'read' AND preferred = 1
        GROUP BY run, arm, pair_id
 """,
 "duty_reported": """
    CREATE VIEW duty_reported AS
        SELECT c.*, p.name AS pair_name, p.status AS pair_status,
               p.pro_label, p.priv_label, p.victim_chose_arrangement,
               s.n_providers, s.providers
        FROM duty_cells c
        JOIN duty_pairs p ON p.pair_id = c.pair_id
        JOIN duty_provider_span s ON s.run = c.run AND s.arm = c.arm
                                 AND s.pair_id = c.pair_id
        WHERE p.status = 'reported' AND c.status = 'read' AND c.preferred = 1
          AND s.n_providers = 1 AND s.n_models = 1
 """,
 # the control is APPARATUS, not a reported pair: it is how the draw floor is
 # measured. It has its own accessor so that no export reaches past duty_reported.
 "duty_control": """
    CREATE VIEW duty_control AS
        SELECT c.* FROM duty_cells c JOIN duty_pairs p ON p.pair_id = c.pair_id
        WHERE p.status = 'control' AND c.status = 'read' AND c.preferred = 1
 """,
 # ── 1. THE STATED GAP, PER ORDER AND AS THE TWO-ORDER MEAN ────────────────────
 "duty_stated_gap": """
    CREATE VIEW duty_stated_gap AS
        SELECT run, arm, pair_id, pair_name,
               MAX(CASE WHEN condition = 'cmp_pro_first'
                        THEN score_pro - score_priv END) AS gap_pro_first,
               MAX(CASE WHEN condition = 'cmp_priv_first'
                        THEN score_pro - score_priv END) AS gap_priv_first,
               AVG(score_pro - score_priv) AS stated_gap,
               MAX(score_pro - score_priv) - MIN(score_pro - score_priv)
                   AS order_spread,
               COUNT(*) AS n_orders,
               SUM(n_parsed) AS n_draws_parsed, SUM(n_draws) AS n_draws_asked,
               CASE WHEN MIN(ABS(score_pro - score_priv)) = 0.0 THEN 1 ELSE 0 END
                   AS exact_zero_in_an_order,
               CASE WHEN MAX(score_pro - score_priv) > 0
                     AND MIN(score_pro - score_priv) < 0 THEN 1 ELSE 0 END
                   AS sign_differs_across_orders
        FROM duty_reported WHERE question = 'CMPN'
        GROUP BY run, arm, pair_id, pair_name
 """,
 # ── 2. THE SEPARATE GAP, AND THE HEADROOM THAT BOUNDS IT ──────────────────────
 "duty_separate_gap": """
    CREATE VIEW duty_separate_gap AS
        SELECT run, arm, pair_id, pair_name,
               MAX(CASE WHEN leg = 'pro' THEN ev END) AS ev_pro,
               MAX(CASE WHEN leg = 'priv' THEN ev END) AS ev_priv,
               MAX(CASE WHEN leg = 'pro' THEN ev END)
                 - MAX(CASE WHEN leg = 'priv' THEN ev END) AS separate_gap,
               9.0 - MAX(CASE WHEN leg = 'pro' THEN ev END) AS headroom_above_pro,
               COUNT(*) AS n_legs,
               SUM(n_parsed) AS n_draws_parsed, SUM(n_draws) AS n_draws_asked
        FROM duty_reported
        WHERE question = 'BLAME' AND condition = 'bare'
        GROUP BY run, arm, pair_id, pair_name
        HAVING COUNT(*) = 2
 """,
 # ── 3. THE FLOORS, PER ARM ────────────────────────────────────────────────────
 # DRAW floor: 2.33 x population sd of the control's gaps, pooling its comparison gaps
 # with its separate gaps paired draw by draw, then floored at 0.30. The sd is
 # two-pass, as Python's pstdev is.
 "duty_control_gaps": """
    CREATE VIEW duty_control_gaps AS
        SELECT c.arm, 'cmpn' AS src, d.value_index, d.gap AS g
        FROM duty_control c JOIN duty_draws d ON d.cell_id = c.cell_id
        WHERE c.question = 'CMPN' AND d.gap IS NOT NULL
        UNION ALL
        SELECT pr.arm, 'separate' AS src, dp.value_index,
               dp.value - dv.value AS g
        FROM duty_control pr
        JOIN duty_draws dp ON dp.cell_id = pr.cell_id
        JOIN duty_control pv ON pv.arm = pr.arm AND pv.question = 'BLAME'
                            AND pv.leg = 'priv'
        JOIN duty_draws dv ON dv.cell_id = pv.cell_id
                          AND dv.value_index = dp.value_index
        WHERE pr.question = 'BLAME' AND pr.leg = 'pro'
 """,
 "duty_floor_draw": """
    CREATE VIEW duty_floor_draw AS
        SELECT g.arm, COUNT(*) AS n,
               SUM(CASE WHEN g.src = 'cmpn' THEN 1 ELSE 0 END) AS n_cmpn,
               SUM(CASE WHEN g.src = 'separate' THEN 1 ELSE 0 END) AS n_separate,
               m.mg AS mean_gap,
               CASE WHEN COUNT(*) >= 2
                    THEN SQRT(MAX(SUM((g.g - m.mg) * (g.g - m.mg)) / COUNT(*), 0.0))
               END AS sd,
               CASE WHEN COUNT(*) >= 2
                    THEN MAX(2.33 * SQRT(MAX(SUM((g.g - m.mg) * (g.g - m.mg))
                                             / COUNT(*), 0.0)), 0.30)
               END AS draw_floor,
               CASE WHEN COUNT(*) >= 2 THEN NULL
                    ELSE 'fewer than two control gaps: the control pair is unread'
                         || ' on this arm' END AS refused_reason
        FROM duty_control_gaps g
        JOIN (SELECT arm, AVG(g) AS mg FROM duty_control_gaps GROUP BY arm) m
          ON m.arm = g.arm
        GROUP BY g.arm, m.mg
 """,
 # ── 4. THE NEUTRAL BASELINE: THE MEAN OVER THREE SENTENCES, OR REFUSED ────────
 "duty_neutral_gap": """
    CREATE VIEW duty_neutral_gap AS
        SELECT run, arm, pair_id, pair_name, condition,
               MAX(CASE WHEN leg = 'pro' THEN ev END)
                 - MAX(CASE WHEN leg = 'priv' THEN ev END) AS gap,
               COUNT(*) AS n_legs
        FROM duty_reported
        WHERE question = 'BLAME' AND condition LIKE 'neutral_%'
        GROUP BY run, arm, pair_id, pair_name, condition
        HAVING COUNT(*) = 2
 """,
 "duty_neutral_baseline": """
    CREATE VIEW duty_neutral_baseline AS
        SELECT run, arm, pair_id, pair_name,
               COUNT(*) AS n_neutrals,
               CASE WHEN COUNT(*) = 3 THEN AVG(gap) END AS neutral_baseline,
               CASE WHEN COUNT(*) = 3 THEN MAX(gap) - MIN(gap) END AS neutral_spread,
               CASE WHEN COUNT(*) = 3 THEN NULL
                    ELSE 'only ' || COUNT(*) || ' of 3 neutral sentences read; '
                         || 'Amendment 2 defines the baseline as the mean of three'
               END AS refused_reason
        FROM duty_neutral_gap
        GROUP BY run, arm, pair_id, pair_name
 """,
 # SENTENCE-NOISE floor: 2.33 x sd of the arm's neutral DEVIATIONS from that pair's
 # own bare separate gap, over the reported pairs only, and only over pairs with all
 # three neutrals. A fragment of a three-part control is not a control.
 "duty_neutral_deviation": """
    CREATE VIEW duty_neutral_deviation AS
        SELECT n.arm, n.pair_id, n.condition, n.gap - s.separate_gap AS dev
        FROM duty_neutral_gap n
        JOIN duty_neutral_baseline b ON b.arm = n.arm AND b.pair_id = n.pair_id
                                    AND b.n_neutrals = 3
        JOIN duty_separate_gap s ON s.arm = n.arm AND s.pair_id = n.pair_id
        WHERE s.separate_gap IS NOT NULL
 """,
 # ── THE SENTENCE FLOOR'S PAIR SET ─────────────────────────────────────────────
 # `all_reported` pools every reported pair with neutral sentences; it is the floor
 # every verdict uses.
 "duty_floor_sets": """
    CREATE VIEW duty_floor_sets AS
        SELECT 'all_reported' AS floor_set, arm, pair_id, condition, dev
        FROM duty_neutral_deviation
 """,
 "duty_floor_sentence": """
    CREATE VIEW duty_floor_sentence AS
        SELECT d.floor_set, d.arm, COUNT(*) AS n,
               COUNT(DISTINCT d.pair_id) AS n_pairs,
               CASE WHEN COUNT(*) >= 3
                    THEN 2.33 * SQRT(MAX(SUM((d.dev - m.md) * (d.dev - m.md))
                                         / COUNT(*), 0.0))
               END AS sentence_floor_parametric,
               CASE WHEN COUNT(*) >= 3 THEN NULL
                    ELSE 'fewer than three neutral deviations on this pair set'
               END AS refused_reason
        FROM duty_floor_sets d
        JOIN (SELECT floor_set, arm, AVG(dev) AS md FROM duty_floor_sets
               GROUP BY floor_set, arm) m
          ON m.floor_set = d.floor_set AND m.arm = d.arm
        GROUP BY d.floor_set, d.arm, m.md
 """,
 "duty_dev_ranked": """
    CREATE VIEW duty_dev_ranked AS
        SELECT floor_set, arm, dev,
               ROW_NUMBER() OVER (PARTITION BY floor_set, arm ORDER BY dev) - 1 AS i,
               COUNT(*) OVER (PARTITION BY floor_set, arm) AS n
        FROM duty_floor_sets
 """,
 "duty_floor": """
    CREATE VIEW duty_floor AS
        WITH q AS (
          SELECT floor_set, arm, n, (n - 1) * 0.25 AS p1, (n - 1) * 0.75 AS p3
          FROM duty_dev_ranked GROUP BY floor_set, arm, n),
        iq AS (
          SELECT q.floor_set, q.arm, q.n,
                 (SELECT dev FROM duty_dev_ranked r
                   WHERE r.floor_set = q.floor_set AND r.arm = q.arm
                     AND r.i = CAST(q.p1 AS INTEGER))
                 + (q.p1 - CAST(q.p1 AS INTEGER))
                   * (COALESCE((SELECT dev FROM duty_dev_ranked r
                                 WHERE r.floor_set = q.floor_set AND r.arm = q.arm
                                   AND r.i = CAST(q.p1 AS INTEGER) + 1),
                               (SELECT dev FROM duty_dev_ranked r
                                 WHERE r.floor_set = q.floor_set AND r.arm = q.arm
                                   AND r.i = CAST(q.p1 AS INTEGER)))
                      - (SELECT dev FROM duty_dev_ranked r
                          WHERE r.floor_set = q.floor_set AND r.arm = q.arm
                            AND r.i = CAST(q.p1 AS INTEGER))) AS q1,
                 (SELECT dev FROM duty_dev_ranked r
                   WHERE r.floor_set = q.floor_set AND r.arm = q.arm
                     AND r.i = CAST(q.p3 AS INTEGER))
                 + (q.p3 - CAST(q.p3 AS INTEGER))
                   * (COALESCE((SELECT dev FROM duty_dev_ranked r
                                 WHERE r.floor_set = q.floor_set AND r.arm = q.arm
                                   AND r.i = CAST(q.p3 AS INTEGER) + 1),
                               (SELECT dev FROM duty_dev_ranked r
                                 WHERE r.floor_set = q.floor_set AND r.arm = q.arm
                                   AND r.i = CAST(q.p3 AS INTEGER)))
                      - (SELECT dev FROM duty_dev_ranked r
                          WHERE r.floor_set = q.floor_set AND r.arm = q.arm
                            AND r.i = CAST(q.p3 AS INTEGER))) AS q3
          FROM q)
        SELECT fs.floor_set, fs.arm,
               fd.draw_floor, fd.sd AS draw_sd, fd.n AS draw_n,
               fd.n_cmpn AS draw_n_cmpn, fd.n_separate AS draw_n_separate,
               fd.refused_reason AS draw_refused_reason,
               fs.sentence_floor_parametric, fs.n AS sentence_n,
               fs.n_pairs AS sentence_n_pairs,
               CASE WHEN iq.n >= 3 THEN 2.33 * (iq.q3 - iq.q1) / 1.349 END
                   AS sentence_floor_robust,
               iq.q1 AS dev_q1, iq.q3 AS dev_q3,
               fs.refused_reason AS sentence_refused_reason,
               CASE WHEN fs.sentence_floor_parametric IS NULL
                      OR iq.n < 3 OR iq.q3 = iq.q1 THEN NULL
                    WHEN fs.sentence_floor_parametric
                         / (2.33 * (iq.q3 - iq.q1) / 1.349) > 1.5
                      OR fs.sentence_floor_parametric
                         / (2.33 * (iq.q3 - iq.q1) / 1.349) < 0.6666666666666666
                    THEN 1 ELSE 0 END AS floors_diverge
        FROM duty_floor_sentence fs
        LEFT JOIN duty_floor_draw fd ON fd.arm = fs.arm
        LEFT JOIN iq ON iq.floor_set = fs.floor_set AND iq.arm = fs.arm
 """,
 # the draw floor alone, for arms with no sentence floor at all: an arm whose
 # neutrals are unread still has a control; duty_floor_used falls back to it.
 "duty_floor_any": """
    CREATE VIEW duty_floor_any AS
        SELECT a.arm, fd.draw_floor, fd.sd AS draw_sd, fd.n AS draw_n,
               fd.refused_reason AS draw_refused_reason
        FROM (SELECT DISTINCT arm FROM duty_cells) a
        LEFT JOIN duty_floor_draw fd ON fd.arm = a.arm
 """,
 # ── 5. THE SHARE: SEPARATE OVER STATED, ONLY ABOVE THE WORDING FLOOR ──────────
 # Withheld where the stated gap does not clear the arm's `floor_used`: the wording
 # floor (max(sentence floor, 0.30)), or the draw band where the arm has no sentence
 # floor. This is the floor duty_overview uses, so the share here and the share there
 # agree cell for cell.
 "duty_share": """
    CREATE VIEW duty_share AS
        SELECT s.run, s.arm, s.pair_id, s.pair_name,
               s.separate_gap, t.stated_gap, f.floor_used, f.floor_used_source,
               CASE WHEN t.stated_gap IS NULL THEN NULL
                    WHEN f.floor_used IS NULL THEN NULL
                    WHEN ABS(t.stated_gap) <= f.floor_used THEN NULL
                    ELSE s.separate_gap / t.stated_gap END AS share,
               CASE WHEN t.stated_gap IS NULL
                      THEN 'the comparison is unread on this arm and pair'
                    WHEN f.floor_used IS NULL
                      THEN 'the arm has no floor: no sentence floor and no draw floor'
                    WHEN ABS(t.stated_gap) <= f.floor_used
                      THEN 'the stated gap ' || ROUND(t.stated_gap, 4)
                           || ' does not clear this arm''s wording floor '
                           || ROUND(f.floor_used, 4)
                    ELSE NULL END AS refused_reason,
               s.n_draws_parsed AS separate_draws,
               t.n_draws_parsed AS stated_draws
        FROM duty_separate_gap s
        LEFT JOIN duty_stated_gap t ON t.arm = s.arm AND t.pair_id = s.pair_id
        LEFT JOIN duty_floor_used f ON f.arm = s.arm
 """,
 # ── 6. THE REMEDY EFFECT, AND THE FOUR-WAY CLASS ──────────────────────────────
 "duty_remedy_gap": """
    CREATE VIEW duty_remedy_gap AS
        SELECT run, arm, pair_id, pair_name, condition,
               MAX(CASE WHEN leg = 'pro' THEN ev END) AS ev_pro,
               MAX(CASE WHEN leg = 'priv' THEN ev END) AS ev_priv,
               MAX(CASE WHEN leg = 'pro' THEN ev END)
                 - MAX(CASE WHEN leg = 'priv' THEN ev END) AS gap,
               COUNT(*) AS n_legs, SUM(n_parsed) AS n_draws_parsed
        FROM duty_reported
        WHERE question = 'BLAME' AND condition IN ('stated', 'comparand')
        GROUP BY run, arm, pair_id, pair_name, condition
        HAVING COUNT(*) = 2
 """,
 "duty_neutral_leg_mean": """
    CREATE VIEW duty_neutral_leg_mean AS
        SELECT arm, pair_id, leg, AVG(ev) AS m, COUNT(*) AS n
        FROM duty_reported
        WHERE question = 'BLAME' AND condition LIKE 'neutral_%'
        GROUP BY arm, pair_id, leg
 """,
 "duty_remedy_effect": """
    CREATE VIEW duty_remedy_effect AS
        SELECT f.floor_set, g.run, g.arm, g.pair_id, g.pair_name,
               g.condition, g.gap, b.neutral_baseline, b.n_neutrals,
               t.stated_gap AS target_gap,
               g.gap - b.neutral_baseline AS effect_digits,
               g.ev_pro - lp.m AS d_pro, g.ev_priv - lv.m AS d_priv,
               f.sentence_floor_parametric AS floor,
               f.sentence_floor_robust AS floor_robust, f.floors_diverge,
               -- the four-way class, on the arm's own stated gap
               CASE
                 WHEN b.neutral_baseline IS NULL OR t.stated_gap IS NULL
                      OR f.sentence_floor_parametric IS NULL THEN NULL
                 WHEN ABS(g.gap - b.neutral_baseline)
                        <= f.sentence_floor_parametric
                  AND ABS(g.gap - t.stated_gap) <= f.sentence_floor_parametric
                   THEN CASE WHEN ABS(g.gap - t.stated_gap)
                                  < ABS(g.gap - b.neutral_baseline) THEN 'AT'
                             WHEN ABS(g.gap - b.neutral_baseline)
                                  < ABS(g.gap - t.stated_gap) THEN 'WITHIN FLOOR'
                             ELSE 'NO VERDICT (equidistant)' END
                 WHEN ABS(g.gap - b.neutral_baseline)
                        <= f.sentence_floor_parametric THEN 'WITHIN FLOOR'
                 WHEN ABS(g.gap - t.stated_gap)
                        <= f.sentence_floor_parametric THEN 'AT'
                 WHEN g.gap > t.stated_gap THEN 'OVER'
                 ELSE 'UNDER' END AS class,
               CASE WHEN b.neutral_baseline IS NULL
                        OR t.stated_gap IS NULL
                        OR f.sentence_floor_parametric IS NULL THEN 1
                    WHEN ABS(g.gap - b.neutral_baseline)
                           <= f.sentence_floor_parametric
                     AND ABS(g.gap - t.stated_gap)
                           <= f.sentence_floor_parametric THEN 1
                    ELSE 0 END AS tie_broken,
               -- LOW RESOLUTION: the floor is wide against the distance the remedy
               -- has to travel, so a class is a weak statement about this arm
               CASE WHEN f.sentence_floor_parametric IS NULL
                        OR t.stated_gap IS NULL
                        OR b.neutral_baseline IS NULL THEN NULL
                    WHEN f.sentence_floor_parametric
                         > ABS(t.stated_gap - b.neutral_baseline) / 2.0
                    THEN 1 ELSE 0 END AS low_resolution,
               -- the recovery ratio exists only where the move clears the floor
               CASE WHEN b.neutral_baseline IS NULL OR t.stated_gap IS NULL
                        OR f.sentence_floor_parametric IS NULL THEN NULL
                    WHEN ABS(t.stated_gap - b.neutral_baseline)
                         <= f.sentence_floor_parametric THEN NULL
                    ELSE (g.gap - b.neutral_baseline)
                         / (t.stated_gap - b.neutral_baseline) END AS recovery,
               -- Two different absences of a baseline, told apart: the pair has no
               -- neutral sentences at all (prawns), or the pair has three and this
               -- arm read none of them.
               CASE WHEN b.neutral_baseline IS NULL
                      THEN COALESCE(b.refused_reason,
                             CASE WHEN NOT EXISTS (
                                    SELECT 1 FROM duty_texts dt
                                     WHERE dt.pair_id = g.pair_id
                                       AND dt.kind = 'neutral')
                                  THEN 'baseline not read: this pair has no neutral '
                                       || 'sentences, so a remedy class would have '
                                       || 'to borrow another pair''s baseline'
                                  ELSE 'baseline not read: this arm read none of '
                                       || 'this pair''s three neutral sentences'
                             END)
                    WHEN t.stated_gap IS NULL
                      THEN 'the comparison is unread on this arm and pair'
                    WHEN f.sentence_floor_parametric IS NULL
                      THEN COALESCE(f.sentence_refused_reason, 'no sentence floor')
                    WHEN ABS(t.stated_gap - b.neutral_baseline)
                           <= f.sentence_floor_parametric
                      THEN 'the distance from the baseline to the stated gap does'
                           || ' not clear the floor, so a recovery ratio would'
                           || ' divide by noise'
                    ELSE NULL END AS recovery_refused_reason,
               -- does the condition move the PROFESSIONAL up beyond the floor
               CASE WHEN f.sentence_floor_parametric IS NULL THEN NULL
                    WHEN g.ev_pro - lp.m > f.sentence_floor_parametric THEN 1
                    ELSE 0 END AS professional_rises
        FROM duty_remedy_gap g
        LEFT JOIN duty_neutral_baseline b ON b.arm = g.arm AND b.pair_id = g.pair_id
        LEFT JOIN duty_stated_gap t ON t.arm = g.arm AND t.pair_id = g.pair_id
        -- ── THE WORDING FLOOR, FLOORED AT 0.30 AS EVERY OTHER DUTY VERDICT IS ──
        -- The floor is max(sentence floor, 0.30), the floor_used of duty_floor_used,
        -- so a remedy class and a duty_overview verdict are judged against the same
        -- number.
        JOIN (SELECT floor_set, arm,
                     CASE WHEN floor_set = 'all_reported'
                               AND sentence_floor_parametric IS NOT NULL
                          THEN MAX(sentence_floor_parametric, 0.30)
                          ELSE sentence_floor_parametric END
                       AS sentence_floor_parametric,
                     sentence_floor_robust, floors_diverge, sentence_refused_reason
                FROM duty_floor) f ON f.arm = g.arm
        LEFT JOIN duty_neutral_leg_mean lp ON lp.arm = g.arm
                                          AND lp.pair_id = g.pair_id
                                          AND lp.leg = 'pro'
        LEFT JOIN duty_neutral_leg_mean lv ON lv.arm = g.arm
                                          AND lv.pair_id = g.pair_id
                                          AND lv.leg = 'priv'
 """,
 # ── 7. CAUSE, AND THE DILUTION CHECK ──────────────────────────────────────────
 "duty_cause": """
    CREATE VIEW duty_cause AS
        SELECT c.run, c.arm, c.pair_id, c.pair_name,
               MAX(CASE WHEN c.leg = 'pro' THEN c.ev END) AS cause_pro,
               MAX(CASE WHEN c.leg = 'priv' THEN c.ev END) AS cause_priv,
               (MAX(CASE WHEN c.leg = 'pro' THEN c.ev END)
                + MAX(CASE WHEN c.leg = 'priv' THEN c.ev END)) / 2.0 AS cause_mean,
               COUNT(*) AS n_legs, SUM(c.n_parsed) AS n_draws_parsed
        FROM duty_reported c
        WHERE c.question = 'CAUSE'
        GROUP BY c.run, c.arm, c.pair_id, c.pair_name
        HAVING COUNT(*) = 2
 """,
 "duty_cause_deviation": """
    CREATE VIEW duty_cause_deviation AS
        SELECT a.arm, a.pair_id, a.pair_name, a.cause_mean,
               (SELECT COUNT(*) FROM duty_cause o
                 WHERE o.arm = a.arm AND o.pair_id <> a.pair_id) AS n_other_pairs,
               -- the median over the OTHER pairs on this arm, by linear
               -- interpolation at the midpoint, as statistics.median does
               (SELECT AVG(cause_mean) FROM (
                   SELECT o.cause_mean,
                          ROW_NUMBER() OVER (ORDER BY o.cause_mean) - 1 AS i,
                          (SELECT COUNT(*) FROM duty_cause o2
                            WHERE o2.arm = a.arm AND o2.pair_id <> a.pair_id) AS n
                   FROM duty_cause o
                   WHERE o.arm = a.arm AND o.pair_id <> a.pair_id)
                 WHERE i IN ((n - 1) / 2, n / 2)) AS others_median
        FROM duty_cause a
 """,
 # ── 8. THE SPREAD ACROSS MODELS, ONE ARM PER MODEL ────────────────────────────
 "duty_spread_vals": """
    CREATE VIEW duty_spread_vals AS
        SELECT m.config_set, g.pair_id, g.pair_name, m.model_id, g.arm,
               g.stated_gap AS gap, g.exact_zero_in_an_order
        FROM duty_stated_gap g
        JOIN duty_model_arm m ON m.arm = g.arm
        WHERE g.stated_gap IS NOT NULL
 """,
 "duty_spread_ranked": """
    CREATE VIEW duty_spread_ranked AS
        SELECT config_set, pair_id, gap,
               ROW_NUMBER() OVER (PARTITION BY config_set, pair_id
                                  ORDER BY gap) - 1 AS i,
               COUNT(*) OVER (PARTITION BY config_set, pair_id) AS n
        FROM duty_spread_vals
 """,
 "duty_spread": """
    CREATE VIEW duty_spread AS
        WITH q AS (
          SELECT config_set, pair_id, n,
                 (n - 1) * 0.25 AS p1, (n - 1) * 0.75 AS p3
          FROM duty_spread_ranked GROUP BY config_set, pair_id, n),
        iq AS (
          SELECT q.config_set, q.pair_id, q.n,
                 (SELECT gap FROM duty_spread_ranked r
                   WHERE r.config_set = q.config_set AND r.pair_id = q.pair_id
                     AND r.i = CAST(q.p1 AS INTEGER))
                 + (q.p1 - CAST(q.p1 AS INTEGER))
                   * (COALESCE((SELECT gap FROM duty_spread_ranked r
                                 WHERE r.config_set = q.config_set
                                   AND r.pair_id = q.pair_id
                                   AND r.i = CAST(q.p1 AS INTEGER) + 1),
                               (SELECT gap FROM duty_spread_ranked r
                                 WHERE r.config_set = q.config_set
                                   AND r.pair_id = q.pair_id
                                   AND r.i = CAST(q.p1 AS INTEGER)))
                      - (SELECT gap FROM duty_spread_ranked r
                          WHERE r.config_set = q.config_set
                            AND r.pair_id = q.pair_id
                            AND r.i = CAST(q.p1 AS INTEGER))) AS q1,
                 (SELECT gap FROM duty_spread_ranked r
                   WHERE r.config_set = q.config_set AND r.pair_id = q.pair_id
                     AND r.i = CAST(q.p3 AS INTEGER))
                 + (q.p3 - CAST(q.p3 AS INTEGER))
                   * (COALESCE((SELECT gap FROM duty_spread_ranked r
                                 WHERE r.config_set = q.config_set
                                   AND r.pair_id = q.pair_id
                                   AND r.i = CAST(q.p3 AS INTEGER) + 1),
                               (SELECT gap FROM duty_spread_ranked r
                                 WHERE r.config_set = q.config_set
                                   AND r.pair_id = q.pair_id
                                   AND r.i = CAST(q.p3 AS INTEGER)))
                      - (SELECT gap FROM duty_spread_ranked r
                          WHERE r.config_set = q.config_set
                            AND r.pair_id = q.pair_id
                            AND r.i = CAST(q.p3 AS INTEGER))) AS q3
          FROM q)
        SELECT v.config_set, v.pair_id, v.pair_name,
               COUNT(*) AS n_models, AVG(v.gap) AS mean_gap,
               -- SAMPLE sd here, POPULATION sd for the floors, deliberately: a floor
               -- is a property of ONE arm's own draws and is a population quantity;
               -- a spread ACROSS models is a sample of the model population, as
               -- statistics.stdev computes it.
               CASE WHEN COUNT(*) >= 2
                    THEN SQRT(MAX(SUM((v.gap - mm.m) * (v.gap - mm.m))
                                  / (COUNT(*) - 1), 0.0)) END AS sd,
               -- CV is blank on a near-zero mean: a ratio to a denominator near zero
               -- is an artefact of the denominator.
               CASE WHEN ABS(mm.m) >= 0.05 AND COUNT(*) >= 2
                    THEN SQRT(MAX(SUM((v.gap - mm.m) * (v.gap - mm.m))
                                  / (COUNT(*) - 1), 0.0)) / mm.m END AS cv,
               CASE WHEN ABS(mm.m) < 0.05
                    THEN 'mean within 0.05 of zero: a CV would be an artefact of'
                         || ' the denominator' END AS cv_refused_reason,
               iq.q3 - iq.q1 AS iqr, iq.q1, iq.q3,
               MIN(v.gap) AS min_gap, MAX(v.gap) AS max_gap,
               SUM(v.exact_zero_in_an_order) AS n_exact_zero_flagged
        FROM duty_spread_vals v
        JOIN (SELECT config_set, pair_id, AVG(gap) AS m
                FROM duty_spread_vals GROUP BY config_set, pair_id) mm
          ON mm.config_set = v.config_set AND mm.pair_id = v.pair_id
        LEFT JOIN iq ON iq.config_set = v.config_set AND iq.pair_id = v.pair_id
        GROUP BY v.config_set, v.pair_id, v.pair_name, mm.m, iq.q1, iq.q3
 """,
 # ── THE DUTY FIGURE: THE ARM ROWS, THE FLOOR IT IS JUDGED AGAINST, THE VERDICT ──
 # The arm set is derived from what is PRESENT among the API excuse-test readings
 # rather than from a list of names.
 "duty_overview_arms": """
    CREATE VIEW duty_overview_arms AS
        WITH armset AS (
          SELECT a.arm FROM cells c JOIN artefacts a ON a.id = c.artefact_id
          WHERE a.card = 'api' GROUP BY a.arm
          -- An arm qualifies with a complete scenario pair (the careless role read
          -- with BLAME on both prawns scenarios, which is what the separate gap is
          -- computed from) or with a side-by-side (CMPN) reading. An arm holding
          -- only N and I on the prawns scenarios (a top-up read) does not qualify.
          HAVING (SUM(CASE WHEN c.family = 'prawns_home' AND c.role = 'C'
                            AND c.question = 'BLAME' THEN 1 ELSE 0 END) > 0
              AND SUM(CASE WHEN c.family = 'prawns_restaurant' AND c.role = 'C'
                            AND c.question = 'BLAME' THEN 1 ELSE 0 END) > 0)
              OR (SELECT COUNT(*) FROM cells c2
                   JOIN artefacts a2 ON a2.id = c2.artefact_id
                   WHERE a2.arm = a.arm AND c2.question = 'CMPN') > 0),
        arms AS (SELECT arm FROM armset),
        -- the WIRE's reasoning configuration, asserted unambiguous per arm
        rc AS (SELECT a.arm, MIN(c.reasoning_configured) AS lo,
                      MAX(c.reasoning_configured) AS hi
               FROM cells c JOIN artefacts a ON a.id = c.artefact_id
               WHERE c.reasoning_configured IS NOT NULL GROUP BY a.arm),
        -- and the DRAWS, for an arm read on a pass that recorded no reasoning field
        -- on the wire. The source is carried beside the answer, never merged into it.
        dr AS (SELECT a.arm, SUM(c.n_reasoning) AS nr, SUM(c.n_draws) AS nd
               FROM cells c JOIN artefacts a ON a.id = c.artefact_id
               WHERE c.n_reasoning IS NOT NULL GROUP BY a.arm),
        res AS (
          SELECT x.arm,
                 CASE WHEN rc.lo IS NOT NULL AND rc.lo = rc.hi THEN rc.lo
                      WHEN dr.nr > 0 THEN 'on'
                      WHEN dr.nd > 0 THEN 'off' END AS reasoning,
                 CASE WHEN rc.lo IS NOT NULL AND rc.lo = rc.hi
                        THEN 'reasoning_configured on the wire'
                      WHEN dr.nr > 0
                        THEN 'from the draws (' || dr.nr || ' of ' || dr.nd
                             || ' reasoned); no reasoning field on the wire'
                      WHEN dr.nd > 0
                        THEN 'from the draws (0 of ' || dr.nd
                             || ' reasoned); no reasoning field on the wire'
                      END AS reasoning_source,
                 CASE WHEN rc.lo IS NOT NULL AND rc.lo <> rc.hi THEN 1 ELSE 0 END
                   AS reasoning_ambiguous
          FROM arms x LEFT JOIN rc ON rc.arm = x.arm LEFT JOIN dr ON dr.arm = x.arm)
        SELECT res.arm,
               (SELECT model_id FROM artefacts WHERE arm = res.arm
                 AND model_id IS NOT NULL ORDER BY id LIMIT 1) AS model_label,
               res.reasoning, res.reasoning_source, res.reasoning_ambiguous,
               CASE WHEN res.reasoning = 'on' THEN 2 ELSE 1 END AS group_order,
               ROW_NUMBER() OVER (
                 PARTITION BY CASE WHEN res.reasoning = 'on' THEN 2 ELSE 1 END
                 ORDER BY res.arm) AS row_order
        FROM res
 """,
 # THE FLOOR THE DUTY FIGURE JUDGES AGAINST, with its provenance on the row.
 # max(sentence floor, 0.30); the draw band where an arm has no sentence floor at all,
 # labelled so a reader never has to work out which quantity a verdict used.
 "duty_floor_used": """
    CREATE VIEW duty_floor_used AS
        SELECT a.arm, a.draw_floor AS floor_draw,
               f.sentence_floor_parametric AS floor_sentence,
               f.sentence_floor_robust AS floor_sentence_robust,
               f.sentence_n, f.floors_diverge,
               CASE WHEN f.sentence_floor_parametric IS NOT NULL
                      THEN MAX(f.sentence_floor_parametric, 0.30)
                    ELSE a.draw_floor END AS floor_used,
               CASE WHEN f.sentence_floor_parametric IS NOT NULL
                      THEN 'sentence floor (all_reported pair set), floored at 0.30'
                    WHEN a.draw_floor IS NOT NULL
                      THEN 'DRAW BAND: this arm has no sentence floor ('
                           || COALESCE(f.sentence_refused_reason,
                                       'no neutral readings') || ')'
                    ELSE NULL END AS floor_used_source
        FROM duty_floor_any a
        LEFT JOIN duty_floor f ON f.arm = a.arm AND f.floor_set = 'all_reported'
 """,
 # ── THE DUTY-FIGURE ROW: ONE PER ARM x PAIR, READ OR MISSING WITH A REASON ─────
 "duty_overview": """
    CREATE VIEW duty_overview AS
        SELECT m.arm, m.model_label, m.reasoning, m.reasoning_source,
               m.group_order, m.row_order,
               p.pair_id, p.name AS pair_label,
               CASE p.pair_id WHEN 'prawns' THEN 1 WHEN '1' THEN 2 WHEN '3' THEN 3
                              WHEN '4' THEN 4 WHEN '5' THEN 5 WHEN '7' THEN 6 END
                 AS panel_order,
               t.gap_pro_first AS stated_gap_pro_first,
               t.gap_priv_first AS stated_gap_priv_first,
               t.stated_gap, t.order_spread, t.exact_zero_in_an_order AS exact_zero_flag,
               t.n_orders,
               s.ev_pro AS sep_pro, s.ev_priv AS sep_priv, s.separate_gap,
               s.headroom_above_pro,
               -- THE SEPARATE GAP WITH THE CONVENTION STATED: RAW digits,
               -- professional minus private, never divided by anything, so the
               -- prawns pair (which has no neutral baseline) is carried on the same
               -- basis as the authored pairs.
               sc.ev_pro AS stated_condition_pro,
               sc.ev_priv AS stated_condition_priv,
               sc.gap AS stated_condition_gap,
               sc.run AS stated_condition_run,
               CASE WHEN sc.gap IS NOT NULL THEN 'read' ELSE 'missing' END
                 AS stated_condition_status,
               CASE WHEN sc.gap IS NOT NULL THEN NULL
                    ELSE COALESCE(
                      (SELECT 'unread: ' || MIN(u.unread_reason) FROM duty_cells u
                        WHERE u.arm = m.arm AND u.pair_id = p.pair_id
                          AND u.condition = 'stated' AND u.status = 'unread'),
                      'no stated-condition cell exists for this arm and pair')
                    END AS stated_condition_reason,
               fu.floor_draw, fu.floor_sentence, fu.floor_used, fu.floor_used_source,
               CASE WHEN t.stated_gap IS NULL OR fu.floor_used IS NULL THEN NULL
                    WHEN ABS(t.stated_gap) > fu.floor_used THEN 1 ELSE 0 END
                 AS convention_stated,
               CASE
                 WHEN t.stated_gap IS NULL OR fu.floor_used IS NULL
                      OR s.separate_gap IS NULL THEN NULL
                 WHEN ABS(t.stated_gap) <= fu.floor_used THEN 'NO CONVENTION STATED'
                 WHEN ABS(s.separate_gap - t.stated_gap) <= fu.floor_used
                  AND ABS(s.separate_gap) <= fu.floor_used
                   THEN CASE WHEN ABS(s.separate_gap - t.stated_gap)
                                  < ABS(s.separate_gap) THEN 'APPLIED'
                             WHEN ABS(s.separate_gap)
                                  < ABS(s.separate_gap - t.stated_gap)
                               THEN 'NOT APPLIED'
                             ELSE 'NO VERDICT (equidistant)' END
                 WHEN ABS(s.separate_gap - t.stated_gap) <= fu.floor_used
                   THEN 'APPLIED'
                 WHEN ABS(s.separate_gap) <= fu.floor_used THEN 'NOT APPLIED'
                 ELSE 'PARTLY' END AS verdict,
               CASE WHEN t.stated_gap IS NULL OR fu.floor_used IS NULL
                      OR s.separate_gap IS NULL THEN NULL
                    WHEN ABS(t.stated_gap) <= fu.floor_used THEN 0
                    WHEN ABS(s.separate_gap - t.stated_gap) <= fu.floor_used
                     AND ABS(s.separate_gap) <= fu.floor_used THEN 1
                    ELSE 0 END AS verdict_tie_broken,
               CASE WHEN t.stated_gap IS NULL OR fu.floor_used IS NULL THEN NULL
                    WHEN fu.floor_used > ABS(t.stated_gap) / 2.0 THEN 1
                    ELSE 0 END AS low_resolution,
               -- WHERE THE STATED CONDITION SITS AGAINST THE ARM'S OWN COMPARISON
               -- GAP: raw digits against raw digits, both read on the same arm,
               -- judged against that arm's own floor_used. WITHIN takes precedence
               -- over BELOW and ABOVE. No neutral baseline is needed, so prawns is
               -- counted on the same basis as the authored pairs.
               CASE WHEN sc.gap IS NULL OR t.stated_gap IS NULL
                      OR fu.floor_used IS NULL THEN NULL
                    WHEN ABS(sc.gap - t.stated_gap) <= fu.floor_used THEN 'WITHIN'
                    WHEN sc.gap < t.stated_gap THEN 'BELOW'
                    ELSE 'ABOVE' END AS stated_condition_vs_stated,
               CASE WHEN t.stated_gap IS NULL OR fu.floor_used IS NULL
                      OR s.separate_gap IS NULL THEN NULL
                    WHEN ABS(t.stated_gap) <= fu.floor_used THEN NULL
                    ELSE s.separate_gap / t.stated_gap END AS share,
               CASE WHEN t.stated_gap IS NOT NULL THEN 'read' ELSE 'missing' END
                 AS status,
               -- why a value is absent, read from the database
               TRIM(COALESCE(
                 CASE WHEN t.stated_gap IS NULL THEN
                   COALESCE((SELECT 'comparison unread: ' || MIN(u.unread_reason)
                               FROM duty_cells u
                              WHERE u.arm = m.arm AND u.pair_id = p.pair_id
                                AND u.question = 'CMPN' AND u.status = 'unread'),
                            'no comparison cell exists for this arm and pair')
                 END || ' ', '')
                 || COALESCE(
                 CASE WHEN s.separate_gap IS NULL THEN
                   COALESCE((SELECT 'separate unread: ' || MIN(u.unread_reason)
                               FROM duty_cells u
                              WHERE u.arm = m.arm AND u.pair_id = p.pair_id
                                AND u.question = 'BLAME' AND u.condition = 'bare'
                                AND u.status = 'unread'),
                            CASE WHEN NOT EXISTS (
                                   SELECT 1 FROM duty_cells u2
                                    WHERE u2.arm = m.arm AND u2.question = 'BLAME'
                                      AND u2.condition = 'bare')
                                 THEN 'comparison only: this arm has no separate-case '
                                      || 'cell in any duty run'
                                 ELSE 'no separate cell exists for this arm and pair'
                            END)
                 END, '')) AS missing_reason
        FROM duty_overview_arms m
        CROSS JOIN duty_pairs p
        LEFT JOIN duty_stated_gap t ON t.arm = m.arm AND t.pair_id = p.pair_id
        LEFT JOIN duty_separate_gap s ON s.arm = m.arm AND s.pair_id = p.pair_id
        LEFT JOIN duty_floor_used fu ON fu.arm = m.arm
        LEFT JOIN duty_remedy_gap sc ON sc.arm = m.arm AND sc.pair_id = p.pair_id
                                    AND sc.condition = 'stated'
        WHERE p.status = 'reported'
 """,
 "duty_overview_panels": """
    CREATE VIEW duty_overview_panels AS
        SELECT f.pair_id, f.pair_label, MIN(f.panel_order) AS panel_order,
               COUNT(f.stated_gap) AS n_arms_stated,
               COUNT(f.separate_gap) AS n_arms_separate,
               SUM(COALESCE(f.convention_stated, 0)) AS n_convention_stated,
               SUM(CASE WHEN f.verdict = 'APPLIED' THEN 1 ELSE 0 END) AS n_applied,
               SUM(CASE WHEN f.verdict = 'PARTLY' THEN 1 ELSE 0 END) AS n_partly,
               SUM(CASE WHEN f.verdict = 'NOT APPLIED' THEN 1 ELSE 0 END)
                 AS n_not_applied,
               SUM(CASE WHEN f.verdict = 'NO CONVENTION STATED' THEN 1 ELSE 0 END)
                 AS n_no_convention,
               SUM(CASE WHEN f.verdict IS NULL THEN 1 ELSE 0 END) AS n_no_verdict,
               SUM(COALESCE(f.low_resolution, 0)) AS n_low_resolution,
               -- the stated condition, per panel
               COUNT(f.stated_condition_gap) AS n_stated_condition,
               SUM(CASE WHEN f.stated_condition_vs_stated = 'BELOW' THEN 1 ELSE 0 END)
                 AS n_sc_below_stated,
               SUM(CASE WHEN f.stated_condition_vs_stated = 'WITHIN' THEN 1 ELSE 0 END)
                 AS n_sc_within_stated,
               SUM(CASE WHEN f.stated_condition_vs_stated = 'ABOVE' THEN 1 ELSE 0 END)
                 AS n_sc_above_stated,
               SUM(CASE WHEN f.stated_condition_gap IS NOT NULL
                         AND f.stated_condition_vs_stated IS NULL THEN 1 ELSE 0 END)
                 AS n_sc_unplaced,
               -- whether this panel has a neutral baseline at all: prawns does not
               -- (no neutral sentences exist for it), so its remedy class is refused
               -- rather than borrowed; the stated-condition columns above need no
               -- baseline
               CASE WHEN EXISTS (SELECT 1 FROM duty_texts dt
                                  WHERE dt.pair_id = f.pair_id
                                    AND dt.kind = 'neutral')
                    THEN 'read: three neutral sentences'
                    ELSE 'NONE: this pair has no neutral sentences, so it has no '
                         || 'remedy class and no recovery ratio' END
                 AS neutral_baseline,
               COUNT(*) AS n_rows
        FROM duty_overview f
        GROUP BY f.pair_id, f.pair_label
 """,
 # ── ONE ARM PER MODEL, ACROSS BOTH CONFIGURATIONS ────────────────────────────
 # duty_model_arm holds one arm per model PER configuration set, which is right for
 # the off/on dispersion blocks and wrong for a single CV per panel: a model present
 # in both sets would be counted twice. The reasoning-off arm is preferred where both
 # exist. The medians and dispersion below use the same linear interpolation as
 # statistics.median and statistics.quantiles.
 "duty_model_arm_one": """
    CREATE VIEW duty_model_arm_one AS
        SELECT model_id,
               COALESCE(MAX(CASE WHEN config_set = 'off' THEN arm END),
                        MAX(CASE WHEN config_set = 'on' THEN arm END)) AS arm,
               CASE WHEN MAX(CASE WHEN config_set = 'off' THEN 1 ELSE 0 END) = 1
                    THEN 'off' ELSE 'on' END AS config_set_used,
               COUNT(*) AS n_config_sets
        FROM duty_model_arm GROUP BY model_id
 """,
 "duty_overview_cv": """
    CREATE VIEW duty_overview_cv AS
        SELECT f.pair_id, COUNT(*) AS n_models_cv, AVG(f.stated_gap) AS mean_gap,
               CASE WHEN COUNT(*) >= 2
                    THEN SQRT(MAX(SUM((f.stated_gap - m.mg) * (f.stated_gap - m.mg))
                                  / (COUNT(*) - 1), 0.0)) END AS sd_stated,
               CASE WHEN COUNT(*) >= 2 AND ABS(m.mg) >= 0.05
                    THEN SQRT(MAX(SUM((f.stated_gap - m.mg) * (f.stated_gap - m.mg))
                                  / (COUNT(*) - 1), 0.0)) / m.mg END AS cv_stated,
               CASE WHEN ABS(m.mg) < 0.05
                    THEN 'mean within 0.05 of zero: a CV would be an artefact of'
                         || ' the denominator' END AS cv_refused_reason
        FROM duty_overview f
        JOIN duty_model_arm_one o ON o.arm = f.arm
        JOIN (SELECT f2.pair_id, AVG(f2.stated_gap) AS mg FROM duty_overview f2
               JOIN duty_model_arm_one o2 ON o2.arm = f2.arm
               WHERE f2.stated_gap IS NOT NULL GROUP BY f2.pair_id) m
          ON m.pair_id = f.pair_id
        WHERE f.stated_gap IS NOT NULL
        GROUP BY f.pair_id, m.mg
 """,
 "duty_overview_stated_ranked": """
    CREATE VIEW duty_overview_stated_ranked AS
        SELECT pair_id, stated_gap AS g,
               ROW_NUMBER() OVER (PARTITION BY pair_id ORDER BY stated_gap) - 1 AS i,
               COUNT(*) OVER (PARTITION BY pair_id) AS n
        FROM duty_overview WHERE stated_gap IS NOT NULL
 """,
 "duty_overview_sc_ranked": """
    CREATE VIEW duty_overview_sc_ranked AS
        SELECT pair_id, stated_condition_gap AS g,
               ROW_NUMBER() OVER (PARTITION BY pair_id
                                  ORDER BY stated_condition_gap) - 1 AS i,
               COUNT(*) OVER (PARTITION BY pair_id) AS n
        FROM duty_overview WHERE stated_condition_gap IS NOT NULL
 """,
 "duty_overview_share_ranked": """
    CREATE VIEW duty_overview_share_ranked AS
        SELECT pair_id, share AS g,
               ROW_NUMBER() OVER (PARTITION BY pair_id ORDER BY share) - 1 AS i,
               COUNT(*) OVER (PARTITION BY pair_id) AS n
        FROM duty_overview WHERE share IS NOT NULL
 """,
 # ── SIGNAL-TO-NOISE ───────────────────────────────────────────────────────────
 # The stated gap divided by the arm's own wording floor (floor_used), signed, per
 # arm and pair, with convention_stated and low_resolution carried beside it as
 # duty_overview defines them. NULL with a reason where either term is missing; a
 # floor_used of zero cannot occur (it is floored at 0.30 or is a draw band, which
 # is itself floored at 0.30), so no division guard is needed beyond NULL.
 "duty_signal_to_noise": """
    CREATE VIEW duty_signal_to_noise AS
        SELECT f.arm, f.model_label, f.reasoning, f.pair_id, f.pair_label,
               f.panel_order, f.stated_gap, f.floor_used, f.floor_used_source,
               CASE WHEN f.stated_gap IS NULL OR f.floor_used IS NULL THEN NULL
                    ELSE f.stated_gap / f.floor_used END AS signal_to_noise,
               f.convention_stated, f.low_resolution,
               CASE WHEN f.stated_gap IS NULL THEN 'no stated gap: ' || f.missing_reason
                    WHEN f.floor_used IS NULL THEN 'no floor_used for this arm'
                    END AS refused_reason
        FROM duty_overview f
 """,
 "duty_snr_ranked": """
    CREATE VIEW duty_snr_ranked AS
        SELECT arm, signal_to_noise AS g,
               ROW_NUMBER() OVER (PARTITION BY arm ORDER BY signal_to_noise) - 1 AS i,
               COUNT(*) OVER (PARTITION BY arm) AS n
        FROM duty_signal_to_noise WHERE signal_to_noise IS NOT NULL
 """,
 # per-arm median over the pairs it has a value for, by the linear interpolation
 # statistics.median uses; n_pairs is the denominator and n_pairs_refused what fell out
 "duty_signal_to_noise_arm": """
    CREATE VIEW duty_signal_to_noise_arm AS
        WITH q AS (SELECT arm, n, (n - 1) * 0.5 AS p FROM duty_snr_ranked
                   GROUP BY arm, n)
        SELECT a.arm, q.n AS n_pairs,
               (SELECT COUNT(*) FROM duty_signal_to_noise s
                 WHERE s.arm = a.arm AND s.signal_to_noise IS NULL) AS n_pairs_refused,
               (SELECT g FROM duty_snr_ranked r WHERE r.arm = q.arm
                   AND r.i = CAST(q.p AS INTEGER))
               + (q.p - CAST(q.p AS INTEGER))
                 * (COALESCE((SELECT g FROM duty_snr_ranked r WHERE r.arm = q.arm
                                 AND r.i = CAST(q.p AS INTEGER) + 1),
                             (SELECT g FROM duty_snr_ranked r WHERE r.arm = q.arm
                                 AND r.i = CAST(q.p AS INTEGER)))
                    - (SELECT g FROM duty_snr_ranked r WHERE r.arm = q.arm
                          AND r.i = CAST(q.p AS INTEGER))) AS median_signal_to_noise
        FROM (SELECT DISTINCT arm FROM duty_signal_to_noise) a
        LEFT JOIN q ON q.arm = a.arm
 """,
 # ── 9. THE ASSUMPTION-OF-RISK GROUPING ────────────────────────────────────────
 "duty_risk_grouping": """
    CREATE VIEW duty_risk_grouping AS
        SELECT g.arm, p.victim_chose_arrangement AS chose,
               COUNT(*) AS n_pairs, AVG(g.stated_gap) AS mean_stated_gap,
               AVG(s.separate_gap) AS mean_separate_gap,
               COUNT(s.separate_gap) AS n_separate
        FROM duty_stated_gap g
        JOIN duty_pairs p ON p.pair_id = g.pair_id
        LEFT JOIN duty_separate_gap s ON s.arm = g.arm AND s.pair_id = g.pair_id
        WHERE p.status = 'reported'
        GROUP BY g.arm, p.victim_chose_arrangement
 """,
}


# ──────────────────────────────────────────────────────────────────────────────
# POPULATING THE REFERENCE TABLES
# ──────────────────────────────────────────────────────────────────────────────
def _sha(t):
    return hashlib.sha256(t.encode()).hexdigest()


def populate_pairs(con, bank, bank_sha, pairs=PAIRS):
    """Insert the duty_pairs rows. `bank` names the bank file the pairs' texts come
    from and `bank_sha` is its sha256 (for the public banks, banks/duty_texts.jsonl
    and its digest in banks/MANIFEST.json)."""
    for pid, name, status, pl, vl, chose, note in pairs:
        con.execute(
            "INSERT INTO duty_pairs(pair_id,name,status,retired_reason,pro_label,"
            "priv_label,victim_chose_arrangement,bank,bank_sha,notes) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (pid, name, status, None, pl, vl, chose, bank, bank_sha, note))


def populate_model_arm(con, off, on):
    """Insert duty_model_arm: one arm per model per configuration set, from two dicts
    {model_id: arm}, one for reasoning off and one for reasoning on."""
    for mid, arm in off.items():
        con.execute("INSERT INTO duty_model_arm(model_id,arm,config_set) "
                    "VALUES(?,?,'off')", (mid, arm))
    for mid, arm in on.items():
        con.execute("INSERT INTO duty_model_arm(model_id,arm,config_set) "
                    "VALUES(?,?,'on')", (mid, arm))


# ──────────────────────────────────────────────────────────────────────────────
# IN-DATABASE CHECKS. (name, ok, n, detail); a check passes when ok == n.
# ──────────────────────────────────────────────────────────────────────────────
def tests(con):
    T = []

    nu = con.execute("SELECT COUNT(*) FROM duty_cells "
                     "WHERE status='unread'").fetchone()[0]
    nur = con.execute("SELECT COUNT(*) FROM duty_cells WHERE status='unread' "
                      "AND unread_reason IS NOT NULL AND unread_reason<>''"
                      ).fetchone()[0]
    T.append(("duty: every unread cell carries a reason", nur, nu, []))

    # ── every text hashes to its stored sha and its word count is the text's ────
    bad = []
    rows = con.execute("SELECT text_id,text,sha256,word_count FROM duty_texts"
                       ).fetchall()
    for tid, txt, sh, wc in rows:
        if _sha(txt) != sh:
            bad.append(f"{tid}: text does not hash to its stored sha")
        if len(txt.split()) != wc:
            bad.append(f"{tid}: word_count {wc} is not the text's")
    T.append(("duty: every duty_texts row hashes to its stored sha", len(rows) - len(bad),
              len(rows), bad[:6]))
    nulltid = con.execute("SELECT COUNT(*) FROM duty_cells WHERE text_id IS NULL "
                          "AND text_id_reason IS NULL").fetchone()[0]
    ntid = con.execute("SELECT COUNT(*) FROM duty_cells").fetchone()[0]
    T.append(("duty: a cell with no text_id says why", ntid - nulltid, ntid, []))
    orphan = con.execute("SELECT COUNT(*) FROM duty_cells c WHERE c.text_id IS NOT "
                         "NULL AND NOT EXISTS(SELECT 1 FROM duty_texts t WHERE "
                         "t.text_id=c.text_id)").fetchone()[0]
    T.append(("duty: every text_id resolves to a frozen text", ntid - orphan, ntid,
              []))

    # ── no reported contrast spans two providers ─────────────────────────────────
    span = con.execute("""SELECT s.run,s.arm,s.pair_id,s.n_providers,s.providers
        FROM duty_provider_span s JOIN duty_pairs p ON p.pair_id=s.pair_id
        WHERE p.status='reported' AND s.n_providers > 1""").fetchall()
    tot = con.execute("""SELECT COUNT(*) FROM duty_provider_span s
        JOIN duty_pairs p ON p.pair_id=s.pair_id
        WHERE p.status='reported'""").fetchone()[0]
    T.append(("duty: no reported contrast spans two providers", tot - len(span), tot,
              [f"{r[1]} pair {r[2]}: {r[4]}" for r in span[:6]]))
    # and the same question asked of every view that differences two cells
    multi = []
    for v in ("duty_stated_gap", "duty_separate_gap", "duty_remedy_gap",
              "duty_cause", "duty_neutral_gap"):
        q = (f"SELECT COUNT(*) FROM {v} v JOIN duty_provider_span s "
             f"ON s.arm=v.arm AND s.pair_id=v.pair_id WHERE s.n_providers>1")
        k = con.execute(q).fetchone()[0]
        if k:
            multi.append(f"{v}: {k} row(s)")
    T.append(("duty: no view row differences cells from two providers",
              5 - len(multi), 5, multi))

    # ── no non-reported pair reaches a reported view ─────────────────────────────
    det, n = [], 0
    for v in ("duty_reported", "duty_stated_gap", "duty_separate_gap",
              "duty_share", "duty_neutral_baseline", "duty_remedy_effect",
              "duty_cause", "duty_spread"):
        n += 1
        k = con.execute(f"SELECT COUNT(*) FROM {v} WHERE pair_id IN "
                        f"(SELECT pair_id FROM duty_pairs WHERE status<>'reported')"
                        ).fetchone()[0]
        if k:
            det.append(f"{v} leaks {k} non-reported row(s)")
    T.append(("duty: no retired or control pair reaches a reported view",
              n - len(det), n, det))

    # ── signal-to-noise: the view against Python, row by row and by median ───────
    rows = con.execute("SELECT arm,stated_gap,floor_used,signal_to_noise,"
                       "refused_reason FROM duty_signal_to_noise").fetchall()
    bad = [f"{a}: {g}/{fu} -> {s}" for a, g, fu, s, why in rows
           if (s is None) != (g is None or fu is None)
           or (s is not None and abs(s - g / fu) > 1e-12)
           or (s is None and not why)]
    T.append(("duty: signal_to_noise equals stated_gap / floor_used on every row, and "
              "every NULL carries a reason", len(rows) - len(bad), len(rows), bad[:6]))
    per = {}
    for a, g, fu, s, _ in rows:
        if s is not None:
            per.setdefault(a, []).append(s)
    arms = con.execute("SELECT arm,n_pairs,median_signal_to_noise "
                       "FROM duty_signal_to_noise_arm").fetchall()
    bad = [f"{a}: view {m} n {n} vs python {statistics.median(per[a]) if a in per else None}"
           for a, n, m in arms
           if (a in per) != (m is not None)
           or (a in per and (abs(m - statistics.median(per[a])) > 1e-12
                             or n != len(per[a])))]
    T.append(("duty: duty_signal_to_noise_arm's median equals statistics.median per arm",
              len(arms) - len(bad), len(arms), bad[:6]))
    return T
