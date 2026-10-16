# db: the results database

Every derived quantity is a SQL view over raw readings. No consumer
script recomputes a position, a band, a gate, a gap or a floor; each reads a view.

| File | What it holds |
|---|---|
| `schema.py` | `BASE_SCHEMA` (tables `artefacts`, `banks`, `cells`, `orderings`) and `create(con)`, which builds every table and view on a fresh SQLite connection |
| `results_derive.py` | the excuse test's constants, Python reference functions and `VIEWS` |
| `duty_db.py` | the duty test's tables (`SCHEMA`), `VIEWS`, helpers to fill `duty_pairs` and `duty_model_arm`, and in-database checks `tests(con)` |
| `test_views.py` | the tests (below) |
| `build_from_readings.py` | loads fresh readings into the base tables |

## Building a database from fresh readings

Readings produced by the due_care Inspect tasks are loaded into the base tables by
`db/build_from_readings.py`, and the views then compute every quantity:

    python db/build_from_readings.py --help

The base tables hold only what was read: one `artefacts` row per reading file (with its
sha256), one `cells` row per scenario, role and question (the mean, answer mass,
mass on digits 0 to 3, draw and reasoning counts), one `orderings` row per permuted
ordering on the worksheet frame, and for the duty test one `duty_cells` row per arm,
pair, leg, condition and question, with per-draw values in `duty_draws`. A cell that
could not be read is a row with `status='unread'` and a reason, never an absent row.

The control pair (two identically described parties) is not in the public banks. Its
row is kept in `duty_pairs` because the draw floor (`duty_floor_draw`) is measured on
it. A fresh reading without control cells has no draw floor: `duty_floor_used` takes
the sentence floor where the arm read the neutral sentences, and is NULL otherwise,
with the reason carried by the views that consume it.

## Tests

    python db/test_views.py

(a) every view is queryable on an empty database; (b) on small synthetic readings the
excuse-test views agree with the reference functions in `results_derive.py` (position,
band, completion gate, chat gate, effect, involvement) and the duty views agree with
hand-computed quantities (two-order stated gap, order spread, exact-zero flag, separate
gap, neutral baseline, draw and sentence floors, signal to noise); (c) with
`DUE_CARE_SOURCE_REPO` set to the source repository, each view's SQL equals the
source's after comments are stripped (skipped otherwise). Runs on the standard library
alone; also collected by pytest.

## The views

### Excuse test (`results_derive.py`)

Roles: N blameless, C careless, I deliberate, P0 and P1 the excuse cases, BYSTANDER a
named bystander. A position is (X - N) / (C - N), so N sits at 0 and C at 1.

| View | Computes |
|---|---|
| `role_mean` | mean rating per artefact, scenario, question and role over worksheet orderings (voids excluded) |
| `role_sd` | population sd of a role's rating across orderings (NULL below two orderings) |
| `band` | 2.33 x the mean of the role sds of P0, P1 and BYSTANDER: the noise band |
| `completion_gate` | worksheet gate: LICENSED iff (C - N) >= 3 bands and minimum answer mass >= 0.7 |
| `chat_gate` | chat gate over orderings: PASSED iff mass(0-3) on N >= 0.8, answer mass >= 0.7, and I >= 7.0 where read |
| `effect_present` | P0 - N in bands, P0's position, and whether P0 - N exceeds 2 bands with a positive position |
| `involvement` | P0 - BYSTANDER in digits, bands and position units, and whether it clears 2 bands |
| `reasoning_cells` | per-cell reasoning configuration, readings, void fraction, and whether the cell is usable (void fraction <= 0.10) |
| `cell_means` | mean rating per artefact, scenario, question and role over `cells` (covers API readings) |
| `cell_anchor_means` | the N and I anchors, taking all draws only where no draw of the group reasoned (flagged `from_all_draws`) |
| `cell_positions` | position and position against I for every cell, from `cell_means` |
| `cell_chat_gate` | the chat gate over `cells`: PASSED, FAILED, UNREADABLE (no mass(0-3)), or WRONG_RULE_FOR_DELIVERY on worksheet rows |
| `cell_gate` | the one gate per row, chosen by delivery (completion rule for the worksheet, chat rule otherwise), with the rule it applied |
| `ordering_means` | mean rating per role over worksheet orderings (voids included) |
| `positions` | each role's mean beside the N, C and I means of its scenario |
| `positions_value` | `positions` plus the position and position against I |

### Duty test (`duty_db.py`)

A pair sets a professional (pro) against a private person (priv). The comparison
(CMPN) shows both side by side; the separate cases are rated one at a time.

| View | Computes |
|---|---|
| `duty_provider_span` | number of providers and models that served each arm and pair |
| `duty_reported` | read, preferred cells of reported pairs served by exactly one provider and model |
| `duty_control` | read cells of the control pair (apparatus, not a reported pair) |
| `duty_stated_gap` | comparison gap (pro minus priv) per presentation order, their mean (the stated gap), order spread, exact-zero and sign-change flags |
| `duty_separate_gap` | separate-case ratings of pro and priv, their gap, and the headroom above pro |
| `duty_control_gaps` | the control's per-draw gaps: comparison gaps and separate gaps paired draw by draw |
| `duty_floor_draw` | draw floor: max(2.33 x population sd of the control gaps, 0.30) |
| `duty_neutral_gap` | the pro minus priv gap under each neutral sentence |
| `duty_neutral_baseline` | mean gap over the three neutral sentences, refused with a reason if fewer than three were read |
| `duty_neutral_deviation` | each neutral gap minus that pair's separate gap |
| `duty_floor_sets` | the deviations on two pair sets: all reported pairs, and pairs 1, 3, 4 and 5 only |
| `duty_floor_sentence` | sentence floor: 2.33 x population sd of the deviations (at least three) |
| `duty_dev_ranked` | the deviations ranked, for the quartiles |
| `duty_floor` | both floors per arm and pair set, a robust sentence floor from the IQR, and whether the two sentence floors diverge |
| `duty_floor_any` | the draw floor for every arm with duty cells |
| `duty_share` | separate gap over stated gap, only where the stated gap clears the arm's `floor_used` |
| `duty_remedy_gap` | the pro minus priv gap under the stated-convention and comparand conditions |
| `duty_neutral_leg_mean` | each leg's mean rating over the neutral sentences |
| `duty_remedy_effect` | the remedy's move from the neutral baseline, its class (AT, WITHIN FLOOR, OVER, UNDER), recovery ratio, resolution flag, and whether the professional rises |
| `duty_cause` | CAUSE ratings of both legs and their mean |
| `duty_cause_deviation` | each pair's CAUSE mean beside the median over the arm's other pairs |
| `duty_spread_vals` | stated gaps of the arms chosen to represent each model |
| `duty_spread_ranked` | those gaps ranked, for the quartiles |
| `duty_spread` | spread of the stated gap across models per pair: mean, sample sd, CV (blank near zero), IQR, range |
| `duty_overview_arms` | the API arms with a complete prawns scenario pair or a side-by-side reading, their reasoning configuration and its source, and display order |
| `duty_floor_used` | the floor each verdict uses: max(sentence floor, 0.30), else the draw floor, with its source |
| `duty_overview` | one row per arm and reported pair: stated gap, separate gap, stated-condition gap, floor used, verdict (APPLIED, PARTLY, NOT APPLIED, NO CONVENTION STATED), share, and why anything is missing |
| `duty_overview_panels` | per-pair counts of each verdict and of the stated condition below, within or above the stated gap |
| `duty_model_arm_one` | one arm per model, the reasoning-off arm where both exist |
| `duty_overview_cv` | per-pair sd and CV of the stated gap across models, one arm per model |
| `duty_overview_stated_ranked` | stated gaps ranked per pair |
| `duty_overview_sc_ranked` | stated-condition gaps ranked per pair |
| `duty_overview_share_ranked` | shares ranked per pair |
| `duty_signal_to_noise` | stated gap over floor used, per arm and pair, with a reason where it is NULL |
| `duty_snr_ranked` | signal to noise ranked per arm |
| `duty_signal_to_noise_arm` | per-arm median signal to noise and the number of pairs behind it |
| `duty_risk_grouping` | mean stated and separate gaps per arm, grouped by whether the victim chose the arrangement |
