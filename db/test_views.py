#!/usr/bin/env python3
"""test_views.py : checks the database views against their reference definitions.

  (a) every view is queryable on an empty database;
  (b) on small synthetic readings, the excuse-test views agree with the Python
      reference functions in results_derive, and the duty views agree with
      hand-computed quantities; duty_db.tests() passes;
  (c) if DUE_CARE_SOURCE_REPO points at the source repository, each view's SQL equals
      the source's after stripping `--` comments and normalising whitespace.

Run from the repository root:  python db/test_views.py   (or: pytest db/test_views.py)
Prints PASS/FAIL per check and exits nonzero on any failure.
"""
import ast
import json
import math
import os
import re
import sqlite3
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from db import duty_db, results_derive as RD, schema  # noqa: E402

TOL = 1e-9
_RESULTS = []


def check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok
                                                   else ""))
    return ok


def close(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def fresh():
    con = sqlite3.connect(":memory:")
    schema.create(con)
    return con


# ── (a) EMPTY DATABASE ────────────────────────────────────────────────────────
def test_empty_database():
    con = fresh()
    names = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")]
    expected = set(RD.VIEWS) | set(duty_db.VIEWS)
    check("empty db: every exported view exists", set(names) == expected,
          f"missing {sorted(expected - set(names))}")
    bad = []
    for n in names:
        try:
            con.execute(f'SELECT * FROM "{n}" LIMIT 1').fetchall()
        except sqlite3.Error as exc:
            bad.append(f"{n}: {exc}")
    check(f"empty db: all {len(names)} views are queryable", not bad, "; ".join(bad))
    for name, ok, n, det in duty_db.tests(con):
        check(f"empty db: {name}", ok == n, f"{ok} of {n} {det}")


# ── (b) SYNTHETIC READINGS ────────────────────────────────────────────────────
# Worksheet artefact: 5 orderings x 6 roles, one scenario, BLAME.
WS = {
    "N":         [0.40, 0.55, 0.30, 0.62, 0.48],
    "C":         [7.10, 7.45, 6.90, 7.30, 7.05],
    "I":         [8.60, 8.80, 8.70, 8.40, 8.75],
    "P0":        [3.20, 3.65, 2.95, 3.40, 3.10],
    "P1":        [2.10, 2.40, 1.95, 2.30, 2.05],
    "BYSTANDER": [1.20, 1.05, 1.40, 1.15, 1.30],
}
WS_AMASS = [0.97, 0.95, 0.99, 0.96, 0.98]
# per-ordering digit distribution for the blameless cell (keys 0..9)
WS_PROBS_N = [[0.55, 0.25, 0.08, 0.04] + [0.08 / 6] * 6,
              [0.50, 0.25, 0.10, 0.05] + [0.10 / 6] * 6,
              [0.60, 0.20, 0.08, 0.04] + [0.08 / 6] * 6,
              [0.45, 0.25, 0.10, 0.06] + [0.14 / 6] * 6,
              [0.52, 0.24, 0.09, 0.05] + [0.10 / 6] * 6]
# Chat artefact: one row per role, sampled API reading; also carries the careless
# role on both prawns scenarios so the arm enters the duty figure's arm set.
CHAT = {"N": 0.20, "C": 6.80, "I": 8.30, "P0": 1.10, "P1": 0.90, "BYSTANDER": 0.40}
CHAT_MASS03 = 0.93
CHAT_AMASS = 0.99
ARM = "synth_api"


def load_excuse(con):
    con.execute("INSERT INTO artefacts(id,path,sha256,run,card,arm,delivery,readout,"
                "model_id,adapter_family,enters_tables) VALUES "
                "(1,'synthetic/worksheet.json','0'*64,'S1','pod','synth_local',"
                "'completion','logprob','example/base-model','lora',1)")
    con.execute("INSERT INTO artefacts(id,path,sha256,run,card,arm,delivery,readout,"
                "model_id,adapter_family,enters_tables) VALUES "
                "(2,'synthetic/chat.json','1'*64,'S2','api',?,'api','sampled',"
                "'example/api-model','none',1)", (ARM,))
    for role, evs in WS.items():
        for k, ev in enumerate(evs):
            frame = f"order{k}"
            con.execute("INSERT INTO orderings(artefact_id,family,role,question,frame,"
                        "ev,answer_mass,void) VALUES (1,'bicycle',?,'BLAME',?,?,?,0)",
                        (role, frame, ev, WS_AMASS[k]))
            probs = json.dumps(WS_PROBS_N[k]) if role == "N" else None
            con.execute("INSERT INTO cells(artefact_id,family,role,question,frame,ev,"
                        "answer_mass,probs,void) VALUES (1,'bicycle',?,'BLAME',?,?,?,?,0)",
                        (role, frame, ev, WS_AMASS[k], probs))
    for role, ev in CHAT.items():
        con.execute("INSERT INTO cells(artefact_id,family,role,question,frame,ev,"
                    "answer_mass,mass_0_3,mass_0_3_source,n_draws,n_void,"
                    "reasoning_configured,void) VALUES "
                    "(2,'bicycle',?,'BLAME','chat',?,?,?,?,20,1,'off',0)",
                    (role, ev, CHAT_AMASS, CHAT_MASS03 if role == "N" else None,
                     "sampled draws" if role == "N" else None))
    for fam, ev in (("prawns_home", 4.0), ("prawns_restaurant", 6.5)):
        con.execute("INSERT INTO cells(artefact_id,family,role,question,frame,ev,"
                    "answer_mass,void) VALUES (2,?,'C','BLAME','chat',?,0.99,0)",
                    (fam, ev))


def test_excuse_views():
    con = fresh()
    load_excuse(con)
    n, c, i = (RD.mean(WS[r]) for r in ("N", "C", "I"))
    p0, by = RD.mean(WS["P0"]), RD.mean(WS["BYSTANDER"])
    bd = RD.band(WS)

    # role means and the band
    for role, evs in WS.items():
        m = con.execute("SELECT m FROM role_mean WHERE artefact_id=1 AND role=?",
                        (role,)).fetchone()[0]
        check(f"excuse: role_mean {role} equals the mean over orderings",
              close(m, RD.mean(evs)), f"{m} vs {RD.mean(evs)}")
    vb = con.execute("SELECT band FROM band WHERE artefact_id=1").fetchone()[0]
    check("excuse: band equals results_derive.band", close(vb, bd), f"{vb} vs {bd}")

    # positions
    for role, evs in WS.items():
        pos, pni = con.execute("SELECT position, position_NI FROM positions_value "
                               "WHERE artefact_id=1 AND role=?", (role,)).fetchone()
        check(f"excuse: position {role} equals results_derive.position",
              close(pos, RD.position(RD.mean(evs), n, c)))
        check(f"excuse: position_NI {role} equals results_derive.position_NI",
              close(pni, RD.position_NI(RD.mean(evs), n, i)))
        cpos = con.execute("SELECT position FROM cell_positions WHERE artefact_id=1 "
                           "AND role=?", (role,)).fetchone()[0]
        check(f"excuse: cell_positions agrees with positions_value for {role}",
              close(cpos, pos))

    # completion gate
    amass = min(WS_AMASS)
    want_v, _ = RD.completion_gate((c - n) / bd, amass)
    v, cnb, am = con.execute("SELECT verdict, c_minus_n_bands, answer_mass FROM "
                             "completion_gate WHERE artefact_id=1").fetchone()
    check("excuse: completion_gate verdict equals results_derive.completion_gate",
          v == want_v, f"{v} vs {want_v}")
    check("excuse: completion_gate C-N in bands", close(cnb, (c - n) / bd))
    check("excuse: completion_gate answer mass is the minimum", close(am, amass))

    # chat gate over orderings (mass is the mean over orderings of each mass(0-3))
    mass = RD.mean([sum(p[:4]) for p in WS_PROBS_N])
    want_v, _ = RD.chat_gate(mass, amass, i)
    v, m03 = con.execute("SELECT verdict, mass_0_3 FROM chat_gate "
                         "WHERE artefact_id=1").fetchone()
    check("excuse: chat_gate mass(0-3) is the mean over orderings", close(m03, mass),
          f"{m03} vs {mass}")
    check("excuse: chat_gate verdict equals results_derive.chat_gate", v == want_v,
          f"{v} vs {want_v}")

    # effect and involvement
    pos0 = RD.position(p0, n, c)
    want = RD.effect_present(p0, n, bd, pos0)
    e, eb = con.execute("SELECT effect_present, p0_minus_n_bands FROM effect_present "
                        "WHERE artefact_id=1").fetchone()
    check("excuse: effect_present equals results_derive.effect_present",
          e == (1 if want else 0), f"{e} vs {want}")
    check("excuse: effect_present P0-N in bands", close(eb, (p0 - n) / bd))
    inv = RD.involvement(p0, by, n, c, bd)
    g, gb, gp, cl = con.execute("SELECT gap_digits, gap_bands, gap_position_units, "
                                "clears_2_bands FROM involvement WHERE artefact_id=1"
                                ).fetchone()
    check("excuse: involvement digits", close(g, inv["digits"]))
    check("excuse: involvement bands", close(gb, inv["bands"]))
    check("excuse: involvement position units", close(gp, inv["position_units"]))
    check("excuse: involvement clears_2_bands", cl == (1 if inv["clears_2_bands"]
                                                       else 0))

    # the chat artefact: gated over cells, by the chat rule
    want_v, _ = RD.chat_gate(CHAT_MASS03, CHAT_AMASS, CHAT["I"])
    v, m03 = con.execute("SELECT verdict, mass_0_3 FROM cell_chat_gate WHERE "
                         "artefact_id=2 AND family='bicycle'").fetchone()
    check("excuse: cell_chat_gate equals results_derive.chat_gate on the chat artefact",
          v == want_v and close(m03, CHAT_MASS03), f"{v} {m03}")
    v = con.execute("SELECT verdict FROM cell_chat_gate WHERE artefact_id=1"
                    ).fetchone()[0]
    check("excuse: cell_chat_gate declines a worksheet row",
          v == "WRONG_RULE_FOR_DELIVERY", v)
    kinds = dict(con.execute("SELECT artefact_id, rule_kind FROM cell_gate "
                             "WHERE family='bicycle'").fetchall())
    check("excuse: cell_gate picks the rule by delivery",
          kinds == {1: "completion", 2: "chat"}, str(kinds))
    pos = con.execute("SELECT position FROM cell_positions WHERE artefact_id=2 AND "
                      "family='bicycle' AND role='P0'").fetchone()[0]
    check("excuse: cell_positions on the chat artefact",
          close(pos, RD.position(CHAT["P0"], CHAT["N"], CHAT["C"])))
    usable, frac = con.execute("SELECT reasoning_usable, void_fraction FROM "
                               "reasoning_cells WHERE artefact_id=2 AND role='N'"
                               ).fetchone()
    ru, rf = RD.reasoning_usable(1, 20)
    check("excuse: reasoning_cells void fraction equals results_derive.reasoning_usable",
          usable == (1 if ru else 0) and close(frac, rf))
    return con


# ── duty: one arm, two reported pairs and the control ─────────────────────────
def duty_cell(con, cid, pair, leg, cond, q, ev=None, sp=None, sv=None, draws=(),
              n_draws=10, arm=ARM):
    con.execute(
        "INSERT INTO duty_cells(cell_id,run,arm,pair_id,leg,condition,question,"
        "text_id,text_id_reason,n_draws,n_parsed,ev,score_pro,score_priv,"
        "answer_mass_note,provider,model_id,wire_sha,source,artefact,artefact_sha,"
        "bank,preferred,status) VALUES (?,?,?,?,?,?,?,NULL,'synthetic',?,?,?,?,?,?,"
        "'example-provider','example/api-model','w','main','synthetic/duty.json',"
        "'2'*64,'synthetic',1,'read')",
        (cid, "D", arm, pair, leg, cond, q, n_draws, n_draws, ev, sp, sv,
         duty_db.MASS_NOTE))
    for k, (v, g) in enumerate(draws):
        con.execute("INSERT INTO duty_draws(cell_id,value_index,value,gap) "
                    "VALUES(?,?,?,?)", (cid, k, v, g))


CMPN = {"1": {"cmp_pro_first": (7.0, 3.0), "cmp_priv_first": (6.0, 3.0)},
        "3": {"cmp_pro_first": (5.0, 5.0), "cmp_priv_first": (6.0, 4.0)}}
SEP = {"1": (6.0, 4.0), "3": (5.5, 5.0)}
NEUTRAL = {"1": [(6.4, 4.1), (5.8, 4.3), (6.1, 3.7)],
           "3": [(5.9, 5.0), (5.2, 5.1), (5.6, 4.6)]}
STATED = {"1": (7.0, 3.6), "3": (6.2, 4.5)}
CTRL_CMPN = [0.0, 1.0, -1.0, 0.0, 2.0, 0.0, -1.0, 1.0, 0.0, 0.0]
CTRL_PRO = [5, 6, 5, 4, 5, 6, 5, 5, 4, 6]
CTRL_PRIV = [5, 5, 6, 4, 5, 5, 5, 6, 5, 5]


def load_duty(con):
    duty_db.populate_pairs(con, "banks/duty_texts.jsonl", "0" * 64)
    duty_db.populate_model_arm(con, off={"example/api-model": ARM}, on={})
    for p, orders in CMPN.items():
        for cond, (a, b) in orders.items():
            duty_cell(con, f"{p}|{cond}", p, "both", cond, "CMPN", sp=a, sv=b)
    for p, (a, b) in SEP.items():
        duty_cell(con, f"{p}|sep|pro", p, "pro", "bare", "BLAME", ev=a)
        duty_cell(con, f"{p}|sep|priv", p, "priv", "bare", "BLAME", ev=b)
    for p, rows in NEUTRAL.items():
        for k, (a, b) in enumerate(rows, 1):
            duty_cell(con, f"{p}|n{k}|pro", p, "pro", f"neutral_{k}", "BLAME", ev=a)
            duty_cell(con, f"{p}|n{k}|priv", p, "priv", f"neutral_{k}", "BLAME", ev=b)
    for p, (a, b) in STATED.items():
        duty_cell(con, f"{p}|st|pro", p, "pro", "stated", "BLAME", ev=a)
        duty_cell(con, f"{p}|st|priv", p, "priv", "stated", "BLAME", ev=b)
    duty_cell(con, "6|cmpn", "6", "both", "cmp_pro_first", "CMPN", sp=5.0, sv=5.0,
              draws=[(None, g) for g in CTRL_CMPN])
    duty_cell(con, "6|sep|pro", "6", "pro", "bare", "BLAME", ev=RD.mean(CTRL_PRO),
              draws=[(float(v), None) for v in CTRL_PRO])
    duty_cell(con, "6|sep|priv", "6", "priv", "bare", "BLAME", ev=RD.mean(CTRL_PRIV),
              draws=[(float(v), None) for v in CTRL_PRIV])


def test_duty_views():
    con = fresh()
    load_excuse(con)
    load_duty(con)
    # hand-computed references
    stated = {p: statistics.mean(a - b for a, b in o.values()) for p, o in CMPN.items()}
    spread = {p: max(a - b for a, b in o.values()) - min(a - b for a, b in o.values())
              for p, o in CMPN.items()}
    zero = {p: int(any(a == b for a, b in o.values())) for p, o in CMPN.items()}
    sep = {p: a - b for p, (a, b) in SEP.items()}
    ctrl_gaps = CTRL_CMPN + [a - b for a, b in zip(CTRL_PRO, CTRL_PRIV)]
    draw_floor = max(duty_db.Z * statistics.pstdev(ctrl_gaps), duty_db.DRAW_FLOOR_MIN)
    devs = [(a - b) - sep[p] for p, rows in NEUTRAL.items() for a, b in rows]
    sentence = duty_db.Z * statistics.pstdev(devs)
    floor_used = max(sentence, duty_db.DRAW_FLOOR_MIN)
    baseline = {p: statistics.mean(a - b for a, b in rows) for p, rows in NEUTRAL.items()}

    for p in CMPN:
        g, sp_, ez = con.execute("SELECT stated_gap, order_spread, "
                                 "exact_zero_in_an_order FROM duty_stated_gap WHERE "
                                 "arm=? AND pair_id=?", (ARM, p)).fetchone()
        check(f"duty: pair {p} stated gap is the two-order mean", close(g, stated[p]),
              f"{g} vs {stated[p]}")
        check(f"duty: pair {p} order spread", close(sp_, spread[p]))
        check(f"duty: pair {p} exact-zero flag", ez == zero[p], f"{ez} vs {zero[p]}")
        s = con.execute("SELECT separate_gap FROM duty_separate_gap WHERE arm=? AND "
                        "pair_id=?", (ARM, p)).fetchone()[0]
        check(f"duty: pair {p} separate gap", close(s, sep[p]))
        b = con.execute("SELECT neutral_baseline FROM duty_neutral_baseline WHERE "
                        "arm=? AND pair_id=?", (ARM, p)).fetchone()[0]
        check(f"duty: pair {p} neutral baseline is the mean of three", close(b, baseline[p]))
    df = con.execute("SELECT draw_floor FROM duty_floor_draw WHERE arm=?",
                     (ARM,)).fetchone()[0]
    check("duty: draw floor is max(2.33 x pstdev(control gaps), 0.30)",
          close(df, draw_floor), f"{df} vs {draw_floor}")
    sf, fu = con.execute("SELECT floor_sentence, floor_used FROM duty_floor_used "
                         "WHERE arm=?", (ARM,)).fetchone()
    check("duty: sentence floor is 2.33 x pstdev(neutral deviations)",
          close(sf, sentence), f"{sf} vs {sentence}")
    check("duty: floor_used is max(sentence floor, 0.30)", close(fu, floor_used))
    rows = con.execute("SELECT pair_id, stated_gap, floor_used, signal_to_noise FROM "
                       "duty_signal_to_noise WHERE arm=? AND stated_gap IS NOT NULL",
                       (ARM,)).fetchall()
    check("duty: the arm enters the duty figure with both pairs read",
          sorted(r[0] for r in rows) == ["1", "3"], str(rows))
    for p, g, f, s in rows:
        check(f"duty: pair {p} signal_to_noise = stated_gap / floor_used",
              close(s, stated[p] / floor_used), f"{s} vs {stated[p] / floor_used}")
    med = con.execute("SELECT median_signal_to_noise, n_pairs FROM "
                      "duty_signal_to_noise_arm WHERE arm=?", (ARM,)).fetchone()
    check("duty: per-arm median signal_to_noise",
          close(med[0], statistics.median(stated[p] / floor_used for p in stated))
          and med[1] == 2)
    leak = con.execute("SELECT COUNT(*) FROM duty_reported WHERE pair_id='6'"
                       ).fetchone()[0]
    check("duty: the control never reaches duty_reported", leak == 0)
    for name, ok, n, det in duty_db.tests(con):
        check(f"synthetic: {name}", ok == n, f"{ok} of {n} {det}")


# ── (c) SQL EQUIVALENCE WITH THE SOURCE REPOSITORY ────────────────────────────
def _strip(sql):
    return " ".join(re.sub(r"--[^\n]*", "", sql).split()).rstrip(";").strip()


def _literal(path, name):
    """A module-level literal from a source file, read with ast (never executed)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not in {path}")


def test_source_equivalence():
    src = os.environ.get("DUE_CARE_SOURCE_REPO")
    if not src:
        print("SKIP  SQL equivalence with the source repository "
              "(set DUE_CARE_SOURCE_REPO to run it)")
        return
    scripts = os.path.join(src, "scripts")
    pairs = [("results_derive VIEWS", _literal(os.path.join(scripts, "results_derive.py"),
                                               "VIEWS"), RD.VIEWS),
             ("duty_db VIEWS", _literal(os.path.join(scripts, "duty_db.py"), "VIEWS"),
              duty_db.VIEWS)]
    for label, theirs, ours in pairs:
        check(f"source: {label} has the same view names", set(theirs) == set(ours),
              f"{sorted(set(theirs) ^ set(ours))}")
        diff = [k for k in ours if k in theirs and _strip(theirs[k]) != _strip(ours[k])]
        check(f"source: {label} SQL identical after stripping comments", not diff,
              ", ".join(diff))
    check("source: duty_db SCHEMA identical after stripping comments",
          _strip(_literal(os.path.join(scripts, "duty_db.py"), "SCHEMA"))
          == _strip(duty_db.SCHEMA))
    full = _literal(os.path.join(scripts, "build_results_db.py"), "SCHEMA")
    base = "".join(part for part in re.split(r"(?=CREATE TABLE )", _strip_keep(full))
                   if re.match(r"CREATE TABLE (artefacts|banks|cells|orderings)\(", part))
    check("source: BASE_SCHEMA identical to the source's four base tables",
          _strip(base) == _strip(schema.BASE_SCHEMA))


def _strip_keep(sql):
    return re.sub(r"--[^\n]*", "", sql)


def main():
    test_empty_database()
    test_excuse_views()
    test_duty_views()
    test_source_equivalence()
    n_fail = sum(1 for _, ok in _RESULTS if not ok)
    print(f"\n{len(_RESULTS) - n_fail} of {len(_RESULTS)} checks passed")
    return 1 if n_fail else 0


def test_all_checks_pass():
    """pytest entry point."""
    _RESULTS.clear()
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
