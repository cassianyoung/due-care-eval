"""Per-model aggregation, computed by the database views.

The scored cells are written into a fresh database (`db.build_from_readings`), and every
quantity below is read from a view defined in `db/results_derive.py` or `db/duty_db.py`.
Nothing here recomputes a position, band, gate, gap or floor.

Rules applied when summarising:
* A scenario whose gate fails enters no count for that model, judgement questions
  included. Its values are still reported, marked by the gate.
* The gate never considers the target case: it is computed from N, C, I and the answer
  mass, as the views define it.
* Chat cells read through an API carry their spread over draws (`draw_sd`). It is
  descriptive only, not a band, and no view reads it.
* Every count carries its denominator.
"""
import math

from inspect_ai.scorer import SampleScore, metric

from db import build_from_readings as B

DEPLOYED_SCALE = ("bicycle", "ladder", "prawns_home", "prawns_restaurant", "scaffolding",
                  "cheque")


def _q(con, sql, *a):
    cur = con.execute(sql, a)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def excuse_summary(con):
    """{(model, delivery, scenario): row} with positions, band, gate and judgements."""
    out = {}
    for g in _q(con, "SELECT g.artefact_id, a.arm, a.delivery, g.family, g.verdict, "
                "g.stat_name, g.stat_value, g.answer_mass, g.i_mean, g.n_mean "
                "FROM cell_gate g JOIN artefacts a ON a.id = g.artefact_id "
                "WHERE g.question = 'BLAME' AND a.path NOT LIKE '%:duty'"):
        key = (g["arm"], "worksheet" if g["delivery"] == "completion" else "chat",
               g["family"])
        row = {"gate": g["verdict"], "gate_statistic": g["stat_name"],
               "gate_value": g["stat_value"], "answer_mass": g["answer_mass"],
               "positions": {}, "means": {}, "draw_sd": {}}
        for p in _q(con, "SELECT role, ev_mean, n_obs, position, n_mean, c_mean, i_mean "
                    "FROM cell_positions WHERE artefact_id=? AND family IS ? "
                    "AND question='BLAME'", g["artefact_id"], g["family"]):
            row["positions"][p["role"]] = p["position"]
            row["means"][p["role"]] = p["ev_mean"]
            row["N"], row["C"], row["I"] = p["n_mean"], p["c_mean"], p["i_mean"]
        for s in _q(con, "SELECT role, ev_sd FROM cells WHERE artefact_id=? AND family IS ? "
                    "AND question='BLAME' AND frame='chat'", g["artefact_id"], g["family"]):
            row["draw_sd"][s["role"]] = s["ev_sd"]
        b = _q(con, "SELECT band, n_roles_in_band FROM band WHERE artefact_id=? "
               "AND family IS ? AND question='BLAME'", g["artefact_id"], g["family"])
        row["band"] = b[0]["band"] if b else None
        inv = _q(con, "SELECT gap_digits, gap_bands, gap_position_units, clears_2_bands "
                 "FROM involvement WHERE artefact_id=? AND family IS ? AND question='BLAME'",
                 g["artefact_id"], g["family"])
        row["bystander"] = inv[0] if inv else None
        eff = _q(con, "SELECT p0_minus_n_bands, effect_present FROM effect_present "
                 "WHERE artefact_id=? AND family IS ? AND question='BLAME'",
                 g["artefact_id"], g["family"])
        row["effect"] = eff[0] if eff else None
        for q in ("REASONABLE", "CARELESS"):
            j = _q(con, "SELECT c.p_yes, c.answer_mass, c.void FROM cells c "
                   "JOIN artefacts a2 ON a2.id = c.artefact_id "
                   "WHERE a2.arm=? AND a2.delivery=? AND c.family IS ? AND c.role='P0' "
                   "AND c.question=?", g["arm"], g["delivery"], g["family"], q)
            row[f"P0_{q.lower()}_p_yes"] = j[0]["p_yes"] if j and not j[0]["void"] else None
        out[key] = row
    return out


def duty_summary(con):
    """{(model, pair): row} from the duty views."""
    out = {}
    for r in _q(con, "SELECT * FROM duty_stated_gap"):
        out.setdefault((r["arm"], r["pair_id"]), {}).update(
            {k: r[k] for k in ("gap_pro_first", "gap_priv_first", "stated_gap",
                               "order_spread", "exact_zero_in_an_order", "n_orders")})
    for r in _q(con, "SELECT * FROM duty_separate_gap"):
        out.setdefault((r["arm"], r["pair_id"]), {}).update(
            {k: r[k] for k in ("ev_pro", "ev_priv", "separate_gap", "headroom_above_pro")})
    for r in _q(con, "SELECT * FROM duty_neutral_baseline"):
        out.setdefault((r["arm"], r["pair_id"]), {}).update(
            {"neutral_baseline": r["neutral_baseline"], "n_neutrals": r["n_neutrals"],
             "neutral_refused_reason": r["refused_reason"]})
    for r in _q(con, "SELECT * FROM duty_remedy_gap WHERE condition='stated'"):
        out.setdefault((r["arm"], r["pair_id"]), {})["stated_condition_gap"] = r["gap"]
    floors = {r["arm"]: r for r in _q(con, "SELECT * FROM duty_floor_used")}
    overview = {(r["arm"], r["pair_id"]): r for r in _q(con, "SELECT * FROM duty_overview")}
    snr = {(r["arm"], r["pair_id"]): r for r in _q(con, "SELECT * FROM duty_signal_to_noise")}
    for (arm, pid), row in out.items():
        f = floors.get(arm) or {}
        row["floor_used"] = f.get("floor_used")
        row["floor_used_source"] = f.get("floor_used_source")
        x = overview.get((arm, pid)) or {}
        for k in ("stated_condition_vs_stated", "convention_stated", "low_resolution",
                  "verdict", "share"):
            row[k] = x.get(k)
        s = snr.get((arm, pid)) or {}
        row["signal_to_noise"] = s.get("signal_to_noise")
    return out


def _finite(xs):
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def _cells(scores):
    """Cell metadata from Score or SampleScore objects."""
    out = []
    for s in scores:
        md = getattr(getattr(s, "score", s), "metadata", None)
        if md and md.get("model_id"):
            out.append(dict(md))
    return out


@metric
def excuse_headline():
    """Target-case position per scenario, the gate, and gate-passing counts."""
    def compute(scores: list[SampleScore]):
        con = B.build(_cells(scores))
        summ = excuse_summary(con)
        out = {}
        passing = [k for k, v in summ.items() if v["gate"] in ("PASSED", "LICENSED")]
        for (arm, frame, scen), v in summ.items():
            p0 = v["positions"].get("P0")
            out[f"{frame}/{scen}/P0_position"] = float("nan") if p0 is None else p0
            out[f"{frame}/{scen}/gate_passed"] = 1.0 if (arm, frame, scen) in passing else 0.0
        out["scenarios_passing_gate"] = float(len(passing))
        out["scenarios_read"] = float(len(summ))
        prevent = _finite([summ[k].get("P0_reasonable_p_yes") for k in passing])
        out["prevention_yes_scenarios"] = float(sum(1 for p in prevent if p > 0.5))
        out["prevention_read_scenarios"] = float(len(prevent))
        return out
    return compute


@metric
def duty_headline():
    """Stated and separate gap per pair."""
    def compute(scores: list[SampleScore]):
        con = B.build(_cells(scores))
        out = {}
        for (arm, pid), v in duty_summary(con).items():
            for k in ("stated_gap", "separate_gap", "stated_condition_gap"):
                x = v.get(k)
                out[f"{pid}/{k}"] = float("nan") if x is None else x
        return out
    return compute
