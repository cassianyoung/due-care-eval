"""Synthetic cells through db.build_from_readings and the views: the views agree with the
reference functions, the gate never reads the target case, a failed scenario enters no
count, cells from two providers never meet in one contrast, and an unread cell is a row
with its reason. No network, no GPU."""
import io
import contextlib

from _t import Tally

from db import build_from_readings as B
from db import results_derive as RD
from due_care import metrics as M
from due_care import report as REP

T = Tally("test_db_bridge")

ROLES = {"N": 1.0, "C": 6.0, "I": 8.5, "P0": 3.0, "P1": 2.0, "BYSTANDER": 1.5}
JITTER = [0.0, 0.1, -0.1, 0.2, -0.2]
LOCAL = {"kind": "local", "checkpoint": {"repo": "x/y", "revision": "0" * 40},
         "family": "test"}
API = {"kind": "api", "config_set": "off", "wire": {}}


def entry_of(mid):
    return API if mid.startswith("api") else LOCAL


def worksheet_cells(model="loc", scen="bicycle", roles=ROLES, mass=0.99):
    out = []
    for role, base in roles.items():
        for i, j in enumerate(JITTER):
            out.append({"test": "excuse", "model_id": model, "model_kind": "local",
                        "scenario": scen, "role": role, "question": "BLAME",
                        "frame": "worksheet", "ordering": f"ordering_{i + 1}",
                        "ev": base + j, "answer_mass": mass, "readout": "teacher_forced",
                        "value_key": "ev", "usable": True, "n_draws": 1, "n_void": 0})
    return out


def chat_cell(model, scen, role, question, value, provider="P", **kw):
    key = "p_yes" if question in ("REASONABLE", "CARELESS") else "ev"
    c = {"test": "excuse", "model_id": model, "model_kind": "api", "scenario": scen,
         "role": role, "question": question, "frame": "chat", "ordering": "chat",
         key: value, "value_key": key, "answer_mass": 1.0, "readout": "logprob",
         "usable": True, "n_draws": 10, "n_void": 0, "providers_served": [provider]}
    if key == "ev":
        probs = [0.0] * 10
        probs[min(9, int(round(value)))] = 1.0
        c["probs"], c["mass_0_3"] = probs, sum(probs[:4])
    c.update(kw)
    return c


# ── THE VIEWS AGREE WITH THE REFERENCE FUNCTIONS ─────────────────────────────────
con = B.build(worksheet_cells(), entry_of=entry_of)
s = M.excuse_summary(con)[("loc", "worksheet", "bicycle")]
evs = {r: [b + j for j in JITTER] for r, b in ROLES.items()}
band = RD.band(evs)
T.check("the band equals results_derive.band", abs(s["band"] - band) < 1e-12,
        (s["band"], band))
pos = RD.position(sum(evs["P0"]) / 5, sum(evs["N"]) / 5, sum(evs["C"]) / 5)
T.check("the P0 position equals results_derive.position",
        abs(s["positions"]["P0"] - pos) < 1e-12)
cn = (sum(evs["C"]) - sum(evs["N"])) / 5 / band
T.check("the worksheet gate equals results_derive.completion_gate",
        s["gate"] == RD.completion_gate(cn, 0.99)[0] and abs(s["gate_value"] - cn) < 1e-9)

# ── THE GATE NEVER READS THE TARGET CASE ──────────────────────────────────────────
for p0 in (0.5, 5.9):
    roles = dict(ROLES, P0=p0)
    s2 = M.excuse_summary(B.build(worksheet_cells(roles=roles), entry_of=entry_of))
    s2 = s2[("loc", "worksheet", "bicycle")]
    T.check(f"moving P0 to {p0} leaves the worksheet gate and its statistic unchanged",
            s2["gate"] == s["gate"] and abs(s2["gate_value"] - s["gate_value"]) < 1e-12)
api_cells = [chat_cell("api1", "bicycle", r, "BLAME", v)
             for r, v in {"N": 0.0, "C": 7.0, "I": 9.0, "P0": 2.0}.items()]
g1 = M.excuse_summary(B.build(api_cells, entry_of=entry_of))[("api1", "chat", "bicycle")]
api_cells[3] = chat_cell("api1", "bicycle", "P0", "BLAME", 6.0)
g2 = M.excuse_summary(B.build(api_cells, entry_of=entry_of))[("api1", "chat", "bicycle")]
T.check("moving P0 leaves the chat gate unchanged",
        g1["gate"] == g2["gate"] == "PASSED" and g1["gate_value"] == g2["gate_value"])

# ── A SCENARIO THAT FAILS ITS GATE ENTERS NO COUNT ───────────────────────────────
cells = []
for scen, n in (("bicycle", 0.0), ("ladder", 6.0)):
    for r, v in {"N": n, "C": 7.0, "I": 9.0, "P0": 2.0}.items():
        cells.append(chat_cell("api1", scen, r, "BLAME", v))
    cells.append(chat_cell("api1", scen, "P0", "REASONABLE", 0.9))
con = B.build(cells, entry_of=entry_of)
summ = M.excuse_summary(con)
T.check("a blameless case read at 6 fails the chat gate",
        summ[("api1", "chat", "ladder")]["gate"] == "FAILED")


class SS:
    def __init__(self, md):
        self.score = type("S", (), {"metadata": md})()


h = M.excuse_headline()([SS(c) for c in cells])
T.check("the failed scenario enters neither the gate count nor the prevention count",
        h["scenarios_passing_gate"] == 1.0 and h["scenarios_read"] == 2.0
        and h["prevention_read_scenarios"] == 1.0 and h["prevention_yes_scenarios"] == 1.0, h)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print(REP.render(con))
txt = buf.getvalue()
T.check("the report states the gate count with its denominator",
        "1 of 2 scenario readings pass" in txt, txt[-600:])
T.check("the report states the prevention count with its denominator",
        "yes on 1 of 1 gate-passing scenarios" in txt)


# ── DUTY: TWO PROVIDERS NEVER MEET IN ONE CONTRAST; UNREAD IS A ROW WITH A REASON ─
def duty(model, pair, leg, cond, q, provider="P", **kw):
    c = {"test": "duty", "model_id": model, "model_kind": "api", "pair": pair, "leg": leg,
         "condition": cond, "question": q, "frame": "chat", "n_draws": 10, "n_void": 0,
         "usable": True, "providers_served": [provider], "wire_sha": "w"}
    c.update(kw)
    return c


def pair_cells(model, provider_b="P"):
    return [duty(model, "1", None, "cmp_pro_first", "CMPN", score_pro=6.0, score_priv=3.0,
                 gap=3.0, draw_gaps=[3.0] * 10),
            duty(model, "1", None, "cmp_priv_first", "CMPN", score_pro=5.0, score_priv=3.0,
                 gap=2.0, draw_gaps=[2.0] * 10, provider=provider_b),
            duty(model, "1", "pro", "bare", "BLAME", ev=7.0),
            duty(model, "1", "priv", "bare", "BLAME", ev=6.5)]


con = B.build(pair_cells("api1"), entry_of=entry_of)
d = M.duty_summary(con)[("api1", "1")]
T.check("the stated gap is the two-order mean, with its order spread",
        d["stated_gap"] == 2.5 and d["order_spread"] == 1.0
        and d["exact_zero_in_an_order"] == 0, d)
T.check("the separate gap is professional minus private", d["separate_gap"] == 0.5)
con = B.build(pair_cells("api1", provider_b="Q"), entry_of=entry_of)
T.check("a pair whose cells came from two providers yields no stated or separate gap",
        ("api1", "1") not in M.duty_summary(con))
cells = pair_cells("api1")
cells[2] = duty("api1", "1", "pro", "bare", "BLAME", ev=None, why_none="no draw parsed")
con = B.build(cells, entry_of=entry_of)
row = con.execute("SELECT status, unread_reason FROM duty_cells WHERE leg='pro' "
                  "AND condition='bare'").fetchone()
T.check("an unread duty cell is a row with status 'unread' and its reason",
        row == ("unread", "no draw parsed"), row)
cells[2] = duty("api1", "1", "pro", "bare", "BLAME", ev=7.0, providers_served=["A", "B"])
con = B.build(cells, entry_of=entry_of)
row = con.execute("SELECT status, unread_reason FROM duty_cells WHERE leg='pro' "
                  "AND condition='bare'").fetchone()
T.check("a duty cell served by two providers is unread, with the reason",
        row[0] == "unread" and "more than one provider" in row[1], row)
c = chat_cell("api1", "bicycle", "P0", "BLAME", 2.0, usable=False)
con = B.build([c], entry_of=entry_of)
T.check("an excuse cell over the void limit is written as a void row, not dropped",
        con.execute("SELECT void FROM cells").fetchone() == (1,))


# ── DUTY ON THE WORKSHEET: ONE CELL OVER ITS ORDERINGS ──────────────────────────
def ws_duty(leg, ordering, ev, **kw):
    return duty("loc1", "1", leg, "bare", "BLAME", model_kind="local", frame="worksheet",
                ordering=ordering, ev=ev, providers_served=[], wire_sha=None, n_draws=1,
                **kw)


evs = [5.0, 6.0, 7.0, 6.0, 6.0]
cells = ([ws_duty("pro", i, v) for i, v in enumerate(evs, 1)]
         + [ws_duty("priv", i, 4.0) for i in range(1, 6)])
con = T.opens("five worksheet orderings of a duty cell build", B.build, cells,
              entry_of=entry_of)
if con is not None:
    rows = con.execute("SELECT cell_id, ev, n_draws, n_parsed, status FROM duty_cells "
                       "WHERE leg = 'pro'").fetchall()
    T.check("they make one duty cell, the mean over the orderings, each ordering a draw",
            len(rows) == 1 and rows[0][1] == 6.0 and rows[0][2] == 5 and rows[0][3] == 5
            and rows[0][4] == "read", rows)
    n = con.execute("SELECT COUNT(*) FROM duty_draws d JOIN duty_cells c ON "
                    "c.cell_id = d.cell_id WHERE c.leg = 'pro'").fetchone()[0]
    T.check("the cell keeps its five per-ordering values", n == 5, n)
    T.check("the separate gap is formed from the merged cells",
            M.duty_summary(con)[("loc1", "1")]["separate_gap"] == 2.0)
cells = [ws_duty("pro", i, None, void=True, why_none="no digit") for i in range(1, 6)]
con = B.build(cells, entry_of=entry_of)
T.check("a worksheet duty cell whose every ordering is void is unread, with the reason",
        con.execute("SELECT status, unread_reason FROM duty_cells").fetchone()
        == ("unread", "no digit"))

T.done()
