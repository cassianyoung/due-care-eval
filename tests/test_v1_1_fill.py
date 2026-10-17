"""Bank v1.1, the rest of the fill: the prawns reference case and duty CAUSE question,
SUPPLIER and cell-level selection on the excuse test, cell selection on the duty test,
Gemini 3.1 Pro's excuse-test wires, and the database views the new cells must reach and
must not reach. Each refusal is paired with a check that the same gate opens. No
network, no GPU.
"""
import hashlib
import json
import os
import shutil
import tempfile

from _t import ROOT, Tally

from db import build_from_readings as B
from due_care import datasets as D
from due_care import model_table as MT
from due_care.tasks.duty_test import duty as duty_task
from due_care.tasks.excuse_test import excuse as excuse_task
from harness import api as API
from harness import frames as F

T = Tally("test_v1_1_fill")


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


v10 = F.load_jsonl("duty_texts.jsonl")
sep = {(r["pair"], r["leg"]): r["text"] for r in v10 if r["kind"] == "separate"}

# ── 1. THE REFERENCE CASE ──────────────────────────────────────────────────────────────
# pair 5's two banked reference cases, from the study's duty bank, which the literal
# rule must reproduce
PAIR5_BANKED = {"pro": "717efd4cbfc616a1a6cc250f0feb275c6a4bb519b87b3e5b125643a7336a6de0",
                "priv": "5e76a5c0d2fea766af95f812cac671334625de9510e4b92ad639bbe88d5c7557"}
PRAWNS_REGISTERED = {
    "pro": "4646bca7d34c03d58a72fe1ec9dff887ac7598584635bef82871c882e50f6a2b",
    "priv": "46b58c2c0a0695e29203e29d279ed23e6dd7ce51292504d413d14ffce7aede26"}


def reference_case(pair, leg):
    other = "priv" if leg == "pro" else "pro"
    return ("Reference case (for comparison only): "
            + sep[(pair, other)].replace("Alaric", "Another man", 1)
            + " Case to rate: " + sep[(pair, leg)])


T.check("the literal rule reproduces pair 5's two banked reference cases byte for byte",
        all(sha(reference_case("5", leg)) == PAIR5_BANKED[leg] for leg in ("pro", "priv")))
comp = {r["leg"]: r for r in F.load_jsonl("duty_texts_v1_1.jsonl") if r["kind"] == "comparand"}
T.check("the prawns reference cases are that rule, with their pre-registered sha256",
        all(comp[leg]["text"] == reference_case("prawns", leg)
            and comp[leg]["sha256"] == PRAWNS_REGISTERED[leg] for leg in ("pro", "priv")))
rc = D.duty_dataset("chat", ("prawns",), conditions=("comparand",))
T.check("conditions=comparand selects the 2 reference-case samples, one per leg",
        sorted(s.id for s in rc) == ["prawns|priv|comparand|chat",
                                     "prawns|pro|comparand|chat"])
qs = F.load_json("excuse_questions.json")
T.check("each asks the banked BLAME question of the case to rate, with the reference "
        "case first and only the case to rate naming Alaric",
        all(s.input == qs["BLAME"]["body"].format(item=comp[s.metadata["leg"]]["text"],
                                                   agent="Alaric")
            and s.metadata["question"] == "BLAME"
            and s.input.index("Reference case") < s.input.index("Case to rate:")
            < s.input.index("Alaric") and s.input.count("Alaric") == 2
            for s in rc))
T.check("the reference case is not read on the worksheet frame",
        len(D.duty_dataset("worksheet", ("prawns",), conditions=("comparand",))) == 0)
T.raises("the reference case under bank 1.0 is refused as an empty selection",
         ValueError, duty_task, pairs="prawns", conditions="comparand", bank="1.0")

# ── 2. THE DUTY CAUSE QUESTION ─────────────────────────────────────────────────────────
cq = F.load_duty_cause("1.1")
T.check("v1.1 holds one duty CAUSE question, for prawns, hashing to its registered sha256",
        list(cq) == ["prawns"] and cq["prawns"]["sha256"] == sha(cq["prawns"]["body"])
        == "1d11458b5b8bcaf8590ed3a6c35565b5a76a27921627eb3baee454616d06fc05")
T.check("v1.0 holds no duty CAUSE question", F.load_duty_cause("1.0") == {})
ca = D.duty_dataset("chat", ("prawns",), conditions=("cause",))
T.check("conditions=cause selects the 2 CAUSE samples, one per leg",
        sorted(s.id for s in ca) == ["prawns|priv|cause|chat", "prawns|pro|cause|chat"])
T.check("both legs ask the same question about the same harm, after their own separate "
        "case, under the digit system message",
        all(s.input == cq["prawns"]["body"].format(item=sep[("prawns", s.metadata["leg"])],
                                                    agent="Alaric")
            and s.metadata["question"] == "CAUSE" and s.metadata["condition"] == "bare"
            and s.metadata["system"] == qs["BLAME"]["system"] for s in ca)
        and len({s.input.split("How much")[1] for s in ca}) == 1)
T.check("the harm noun ('illness') occurs in neither leg's case",
        all("illness" not in sep[("prawns", leg)] for leg in ("pro", "priv")))
T.check("CAUSE is not read on the worksheet frame",
        len(D.duty_dataset("worksheet", ("prawns",), conditions=("cause",))) == 0)
T.raises("CAUSE under bank 1.0 is refused as an empty selection", ValueError,
         duty_task, pairs="prawns", conditions="cause", bank="1.0")


def tamper_cause(bank_dir):
    p = os.path.join(bank_dir, "duty_questions_v1_1.json")
    d = json.load(open(p))
    d["cause"]["prawns"]["body"] = d["cause"]["prawns"]["body"].replace("illness", "harm")
    json.dump(d, open(p, "w"))
    mp = os.path.join(bank_dir, "MANIFEST.json")
    m = json.load(open(mp))
    m["files"]["duty_questions_v1_1.json"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(m, open(mp, "w"))


def with_banks(edit, fn):
    tmp = tempfile.mkdtemp()
    try:
        shutil.copytree(F.BANKS, os.path.join(tmp, "banks"))
        edit(os.path.join(tmp, "banks"))
        old, F.BANKS = F.BANKS, os.path.join(tmp, "banks")
        try:
            return fn()
        finally:
            F.BANKS = old
    finally:
        shutil.rmtree(tmp)


T.raises("a tampered CAUSE body is refused, even with the manifest updated to match",
         F.BankError, with_banks, tamper_cause, lambda: F.load_duty_cause("1.1"))
T.check("the same copy, untampered, opens",
        with_banks(lambda b: None, lambda: list(F.load_duty_cause("1.1"))) == ["prawns"])

# ── 3. THE DATABASE: NEW CELLS REACH THEIR VIEWS AND NOTHING ELSE ──────────────────────


def duty_cell(pair, leg, cond, ev, q="BLAME"):
    return {"test": "duty", "model_id": "api1", "model_kind": "api", "pair": pair,
            "leg": leg, "condition": cond, "question": q, "frame": "chat", "n_draws": 10,
            "n_void": 0, "usable": True, "providers_served": ["P"], "wire_sha": "w",
            "ev": ev, "ev_sd": 0.1, "bank_version": "1.1"}


cells = [duty_cell("prawns", "pro", "bare", 6.0), duty_cell("prawns", "priv", "bare", 5.0),
         duty_cell("prawns", "pro", "bare", 9.0, "CAUSE"),
         duty_cell("prawns", "priv", "bare", 1.0, "CAUSE"),
         duty_cell("prawns", "pro", "comparand", 3.0),
         duty_cell("prawns", "priv", "comparand", 8.0)]
con = T.opens("a build with prawns CAUSE and reference-case cells opens", B.build, cells,
              entry_of=lambda mid: {"kind": "api", "config_set": "off", "wire": {}})
if con is not None:
    sg = con.execute("SELECT separate_gap, n_legs FROM duty_separate_gap WHERE "
                     "pair_id='prawns'").fetchone()
    T.check("the separate gap is the bare BLAME cells' alone (1.0 over 2 legs): neither "
            "CAUSE (+8) nor the reference case (-5) enters it", sg == (1.0, 2), sg)
    cv = con.execute("SELECT cause_pro, cause_priv FROM duty_cause WHERE "
                     "pair_id='prawns'").fetchone()
    T.check("the CAUSE cells reach duty_cause (9.0 and 1.0)", cv == (9.0, 1.0), cv)
    nt = con.execute("SELECT COUNT(*), MIN(bank) FROM duty_texts WHERE pair_id='prawns' "
                     "AND kind='comparand'").fetchone()
    T.check("the two reference-case texts are in the database, from the v1.1 file",
            nt == (2, "banks/duty_texts_v1_1.jsonl"), nt)

# a void full-text prompt leaves prawns' stated gap MISSING, never the summary's
def cmp_cell(cond, pro, priv, **kw):
    c = duty_cell("prawns", None, cond, None, "CMPN")
    g = None if pro is None else pro - priv
    c.update({"leg": None, "score_pro": pro, "score_priv": priv, "gap": g,
              "draw_gaps": [] if g is None else [g] * 10})
    c.update(kw)
    return c


base = [duty_cell("prawns", "pro", "bare", 6.0), duty_cell("prawns", "priv", "bare", 5.0),
        cmp_cell("cmp_pro_first", 7.0, 4.0), cmp_cell("cmp_priv_first", 6.0, 4.0)]
void_full = [cmp_cell(c, None, None, why_none="no draw parsed", gap=None)
             for c in ("cmp_full_pro_first", "cmp_full_priv_first")]
one_order = [cmp_cell("cmp_full_pro_first", 8.0, 5.0),
             cmp_cell("cmp_full_priv_first", None, None, why_none="no draw parsed",
                      gap=None)]
both_full = [cmp_cell("cmp_full_pro_first", 8.0, 5.0),
             cmp_cell("cmp_full_priv_first", 9.0, 5.0)]
# THE GATE OPENS: both orders of the full text read give its two-order mean (3.5). THE
# GATE REFUSES: on a v1.1 bank, where prawns has a full-text prompt, an arm with no
# full-text cell, void ones, or one order read has NO stated gap, never the summary's
# (Cassian, 2026-10-17). Plus the v1.0 case below, where the summary is the definition.
for label, extra, want in (
        ("both full-text orders", both_full, [(3.5, "full_text")]),
        ("no full-text cell", [], []),
        ("void full-text cells", void_full, []),
        ("one full-text order read", one_order, [(None, "full_text")])):
    con = T.opens(f"a build with the summary read and {label} opens", B.build,
                  base + extra,
                  entry_of=lambda mid: {"kind": "api", "config_set": "off", "wire": {}})
    if con is None:
        continue
    got = con.execute("SELECT stated_gap, stated_prompt FROM duty_stated_gap WHERE "
                      "pair_id='prawns'").fetchall()
    T.check(f"{label}: the stated gap is {want or 'MISSING, not the summary'}",
            got == want, got)
    keep = con.execute("SELECT stated_gap FROM duty_stated_gap_summary WHERE "
                       "pair_id='prawns'").fetchall()
    T.check(f"{label}: the summary reading is still in its own view (2.5)",
            keep == [(2.5,)], keep)
    if extra is void_full:
        why = con.execute("SELECT share, status, missing_reason FROM duty_overview WHERE "
                          "pair_id='prawns'").fetchone()
        T.check("void full-text cells: no share, and the reason is the full-text read's",
                why is not None and why[0] is None and "no draw parsed" in (why[2] or ""),
                why)

# a build from v1.0 readings holds only the v1.0 texts, where the summary IS prawns'
# side-by-side prompt, so its stated gap is the summary's
base10 = [dict(c, bank_version="1.0") for c in base]
con = T.opens("a build from v1.0 readings opens", B.build, base10,
              entry_of=lambda mid: {"kind": "api", "config_set": "off", "wire": {}})
if con is not None:
    got = con.execute("SELECT stated_gap, stated_prompt FROM duty_stated_gap WHERE "
                      "pair_id='prawns'").fetchall()
    T.check("v1.0 readings: the stated gap is the summary's (2.5), the only prompt v1.0 has",
            got == [(2.5, "summary")], got)

# ── 4. THE EXCUSE TEST: ROLES, SUPPLIER AND CELLS ──────────────────────────────────────
T.check("by default the excuse test reads the ten roles (216 chat samples, no SUPPLIER)",
        len(excuse_task().dataset) == 216
        and not any(s.metadata["role"] == "SUPPLIER" for s in excuse_task().dataset))
sup = excuse_task(roles="SUPPLIER").dataset
T.check("-T roles=SUPPLIER reads the 8 SUPPLIER cells (two food scenarios x 4 questions)",
        len(sup) == 8 and {s.metadata["scenario"] for s in sup}
        == {"prawns_home", "prawns_restaurant"})
T.check("-T roles=all reads 224 chat samples", len(excuse_task(roles="all").dataset) == 224)
T.check("-T roles=P0,P1 on the bicycle reads 8 samples",
        len(excuse_task(scenarios="bicycle", roles="P0,P1").dataset) == 8)
T.raises("an unknown role is refused", ValueError, excuse_task, roles="P9")
T.raises("an unknown scenario is refused", ValueError, excuse_task, scenarios="bicycles")
want = ["bicycle|C|REASONABLE|chat", "bicycle|BYSTANDER|CAUSE|chat",
        "cheque|I|BLAME|chat"]
sel = excuse_task(cells=",".join(want)).dataset
T.check("-T cells= reads exactly the named cells", sorted(s.id for s in sel) == sorted(want))
tmp = tempfile.mkdtemp()
try:
    cf = os.path.join(tmp, "cells.json")
    json.dump({"cells": want}, open(cf, "w"))
    T.check("-T cells= also takes a JSON file of ids",
            sorted(s.id for s in excuse_task(cells=cf).dataset) == sorted(want))
    sf = os.path.join(tmp, "supplier.json")
    json.dump(["prawns_home|SUPPLIER|BLAME|chat"], open(sf, "w"))
    json.dump(["prawns_home|SUPPLIER|BLAME|chat", "cheque|I|BLAME|chat"], open(sf, "w"))
    T.raises("a SUPPLIER cell is refused unless SUPPLIER is selected, even beside a cell "
             "that is read", ValueError, excuse_task, cells=sf)
    json.dump(["prawns_home|SUPPLIER|BLAME|chat"], open(sf, "w"))
    T.check("the same SUPPLIER cell opens with -T roles=all",
            [s.id for s in excuse_task(roles="all", cells=sf).dataset]
            == ["prawns_home|SUPPLIER|BLAME|chat"])
    pf = os.path.join(tmp, "pf.json")
    json.dump([F.prompt_sha16(s.metadata["system"], s.input) for s in sel], open(pf, "w"))
    T.opens("the pre-flight opens on a cell selection whose prompts are all listed",
            excuse_task, cells=",".join(want), prompt_check=pf)
    T.raises("the pre-flight refuses one more cell than its file lists",
             D.PromptCheckRefused, excuse_task,
             cells=",".join(want + ["ladder|P0|BLAME|chat"]), prompt_check=pf)
finally:
    shutil.rmtree(tmp)
# each refused id is requested beside cells that are read, so the refusal is the cell
# check's and not the empty-selection guard's
T.raises("a cell id the frame does not render is refused (N is not asked REASONABLE), "
         "not dropped from a selection that is otherwise read", ValueError, excuse_task,
         cells="bicycle|N|REASONABLE|chat,bicycle|C|REASONABLE|chat")
T.raises("a mistyped cell id is refused, not dropped", ValueError, excuse_task,
         cells="bicycle|C|REASONABLE|chatt,bicycle|C|CAUSE|chat")
T.raises("a chat cell id on the worksheet frame is refused", ValueError, excuse_task,
         frame="worksheet", cells="bicycle|C|BLAME|chat")
T.check("the worksheet still reads 300 excuse samples by default",
        len(excuse_task(frame="worksheet").dataset) == 300)

# ── 5. THE DUTY TEST: CELLS ────────────────────────────────────────────────────────────
dsel = duty_task(cells="prawns|pro|cause|chat,prawns|cmp|full_priv_first").dataset
T.check("-T cells= on the duty test reads exactly the named cells",
        sorted(s.id for s in dsel) == ["prawns|cmp|full_priv_first", "prawns|pro|cause|chat"])
T.raises("a duty cell outside the pair selection is refused, beside one inside it",
         ValueError, duty_task, pairs="1", cells="prawns|pro|cause|chat,1|pro|bare|chat")

# ── 6. GEMINI 3.1 PRO'S EXCUSE-TEST WIRES ──────────────────────────────────────────────
g = MT.load("gemini_pro")
blocks = {k: MT.wire_group(g, k)[1] for k in ("bicycle", "scenarios", "duty_separate")}
T.check("Gemini 3.1 Pro's bicycle and scenarios wires equal its duty_separate wire, read "
        "sampled at 10 draws, and hash as the harness hashes them",
        all(b["wire"] == blocks["duty_separate"]["wire"] and b["readout"] == "sampled"
            and b["draws"] == 10 and API.wire_sha(b["wire"]) == b["wire_sha"]
            == "94f0bcfaa46e2539228727327fda0d10" for b in blocks.values()))
T.check("its requests pin google-vertex with fallbacks off and leave reasoning at the "
        "vendor's default",
        all((lambda p: p["provider"] == {"order": ["google-vertex"], "allow_fallbacks": False}
             and "reasoning" not in p)(API.payload_for(b["wire"], [])) for b in blocks.values()))

# ── 7. GEMINI 3.8 FLASH'S BICYCLE WIRE IS THE REPORTED READING'S, NOT THE WITHDRAWN ONE ─
# the reported and withdrawn settings, as the readings' call logs record them, are in
# tests/fixtures/gemini_r_bicycle_wire.json
WIRES = json.load(open(os.path.join(ROOT, "tests", "fixtures", "gemini_r_bicycle_wire.json"),
                       encoding="utf-8"))
gr = MT.load("gemini_r")
bw = MT.wire_group(gr, "bicycle")[1]
T.check("gemini_r's bicycle wire is the reported reading's (2048 tokens, effort low, "
        "Google AI Studio)",
        bw["wire"] == WIRES["reported"] and API.wire_sha(bw["wire"]) == bw["wire_sha"]
        and bw["n_calls"] == WIRES["reported_calls"]
        and bw["readout"] == "reasoning" and bw["draws"] == 20, bw["wire"])
T.check("and the withdrawn 8-token wire is kept only as a withdrawn setting in the logs",
        bw["wire"] != WIRES["withdrawn"] and any(
            {k: o.get(k) for k in WIRES["withdrawn"]} == WIRES["withdrawn"]
            and o.get("withdrawn") for o in bw["other_settings_in_the_logs"]))
T.check("a bicycle cell on gemini_r is sent on that wire",
        API.payload_for(bw["wire"], [])["max_tokens"] == 2048
        and API.payload_for(bw["wire"], [])["reasoning"] == {"effort": "low"})

# ── THE REASONING CONFIGURATION, BY WHAT THE MODEL DID (Cassian, 2026-10-17) ────────────
from due_care.scorers import configured
vd = {"readout": "sampled", "wire": {"reasoning": "vendor_default"}}
T.check("a vendor-default wire whose every draw reasoned is configured on (Gemini 3.1 Pro)",
        configured(vd, {"n_draws": 10, "n_reasoning": 10}, False) == "on")
T.check("the same wire with a draw that did not reason stays off",
        configured(vd, {"n_draws": 10, "n_reasoning": 9}, False) == "off")
T.check("a reasoning-off wire whose draws reasoned stays off (flagged, never reclassed)",
        configured({"readout": "sampled", "wire": {"reasoning": {"enabled": False}}},
                   {"n_draws": 10, "n_reasoning": 10}, False) == "off")
T.check("a log-prob cell is logprob, a reasoning readout on",
        configured({"readout": "logprob", "wire": {}}, {}, False) == "logprob"
        and configured({"readout": "reasoning", "wire": {}}, {}, True) == "on")

T.done()
