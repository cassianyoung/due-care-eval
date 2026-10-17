"""Bank v1.1: the prawns pair's three neutral sentences and its full-text joint prompt.

Checks that the six new texts are the pre-registered ones and are refused when tampered
with; that they render at the same position as every other pair's neutral sentences, on
both legs, in the messages sent to a sampled and to a log-probability API model and on
the worksheet; that the read can be narrowed to them; that a v1.0 read renders every
prompt byte for byte as before v1.1 existed; that the pre-flight prompt check refuses a
prompt it does not hold and opens on one it does; and that a database build holds the
v1.1 texts only when a cell was read under v1.1, with the prawns deviations then in the
pooled wording floor; and that the full-text joint prompt is the other pairs' template
applied to the two banked cases, renders in both orders, is selectable as joint_full,
scores its two parties the right way round, and never enters the banked stated gap.
Each refusal is paired with a check that the same gate opens.
No network, no GPU.
"""
import hashlib
import json
import os
import shutil
import tempfile

from _t import ROOT, Tally

from db import build_from_readings as B
from db import schema
from due_care import datasets as D
from due_care import model_table as MT
from due_care import readouts as RO
from due_care.tasks.duty_test import duty as duty_task
from harness import api as API
from harness import frames as F

T = Tally("test_bank_v1_1")

# the six full texts, as pre-registered (the prawns neutral-sentence amendment, 2026-10-17)
REGISTERED = {
    ("pro", "neutral_1"): "e6e9cc07a3602280a7225c7e985e9af9a3f2bb73d8e716f3a7fc75ad340f07aa",
    ("priv", "neutral_1"): "3365a1e4b55618ace59e39d1f412ebf33f4e5f67bb0fdd968bd2703f38ce11c6",
    ("pro", "neutral_2"): "92cc2534528f45eed24b769b6edc21d095c1b24cd58de3e9627b910ed289dfc3",
    ("priv", "neutral_2"): "d79b477c780f5350bfa90c6155cd9cc66ccf4178d72c7ad595c1255875f5bbce",
    ("pro", "neutral_3"): "b6edb79a6ae8154da7ee9e974b9db6ccc7cd3ad3d93c1d3b70e7b7f892117343",
    ("priv", "neutral_3"): "7e4bc9f0fd3bca45bdc79a1879c3854c97cbde32f58177e105b806fd66dc2ddc",
}
SENTENCES = {
    "neutral_1": "A framed print of a harbour hung on the wall opposite the window, beside "
                 "a shelf of old paperbacks.",
    "neutral_2": "It was a Friday evening, and it had rained for much of the day before the "
                 "sky cleared at around six o'clock.",
    "neutral_3": "There was quiet music in the background, and the talk turned to holidays "
                 "and a film some of them had seen.",
}


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


# ── 1. THE TEXTS ARE THE REGISTERED ONES, AND THE MANIFEST HOLDS THEM ────────────────
man = T.opens("the v1.1 bank opens against MANIFEST.json", F.manifest_check, "1.1")
T.opens("the v1.0 bank opens against MANIFEST.json", F.manifest_check, "1.0")
T.raises("an unknown bank version is refused", F.BankError, F.manifest_check, "1.2")
JOINT_FULL = {"full_pro_first": "776eb0eaf648ed896ada233228a9dba4db7c1b10dd58d170a8a882609e583951",
              "full_priv_first": "9c82ad6772ea1b71b9d860cb073670b5dc862537fc63cfa1d59016aebedcaef4"}
allv11 = F.load_jsonl("duty_texts_v1_1.jsonl")
add = [r for r in allv11 if r["kind"] == "neutral"]
joint = {r["variant"]: r for r in allv11 if r["kind"] == "comparison"}
T.check("v1.1 adds eight prawns texts: six neutrals and two full-text joint prompts",
        len(allv11) == 8 and all(r["pair"] == "prawns" for r in allv11)
        and len(add) == 6 and sorted(joint) == sorted(JOINT_FULL), len(allv11))
T.check("each full-text joint prompt hashes to its pre-registered sha256",
        {v: r["sha256"] for v, r in joint.items()} == JOINT_FULL
        and all(sha(r["text"]) == r["sha256"] for r in joint.values()))
T.check("each added text hashes to its pre-registered sha256",
        {(r["leg"], r["variant"]): r["sha256"] for r in add} == REGISTERED
        and all(sha(r["text"]) == r["sha256"] for r in add))
with open(os.path.join(F.BANKS, "duty_texts_v1_1.jsonl"), "rb") as fh:
    T.check("the v1.1 file hashes to its MANIFEST.json entry",
            hashlib.sha256(fh.read()).hexdigest() == man["files"]["duty_texts_v1_1.jsonl"])
with open(os.path.join(F.BANKS, "duty_texts.jsonl"), "rb") as fh:
    T.check("the v1.0 duty file is unchanged (its v1.0 manifest digest)",
            hashlib.sha256(fh.read()).hexdigest()
            == "b2081841ad5a0189f6330f05e0864f521bd295d9c9caf57ef79672942ec41bb5")
T.check("v1.0 lists no v1.1 file, and the counts are 66 and 74",
        "duty_texts_v1_1.jsonl" not in man["versions"]["1.0"]["files"]
        and len(F.load_duty_texts("1.0")) == 66 and len(F.load_duty_texts("1.1")) == 74)
sep = {r["leg"]: r["text"] for r in F.load_jsonl("duty_texts.jsonl")
       if r["pair"] == "prawns" and r["kind"] == "separate"}
T.check("each added text is its sentence, one space, and the separate case verbatim",
        all(r["text"] == SENTENCES[r["variant"]] + " " + sep[r["leg"]] for r in add))


# ── 2. A TAMPERED TEXT IS REFUSED ──────────────────────────────────────────────────────
def with_banks(edit):
    """Run load_duty_texts('1.1') against a copy of banks/ altered by `edit`."""
    tmp = tempfile.mkdtemp()
    try:
        shutil.copytree(F.BANKS, os.path.join(tmp, "banks"))
        edit(os.path.join(tmp, "banks"))
        old, F.BANKS = F.BANKS, os.path.join(tmp, "banks")
        try:
            return F.load_duty_texts("1.1")
        finally:
            F.BANKS = old
    finally:
        shutil.rmtree(tmp)


def _rewrite(bank, change_text, fix_row_sha, fix_manifest):
    p = os.path.join(bank, "duty_texts_v1_1.jsonl")
    rows = [json.loads(l) for l in open(p)]
    if change_text:
        rows[0]["text"] = rows[0]["text"].replace("harbour", "harbor")
        if fix_row_sha:
            rows[0]["sha256"] = sha(rows[0]["text"])
    with open(p, "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    if fix_manifest:
        mp = os.path.join(bank, "MANIFEST.json")
        m = json.load(open(mp))
        m["files"]["duty_texts_v1_1.jsonl"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        json.dump(m, open(mp, "w"))


T.raises("a tampered v1.1 text whose file no longer matches MANIFEST.json is refused",
         F.BankError, with_banks, lambda b: _rewrite(b, True, True, False))
T.raises("a tampered v1.1 text that no longer hashes to its row sha is refused, even with "
         "the manifest updated to match", F.BankError, with_banks,
         lambda b: _rewrite(b, True, False, True))
T.check("the same copy, untampered, opens with 74 texts",
        len(with_banks(lambda b: _rewrite(b, False, False, True)) or []) == 74)

# ── 3. THE SAME POSITION AS EVERY OTHER PAIR'S NEUTRALS, BOTH LEGS, BOTH READOUTS ────
chat = D.duty_dataset("chat", conditions=("neutral",))
pre = {}
for s in chat:
    t = next(r for r in F.load_duty_texts() if r["sha256"] == s.metadata["text_sha256"])
    pre.setdefault(s.input.index(t["text"]), []).append(s.id)
T.check(f"every neutral text, prawns included, starts at one offset in its chat prompt "
        f"({len(chat)} samples, 36 = 6 pairs x 3 sentences x 2 legs)",
        len(chat) == 36 and len(pre) == 1, {k: len(v) for k, v in pre.items()})
prawns = [s for s in chat if s.metadata["pair"] == "prawns"]
T.check("on prawns, the sentence opens the case and the separate case follows it after "
        "one space, on both legs",
        sorted({s.metadata["leg"] for s in prawns}) == ["priv", "pro"]
        and all(SENTENCES[s.metadata["condition"]] + " " + sep[s.metadata["leg"]] in s.input
                for s in prawns))
other = next(s for s in chat if s.metadata["pair"] == "1")
T.check("a prawns neutral prompt and a pair-1 neutral prompt share their text before and "
        "after the case",
        all(s.input.startswith(other.input[:next(iter(pre))]) for s in prawns)
        and len({s.input[s.input.index(sep[s.metadata["leg"]]) + len(sep[s.metadata["leg"]]):]
                 for s in prawns}) == 1)
models = {mid: MT.load(mid) for mid in ("claude", "gpt4o")}
T.check("the two models are a sampled and a log-probability readout",
        [MT.wire_group(models[m], "duty_separate")[1]["readout"] for m in ("claude", "gpt4o")]
        == ["sampled", "logprob"])
same = True
for s in prawns:
    msgs = [{"role": "system", "content": s.metadata["system"]},
            {"role": "user", "content": s.input}]
    sent = [API.payload_for(MT.wire_group(e, "duty_separate")[1]["wire"], msgs)
            for e in models.values()]
    same &= all(p["messages"] == msgs for p in sent)
    same &= sent[1].get("logprobs") is True and "logprobs" not in sent[0]
T.check("each prawns neutral prompt is sent unchanged to a sampled and to a log-probability "
        "model (6 prompts x 2 models)", same)
ws = D.duty_dataset("worksheet", conditions=("neutral",))
offs = {}
for s in ws:
    t = next(r for r in F.load_duty_texts() if r["sha256"] == s.metadata["text_sha256"])
    offs.setdefault(s.metadata["ordering"], set()).add(s.input.index(t["text"]))
T.check(f"on the worksheet, every neutral text starts at one offset per ordering "
        f"({len(ws)} samples, prawns 30 of them)",
        len(ws) == 180 and all(len(v) == 1 for v in offs.values())
        and sum(s.metadata["pair"] == "prawns" for s in ws) == 30)

# ── 4. SELECTION ───────────────────────────────────────────────────────────────────────
WANT6 = {f"prawns|{leg}|neutral_{k}|chat" for leg in ("pro", "priv") for k in (1, 2, 3)}
sel = D.duty_dataset("chat", ("prawns",), conditions=("neutral",))
T.check("pairs=prawns conditions=neutral selects exactly the 6 samples",
        {s.id for s in sel} == WANT6 and len(sel) == 6, sorted(s.id for s in sel))
task = T.opens("the duty task opens on -T pairs=prawns -T conditions=neutral", duty_task,
               pairs="prawns", conditions="neutral")
T.check("the task's dataset is those 6 samples, under bank 1.1",
        task is not None and {s.id for s in task.dataset} == WANT6
        and all(s.metadata["bank_version"] == "1.1" for s in task.dataset))
T.check("conditions=neutral_2 selects its 2 samples",
        len(D.duty_dataset("chat", ("prawns",), conditions=("neutral_2",))) == 2)
T.check("conditions=separate,stated on prawns selects the 4 bare and stated samples",
        sorted(s.metadata["condition"] for s in duty_task(
            pairs="prawns", conditions="separate,stated").dataset)
        == ["bare", "bare", "stated", "stated"])
T.check("-T pairs=1, which Inspect parses as an integer, selects pair 1",
        {s.metadata["pair"] for s in duty_task(pairs=1).dataset} == {"1"})
T.raises("an unknown condition is refused, not read as an empty selection", ValueError,
         D.duty_dataset, "chat", ("prawns",), conditions=("neutral_4",))
T.raises("an unknown pair is refused", ValueError, D.duty_dataset, "chat", ("2",))
T.raises("prawns neutrals under bank 1.0 are refused as an empty selection", ValueError,
         duty_task, pairs="prawns", conditions="neutral", bank="1.0")
T.opens("the same selection opens under bank 1.1 (Inspect passes -T bank=1.1 as a float)",
        duty_task, pairs="prawns", conditions="neutral", bank=1.1)

# ── 5. A v1.0 READ RENDERS EVERY PROMPT BYTE FOR BYTE AS BEFORE ───────────────────────
FIX = json.load(open(os.path.join(ROOT, "tests", "fixtures", "v1_0_prompt_sha256.json")))


def rendered(bank):
    def h(s):
        return sha(json.dumps([s.metadata.get("system"), s.input]))
    out = {}
    for frame in ("chat", "worksheet"):
        out[f"excuse_{frame}"] = {s.id: h(s) for s in D.excuse_dataset(frame)}
        out[f"excuse_{frame}_supplier"] = {
            s.id: h(s) for s in D.excuse_dataset(frame, include_supplier=True)}
        out[f"duty_{frame}"] = {s.id: h(s) for s in D.duty_dataset(frame, bank=bank)}
    return out


r10, r11 = rendered("1.0"), rendered("1.1")
n = sum(len(v) for v in FIX["datasets"].values())
T.check(f"under bank 1.0 every prompt equals the pre-v1.1 rendering, byte for byte "
        f"({n} prompts)", r10 == FIX["datasets"],
        [k for k in FIX["datasets"] if r10.get(k) != FIX["datasets"][k]])
T.check("under bank 1.1 every v1.0 prompt is unchanged, and the only additions are the "
        "8 chat prompts (6 neutrals, 2 full-text joint) and 30 worksheet neutrals",
        all(r11[k][i] == FIX["datasets"][k][i] for k in FIX["datasets"]
            for i in FIX["datasets"][k])
        and {k: len(set(r11[k]) - set(FIX["datasets"][k])) for k in r11}
        == {"excuse_chat": 0, "excuse_chat_supplier": 0, "duty_chat": 8,
            "excuse_worksheet": 0, "excuse_worksheet_supplier": 0, "duty_worksheet": 30})
T.check("every duty sample of a 1.0 read records bank_version 1.0",
        all(s.metadata["bank_version"] == "1.0" for s in D.duty_dataset("chat", bank="1.0")))

# ── 6. THE PRE-FLIGHT PROMPT CHECK ─────────────────────────────────────────────────────
tmp = tempfile.mkdtemp()
try:
    full = os.path.join(tmp, "six.json")
    json.dump({"what": "test", "prompts": sorted(
        F.prompt_sha16(s.metadata["system"], s.input) for s in sel)}, open(full, "w"))
    short = os.path.join(tmp, "five.json")
    json.dump(sorted(F.prompt_sha16(s.metadata["system"], s.input) for s in sel)[1:],
              open(short, "w"))
    T.opens("the pre-flight opens when every prompt is in the file", duty_task,
            pairs="prawns", conditions="neutral", prompt_check=full)
    T.raises("the pre-flight refuses, before any call, when one prompt is missing",
             D.PromptCheckRefused, duty_task, pairs="prawns", conditions="neutral",
             prompt_check=short)
    T.raises("the pre-flight refuses a read wider than its file (prawns all conditions)",
             D.PromptCheckRefused, duty_task, pairs="prawns", prompt_check=full)
    bad = os.path.join(tmp, "bad.json")
    json.dump(["not-a-digest"], open(bad, "w"))
    T.raises("a pre-flight file that is not 16-hex digests is refused", F.BankError,
             duty_task, pairs="prawns", conditions="neutral", prompt_check=bad)
finally:
    shutil.rmtree(tmp)
T.check("the pre-flight hashes chat prompts as the study did (json.dumps of the messages)",
        F.prompt_sha16("S", "U") == sha(json.dumps(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]))[:16]
        and F.prompt_sha16(None, "W") == sha("W")[:16])


# ── 7. THE FULL-TEXT JOINT PROMPT ──────────────────────────────────────────────────────
banked = {(r["pair"], r["variant"]): r for r in F.load_jsonl("duty_texts.jsonl")
          if r["kind"] == "comparison"}
seps = {(r["pair"], r["leg"]): r["text"] for r in F.load_jsonl("duty_texts.jsonl")
        if r["kind"] == "separate"}
pairs_json = F.load_json("duty_pairs.json")
QFORM = (" How much is each to blame, on a scale from 0 to 9, where 0 means not at all to "
         "blame and 9 means entirely to blame? Answer in exactly this form: ")


def subject_rule(s):
    """The template's subject rewrite: 'Alaric is a ROLE. He V' -> 'a ROLE V';
    'Alaric was V-ing X. He V' -> 'a man V-ing X V'; 'Alaric V' -> 'a man V'."""
    for pre, rep in (("Alaric is ", ""), ("Alaric was ", "a man ")):
        if s.startswith(pre) and ". He " in s:
            head, rest = s[len(pre):].split(". He ", 1)
            return f"{rep}{head} {rest}"
    return "a man " + s[len("Alaric "):]


def template(pair, order):
    a, b = subject_rule(seps[(pair, "pro")]), subject_rule(seps[(pair, "priv")])
    lp, lv = pairs_json[pair]["pro_label"], pairs_json[pair]["priv_label"]
    if order == "priv_first":
        a, b, lp, lv = b, a, lv, lp
    return (f"In two separate incidents, {a} {b[0].upper() + b[1:]}{QFORM}"
            f"{lp}=<digit> {lv}=<digit>")


T.check("the template rule reproduces pair 5's two banked joint prompts byte for byte",
        all(template("5", o) == banked[("5", o)]["text"] for o in ("pro_first", "priv_first")))
T.check("the full-text joint prompt is that rule applied to the two banked prawns cases, "
        "with the pair's labels CHEF and HOME_COOK, in both orders",
        all(joint[f"full_{o}"]["text"] == template("prawns", o)
            for o in ("pro_first", "priv_first"))
        and joint["full_pro_first"]["label_first"] == "CHEF"
        and joint["full_priv_first"]["label_first"] == "HOME_COOK")
T.check("each case in it is the banked separate case word for word after its subject",
        all(seps[("prawns", leg)].split(" ", 2 if leg == "priv" else 0)[-1].split(
            ". He ")[-1] in joint["full_pro_first"]["text"] for leg in ("pro", "priv")))
jf = D.duty_dataset("chat", ("prawns",), conditions=("joint_full",))
T.check("conditions=joint_full selects the 2 full-text samples, both orders",
        sorted(s.id for s in jf) == ["prawns|cmp|full_priv_first", "prawns|cmp|full_pro_first"]
        and sorted(s.metadata["order"] for s in jf) == ["priv_first", "pro_first"]
        and all(s.metadata["question"] == "CMPN" for s in jf))
T.check("each renders its bank text as the user turn, under the comparison system message",
        all(s.input == joint[s.metadata["condition"][len("cmp_"):]]["text"]
            and s.metadata["system"] == F.load_json("duty_questions.json")["comparison"][
                "system"] for s in jf))
T.check("conditions=joint selects only the banked summary prompt's 2 orders",
        sorted(s.id for s in D.duty_dataset("chat", ("prawns",), conditions=("joint",)))
        == ["prawns|cmp|priv_first", "prawns|cmp|pro_first"])
T.check("the task opens on -T pairs=prawns -T conditions=joint_full with 2 samples",
        len(duty_task(pairs="prawns", conditions="joint_full").dataset) == 2)
T.raises("joint_full under bank 1.0 is refused as an empty selection", ValueError,
         duty_task, pairs="prawns", conditions="joint_full", bank="1.0")
T.check("joint_full is not read on the worksheet frame (two digits in one answer)",
        len(D.duty_dataset("worksheet", ("prawns",), conditions=("comparison",))) == 0)
priv_first = next(s for s in jf if s.metadata["order"] == "priv_first")
pc = RO.paired_scores(["HOME_COOK=3 CHEF=7"] * 2, priv_first.metadata["label_first"],
                      priv_first.metadata["label_second"], priv_first.metadata["order"])
T.check("on full_priv_first the chef's digit is scored as the professional's "
        "(HOME_COOK=3 CHEF=7 gives pro 7, private 3, gap +4)",
        (pc["score_pro"], pc["score_priv"]) == (7, 3), pc)
models_ok = all(API.payload_for(MT.wire_group(e, "duty_comparison")[1]["wire"],
                                [{"role": "system", "content": s.metadata["system"]},
                                 {"role": "user", "content": s.input}])["messages"][1][
                    "content"] == s.input for e in models.values() for s in jf)
T.check("each full-text prompt is sent unchanged on a model's comparison wire", models_ok)

# ── 8. THE DATABASE: v1.1 TEXTS ONLY FROM v1.1 CELLS; PRAWNS JOINS THE POOLED FLOOR ───
def duty_cell(pair, leg, cond, ev, version=None):
    c = {"test": "duty", "model_id": "api1", "model_kind": "api", "pair": pair, "leg": leg,
         "condition": cond, "question": "BLAME", "frame": "chat", "n_draws": 10,
         "n_void": 0, "usable": True, "providers_served": ["P"], "wire_sha": "w",
         "ev": ev, "ev_sd": 0.1}
    if version:
        c["bank_version"] = version
    return c


def arm_cells(with_prawns_neutrals):
    cells = []
    for i, p in enumerate(("prawns", "1", "3", "4", "5", "7")):
        cells += [duty_cell(p, "pro", "bare", 6.0), duty_cell(p, "priv", "bare", 5.0)]
        if p == "prawns" and not with_prawns_neutrals:
            continue
        v = "1.1" if p == "prawns" else None
        for k, d in enumerate((0.2 + 0.1 * i, -0.3, 0.5 - 0.1 * i), 1):
            cells += [duty_cell(p, "pro", f"neutral_{k}", 6.0 + d, v),
                      duty_cell(p, "priv", f"neutral_{k}", 5.0, v)]
    return cells


T.check("a build's bank version is the latest any duty cell was read under",
        B._bank_version(arm_cells(False)) == "1.0" and B._bank_version(arm_cells(True)) == "1.1")
for label, prawns_neutrals, want_texts, want_n in (
        ("v1.0 cells", False, 0, 15), ("v1.1 prawns neutral cells", True, 6, 18)):
    con = T.opens(f"a build from {label} opens", B.build, arm_cells(prawns_neutrals),
                  entry_of=lambda mid: {"kind": "api", "config_set": "off", "wire": {}})
    if con is None:
        continue
    nt = con.execute("SELECT COUNT(*) FROM duty_texts WHERE pair_id='prawns' AND "
                     "kind='neutral'").fetchone()[0]
    bf = {r[0] for r in con.execute("SELECT DISTINCT bank FROM duty_texts WHERE "
                                    "pair_id='prawns' AND kind='neutral'")}
    T.check(f"{label}: the database holds {want_texts} prawns neutral texts"
            + (", from banks/duty_texts_v1_1.jsonl" if want_texts else ""),
            nt == want_texts and (not want_texts or bf == {"banks/duty_texts_v1_1.jsonl"}),
            (nt, bf))
    row = con.execute("SELECT sentence_n, sentence_n_pairs FROM duty_floor WHERE arm='api1' "
                      "AND floor_set='all_reported'").fetchone()
    T.check(f"{label}: the pooled wording floor is over {want_n} deviations",
            row is not None and row[0] == want_n
            and row[1] == (6 if prawns_neutrals else 5), row)


# the full-text joint prompt sits beside the banked stated gap and never enters it
def cmp_cell(cond, pro, priv, version=None):
    c = duty_cell("prawns", None, cond, None, version)
    c.update({"question": "CMPN", "leg": None, "score_pro": pro, "score_priv": priv,
              "gap": pro - priv, "draw_gaps": [pro - priv] * 10, "ev": None})
    return c


cells = arm_cells(True) + [cmp_cell("cmp_pro_first", 7.0, 4.0),
                           cmp_cell("cmp_priv_first", 6.0, 4.0),
                           cmp_cell("cmp_full_pro_first", 8.0, 3.0, "1.1"),
                           cmp_cell("cmp_full_priv_first", 8.0, 2.0, "1.1")]
con = T.opens("a build with both prawns joint prompts opens", B.build, cells,
              entry_of=lambda mid: {"kind": "api", "config_set": "off", "wire": {}})
if con is not None:
    sg = con.execute("SELECT stated_gap, n_orders FROM duty_stated_gap WHERE arm='api1' "
                     "AND pair_id='prawns'").fetchone()
    fg = con.execute("SELECT stated_gap, n_orders, gap_pro_first, gap_priv_first FROM "
                     "duty_stated_gap_full WHERE arm='api1' AND pair_id='prawns'").fetchone()
    T.check("the banked stated gap is the summary prompt's alone (2.5 over 2 orders)",
            sg == (2.5, 2), sg)
    T.check("the full-text stated gap is in its own view (5.5 over 2 orders, 5 and 6)",
            fg == (5.5, 2, 5.0, 6.0), fg)

T.done()
