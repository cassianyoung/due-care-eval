"""Build a results database from fresh readings.

Input: the cells the due_care scorers produce (each Score's metadata, one dict per cell),
from any number of Inspect logs. Output: an SQLite database with the schema in
`db/schema.py` and every view in `db/results_derive.py` and `db/duty_db.py`, so every
position, band, gate, gap, floor, share and class is computed by the SQL views in `db/`.

    python -m db.build_from_readings OUT.sqlite LOG [LOG ...]

Rows written:
  artefacts  one per (model, delivery, test)
  cells      one per excuse-test cell (one per ordering on the worksheet frame)
  orderings  the same rows for local reads, which is what the worksheet band is built on
  duty_*     the duty test's pairs, texts, cells and per-draw values

A cell that has no reading, or that loses more than the allowed share of its draws, is
written with void = 1 (excuse test) or status = 'unread' with its reason (duty test),
never with a substituted value. A duty cell served by more than one provider is unread.
"""
import hashlib
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from db import schema  # noqa: E402
from harness import frames as F  # noqa: E402

RUN = "fresh"
MASS_NOTE = ("not recorded: a duty cell's ev is the mean over draws, and its digit "
             "distribution is carried in the excuse-test cells only")


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _delivery(c):
    return "completion" if c.get("frame") == "worksheet" else "chat"


def _artefact(con, cache, c, entry_of):
    key = (c["model_id"], _delivery(c), c.get("test"))
    if key in cache:
        return cache[key]
    e = entry_of(c["model_id"]) or {}
    ck = e.get("checkpoint") or {}
    ad = e.get("adapter") or {}
    cur = con.execute(
        "INSERT INTO artefacts(path, sha256, run, card, arm, delivery, readout, model_id, "
        "revision, adapter_digest, bank_name, bank_sha256, adapter_family, enters_tables) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
        (f"inspect:{key[0]}:{key[1]}:{key[2]}", "", RUN,
         "api" if c.get("model_kind") == "api" else "local", key[0], key[1],
         c.get("readout"), (e.get("wire") and next(
             (b["wire"]["model"] for b in e["wire"].values() if b), None)) or ck.get("repo"),
         ck.get("revision"), ad.get("digest"), "banks", F.manifest_check()["files"].get(
             "excuse_cases.jsonl" if c.get("test") == "excuse" else "duty_texts.jsonl"),
         e.get("family") or ("adapter" if ad else "none")))
    cache[key] = cur.lastrowid
    return cur.lastrowid


def _void(c, key):
    return 1 if (c.get(key) is None or c.get("usable") is False) else 0


def _excuse(con, c, aid):
    key = c.get("value_key", "ev")
    probs = c.get("probs")
    con.execute(
        "INSERT INTO cells(artefact_id, family, role, question, frame, ev, ev_sd, "
        "answer_mass, mass_0_3, p_yes, probs, n_draws, n_reasoning, n_void, "
        "reasoning_share, thin, void, ev_reasoning_only, ev_all_draws, "
        "reasoning_configured, ev_source, mass_0_3_source, answer_mass_source, "
        "row_text_sha256, prompt_sha256_16) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (aid, c.get("scenario"), c.get("role"), c.get("question"), c.get("ordering"),
         c.get("ev"), c.get("ev_sd"), c.get("answer_mass"), c.get("mass_0_3"),
         c.get("p_yes"), json.dumps(probs) if probs else None, c.get("n_draws"),
         c.get("n_reasoning"), c.get("n_void"), c.get("reasoning_share"),
         (1 if c.get("readout") == "reasoning" and (c.get("n_reasoning") or 0) < 5
          else 0), _void(c, key), c.get("ev_reasoning_only"), c.get("ev_all_draws"),
         c.get("reasoning_configured"),
         "reasoning_only" if c.get("readout") == "reasoning" else c.get("readout"),
         c.get("mass_0_3_source"), c.get("answer_mass_source"), c.get("text_sha256"),
         c.get("prompt_sha256_16")))
    if c.get("model_kind") == "local":
        con.execute(
            "INSERT INTO orderings(artefact_id, family, role, question, frame, ev, "
            "answer_mass, void) VALUES (?,?,?,?,?,?,?,?)",
            (aid, c.get("scenario"), c.get("role"), c.get("question"),
             c.get("ordering") or "chat", c.get("ev") if key == "ev" else c.get("p_yes"),
             c.get("answer_mass"), _void(c, key)))


def _duty_static(con, texts_seen):
    pairs = F.load_json("duty_pairs.json")
    man = F.manifest_check()["files"]
    for pid, p in pairs.items():
        con.execute(
            "INSERT OR IGNORE INTO duty_pairs(pair_id, name, status, pro_label, "
            "priv_label, victim_chose_arrangement, bank, bank_sha) "
            "VALUES (?,?,'reported',?,?,?,?,?)",
            (pid, p["name"], p["pro_label"], p["priv_label"],
             int(p["victim_chose_arrangement"]), "banks/duty_texts.jsonl",
             man["duty_texts.jsonl"]))
    for t in F.load_jsonl("duty_texts.jsonl"):
        tid = f"{t['pair']}|{t['leg']}|{t['kind']}|{t['variant']}"
        kind = {"stated": "convention"}.get(t["kind"], t["kind"])
        con.execute(
            "INSERT OR IGNORE INTO duty_texts(text_id, pair_id, leg, kind, variant, text, "
            "sha256, word_count, bank, bank_sha) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (tid, t["pair"], t["leg"], kind, t["variant"], t["text"], t["sha256"],
             len(t["text"].split()), "banks/duty_texts.jsonl", man["duty_texts.jsonl"]))
        texts_seen[t["sha256"]] = tid


def _duty(con, c, texts_seen, entry_of):
    e = entry_of(c["model_id"]) or {}
    q = c.get("question")
    cond = c.get("condition")
    providers = [p for p in (c.get("providers_served") or []) if p]
    reason = None
    if len(providers) > 1:
        reason = f"served by more than one provider: {providers}"
    elif q == "CMPN" and c.get("gap") is None:
        reason = c.get("why_none") or "no draw parsed"
    elif q != "CMPN" and c.get("ev") is None:
        reason = c.get("why_none") or "no reading"
    elif c.get("usable") is False:
        reason = (f"{c.get('n_void')} of {c.get('n_asked') or c.get('n_draws')} draws void, "
                  f"above the allowed share")
    leg = "both" if q == "CMPN" else c.get("leg")
    cid = f"{RUN}|{c['model_id']}|{c.get('pair')}|{cond}|{leg}|{q}|{c.get('frame')}"
    wire = c.get("wire_sha") or "local"
    con.execute(
        "INSERT INTO duty_cells(cell_id, run, arm, pair_id, leg, condition, question, "
        "text_id, n_draws, n_parsed, ev, score_pro, score_priv, sd, answer_mass, "
        "answer_mass_note, n_reasoning, reasoning_share, provider, provider_served, "
        "model_id, wire_sha, source, artefact, artefact_sha, bank, preferred, status, "
        "unread_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,?,?,?,?,?,?,'main',"
        "?,?,?,1,?,?)",
        (cid, RUN, c["model_id"], c.get("pair"), leg, cond, q,
         texts_seen.get(c.get("text_sha256")), c.get("n_draws") or 0,
         (c.get("n_draws") or 0) - (c.get("n_void") or 0),
         None if reason else c.get("ev"), None if reason else c.get("score_pro"),
         None if reason else c.get("score_priv"), c.get("ev_sd"), MASS_NOTE,
         c.get("n_reasoning"), c.get("reasoning_share"),
         providers[0] if providers else ("local" if c.get("model_kind") == "local"
                                         else "unrecorded"),
         ",".join(providers) or None, c["model_id"], wire,
         f"inspect:{c['model_id']}", "", "banks/duty_texts.jsonl",
         "unread" if reason else "read", reason))
    vals = c.get("draw_gaps") if q == "CMPN" else c.get("draw_evs")
    for i, v in enumerate(vals or []):
        con.execute("INSERT INTO duty_draws(cell_id, value_index, value, gap) "
                    "VALUES (?,?,?,?)", (cid, i, None if q == "CMPN" else v,
                                         v if q == "CMPN" else None))
    if e.get("config_set"):
        con.execute("INSERT OR IGNORE INTO duty_model_arm(model_id, arm, config_set) "
                    "VALUES (?,?,?)", (c["model_id"], c["model_id"], e["config_set"]))


def _merge_orderings(rows):
    """One duty cell from a worksheet cell's readings under its orderings: the mean over
    the orderings that read, each ordering's value kept as a draw, the population sd
    over them (as the worksheet band is built), and void only if every ordering is."""
    read = [r for r in rows if not r.get("void") and r.get("ev") is not None]
    evs = [r["ev"] for r in read]
    m = sum(evs) / len(evs) if evs else None
    sd = (sum((x - m) ** 2 for x in evs) / len(evs)) ** 0.5 if evs else None
    first = rows[0]
    out = {k: v for k, v in first.items() if k not in ("ordering", "ev", "void")}
    out.update({"ev": m, "ev_sd": sd, "draw_evs": evs, "n_draws": len(rows),
                "n_void": len(rows) - len(read), "void": not read, "usable": bool(read),
                "why_none": None if read else (first.get("why_none")
                                               or "no ordering read")})
    return out


def _duty_cells(cells):
    """Duty cells as the views consume them: chat cells as read, worksheet cells merged
    over their orderings."""
    out, groups = [], {}
    for c in cells:
        if c.get("test") != "duty" or not c.get("model_id"):
            continue
        if c.get("frame") == "worksheet":
            key = (c["model_id"], c.get("pair"), c.get("condition"), c.get("leg"),
                   c.get("question"))
            groups.setdefault(key, []).append(c)
        else:
            out.append(c)
    return out + [_merge_orderings(g) for g in groups.values()]


def build(cells, out=":memory:", entry_of=None):
    """Write `cells` into a fresh database at `out` and return the connection."""
    if out != ":memory:" and os.path.exists(out):
        raise FileExistsError(f"{out} exists; a database is built once, never edited")
    if entry_of is None:
        from due_care import model_table

        def entry_of(mid):
            try:
                return model_table.load(mid)
            except Exception:
                return None
    con = sqlite3.connect(out)
    schema.create(con)
    cache, texts_seen = {}, {}
    if any(c.get("test") == "duty" for c in cells):
        _duty_static(con, texts_seen)
    for c in _duty_cells(cells):
        _duty(con, c, texts_seen, entry_of)
    for c in cells:
        if not c.get("model_id"):
            continue
        if c.get("test") == "duty":
            if (c.get("pair") == "prawns" and c.get("condition") == "bare"
                    and c.get("question") == "BLAME"):
                # the prawns pair's two cases are the careless role of the two food
                # scenarios, so they are also excuse-test cells; the duty views find an
                # arm's prawns separate gap there
                scen = "prawns_restaurant" if c.get("leg") == "pro" else "prawns_home"
                twin = {**c, "scenario": scen, "role": "C"}
                _excuse(con, twin, _artefact(con, cache, twin, entry_of))
        else:
            _excuse(con, c, _artefact(con, cache, c, entry_of))
    con.commit()
    return con


def cells_from_logs(paths):
    """Every scored cell's metadata from one or more Inspect logs."""
    from inspect_ai.log import read_eval_log
    cells = []
    for p in paths:
        log = read_eval_log(p)
        for s in log.samples or []:
            for sc in (s.scores or {}).values():
                if sc.metadata and sc.metadata.get("model_id"):
                    cells.append(dict(sc.metadata))
    return cells


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2:
        sys.exit(__doc__)
    con = build(cells_from_logs(argv[1:]), argv[0])
    n = con.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
    d = con.execute("SELECT COUNT(*) FROM duty_cells").fetchone()[0]
    print(f"{argv[0]}: {n} excuse cells, {d} duty cells")


if __name__ == "__main__":
    main()
