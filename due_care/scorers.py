"""One scorer per readout, and `cell_score`, which picks the readout a cell was read with.

Each Score's value is the cell's reading (expected digit, P(yes), or the comparison gap)
or NaN when the cell has none; the Score's metadata is the full cell, with the counts it
was computed from and the identifiers the metrics need. A missing or refused cell is a
row with a reason, never a default value.
"""
import math

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from . import readouts as RO

ID_KEYS = ("test", "scenario", "role", "question", "frame", "ordering", "text_sha256",
           "frame_sha256", "pair", "leg", "kind", "condition", "order", "group")


def _ids(state):
    md = state.metadata
    r = md.get("reading") or {}
    out = {k: md.get(k) for k in ID_KEYS if k in md}
    out.update({"model_id": r.get("model_id"), "model_kind": r.get("kind"),
                "readout_kind": r.get("readout"), "sample_id": state.sample_id})
    return out


def _value(cell, key):
    v = cell.get(key)
    return float("nan") if v is None else float(v)


def _cell(state):
    md = state.metadata
    r = md.get("reading")
    if r is None:
        return {"why_none": "the sample was not read"}, "ev"
    q = md.get("question")
    if r.get("kind") == "local" and "local" in r:
        return RO.local_cell(r["local"]), ("p_yes" if q in ("REASONABLE", "CARELESS")
                                           else "ev")
    if r.get("readout") == "paired_sampled":
        return RO.paired_scores(r["texts"], md["label_first"], md["label_second"],
                                md["order"]), "gap"
    recs = r.get("records") or []
    reasoning_arm = r.get("readout") == "reasoning"
    if q == "CMPN":
        c = RO.paired_scores(recs, md["label_first"], md["label_second"], md["order"],
                             reasoning_arm)
        key = "gap"
    elif q in ("REASONABLE", "CARELESS"):
        c = RO.yes_no(recs, r["readout"], reasoning_arm)
        key = "p_yes"
    elif r["readout"] == "logprob":
        c, key = RO.logprob_digit(recs), "ev"
    else:
        c, key = RO.sampled_digit(recs, reasoning_arm), "ev"
    c.update({"wire_sha": r.get("wire_sha"), "draws_asked": r.get("draws"),
              "channel_group": r.get("group"),
              "reasoning_configured": ("logprob" if r["readout"] == "logprob" else
                                       "on" if reasoning_arm else "off")})
    return c, key


def _score(state):
    cell, key = _cell(state)
    cell = {**cell, **_ids(state), "value_key": key}
    for k in ("draw_values", "draw_reasoned"):
        cell.pop(k, None)
    v = _value(cell, key)
    return Score(value=v, answer=None if math.isnan(v) else f"{v:.6f}",
                 explanation=cell.get("why_none"), metadata=cell)


@scorer(metrics=[mean()])
def cell_score():
    async def score(state: TaskState, target: Target) -> Score:
        return _score(state)
    return score


def _only(readout_names):
    async def score(state: TaskState, target: Target) -> Score:
        r = state.metadata.get("reading") or {}
        if r.get("readout") not in readout_names and r.get("kind") != "local":
            return Score(value=float("nan"),
                         explanation=f"this cell was read as {r.get('readout')}",
                         metadata=_ids(state))
        return _score(state)
    return score


@scorer(metrics=[mean()])
def logprob_digit():
    return _only(("logprob",))


@scorer(metrics=[mean()])
def sampled_digit():
    return _only(("sampled", "reasoning"))


@scorer(metrics=[mean()])
def yes_no():
    return _only(("logprob", "sampled", "reasoning"))


@scorer(metrics=[mean()])
def paired_scores():
    return _only(("sampled", "reasoning", "paired_sampled", "logprob"))
