"""From raw draws to one cell: the four readouts.

Each function takes the records a solver collected and returns a cell dict with every
count it was computed from. A void draw is counted, never replaced by a value, and a
cell with no usable draw has `ev` (or `p_yes`, or the gap) None and a reason.

Reasoning arms are reported from their reasoning draws. A cell that loses more than
VOID_MAX_FRACTION of its draws (to truncation, or to any other void) is reported with its
void count and marked not usable; truncations are also counted on their own.
"""
import statistics

from harness import api as API

VOID_MAX_FRACTION = 0.10


def _base(records):
    errors = [r for r in records if "error" in r]
    ok = [r for r in records if "error" not in r]
    providers = sorted({r.get("provider_served") or "" for r in ok})
    served = sorted({r.get("model_served") or "" for r in ok})
    return ok, {"n_asked": len(records), "n_errors": len(errors),
                "errors": [e["error"] for e in errors][:3],
                "providers_served": providers, "models_served": served,
                "provider_constant": len(providers) <= 1,
                "wire_sha": (records[0].get("wire_sha") if records else None),
                "reasoning_on_the_wire": (ok[0].get("reasoning_on_the_wire") if ok else None),
                "top_k": (ok[0].get("top_k") if ok else None),
                "cost_usd": sum(r.get("cost_usd") or 0.0 for r in ok)}


def _why(cell, reason):
    """The reason a cell has no reading: every call failing comes first."""
    if cell["n_asked"] and cell["n_errors"] == cell["n_asked"]:
        return f"every call failed: {cell['errors'][0]}"
    return reason


def _usable(n_void_trunc, n_asked):
    if not n_asked:
        return None, None
    frac = n_void_trunc / n_asked
    return frac <= VOID_MAX_FRACTION, frac


def logprob_digit(records):
    """Expected digit from each draw's returned top-k, renormalised over the digits;
    the cell is the mean over draws. Answer mass is the mean raw digit mass."""
    ok, cell = _base(records)
    evs, dists, masses, void = [], [], [], 0
    for r in ok:
        dist, mass = API.digit_distribution([{"token": t["token"], "logprob": t["logprob"]}
                                             for t in r.get("top_tokens") or []])
        masses.append(mass)
        if dist is None:
            void += 1
            continue
        dists.append(dist)
        evs.append(sum(d * p for d, p in dist.items()))
    probs = ([statistics.mean(d[i] for d in dists) for i in range(10)] if dists else None)
    usable, frac = _usable(cell["n_errors"] + void, cell["n_asked"])
    cell.update({"readout": "logprob", "n_draws": len(ok), "n_void": void,
                 "ev": statistics.mean(evs) if evs else None,
                 "ev_sd": statistics.pstdev(evs) if len(evs) > 1 else None,
                 "draw_evs": evs, "probs": probs, "mass_0_3": sum(probs[:4]) if probs else None,
                 "mass_0_3_source": "sum of the mean renormalised digit distribution",
                 "answer_mass": statistics.mean(masses) if masses else None,
                 "answer_mass_source": "mean digit mass in the returned top-k",
                 "usable": usable, "void_fraction": frac,
                 "why_none": (None if evs
                              else _why(cell, "no digit token in any returned top-k"))})
    return cell


def sampled_digit(records, reasoning_arm):
    """Mean parsed digit over draws. On a reasoning arm the reading is the mean over
    draws that reasoned; the mean over all parsed draws is carried beside it."""
    ok, cell = _base(records)
    vals, reasoned, void, trunc = [], [], 0, 0
    for r in ok:
        v = API.parse_digit(r.get("raw_text"))
        if v is None:
            void += 1
            trunc += API.truncated(r)
            continue
        vals.append(float(v))
        reasoned.append(API.reasoned(r))
    rv = [v for v, x in zip(vals, reasoned) if x]
    reading = rv if reasoning_arm else vals
    usable, frac = _usable(cell["n_errors"] + void, cell["n_asked"])
    cell.update({
        "readout": "reasoning" if reasoning_arm else "sampled",
        "n_draws": len(ok), "n_void": void, "n_truncated": trunc,
        "n_reasoning": sum(reasoned), "parse_rate": (len(vals) / len(ok)) if ok else None,
        "reasoning_share": (sum(reasoned) / len(vals)) if vals else None,
        "ev": statistics.mean(reading) if reading else None,
        "ev_sd": statistics.pstdev(reading) if len(reading) > 1 else None,
        "ev_reasoning_only": statistics.mean(rv) if rv else None,
        "ev_all_draws": statistics.mean(vals) if vals else None,
        "mass_0_3": (sum(1 for v in vals if v <= 3) / len(vals)) if vals else None,
        "mass_0_3_source": f"share of {len(vals)} parsed draw(s) answering 0-3",
        "answer_mass": (1 - void / len(ok)) if ok else None,
        "answer_mass_source": "1 - void rate",
        "draw_values": vals, "draw_reasoned": reasoned, "draw_evs": reading,
        "usable": usable, "void_fraction": frac,
        "why_none": None if reading else _why(cell, "no reasoning draw parsed"
                                              if reasoning_arm else "no draw parsed")})
    return cell


def yes_no(records, readout, reasoning_arm=False):
    """P(yes): from the top-k on a log-prob readout, else the share of parsed yes."""
    ok, cell = _base(records)
    if readout == "logprob":
        ys, void = [], 0
        for r in ok:
            y, _ = API.yes_no_share([{"token": t["token"], "logprob": t["logprob"]}
                                     for t in r.get("top_tokens") or []])
            if y is None:
                void += 1
            else:
                ys.append(y)
        usable, frac = _usable(cell["n_errors"] + void, cell["n_asked"])
        cell.update({"readout": "logprob", "n_draws": len(ok), "n_void": void,
                     "p_yes": statistics.mean(ys) if ys else None,
                     "answer_mass": (1 - void / len(ok)) if ok else None,
                     "usable": usable, "void_fraction": frac,
                     "why_none": (None if ys
                                  else _why(cell, "no yes or no in any returned top-k"))})
        return cell
    vals, reasoned, void, trunc = [], [], 0, 0
    for r in ok:
        v = API.parse_yes_no(r.get("raw_text"))
        if v is None:
            void += 1
            trunc += API.truncated(r)
            continue
        vals.append(1.0 if v == "yes" else 0.0)
        reasoned.append(API.reasoned(r))
    rv = [v for v, x in zip(vals, reasoned) if x]
    reading = rv if reasoning_arm else vals
    usable, frac = _usable(cell["n_errors"] + void, cell["n_asked"])
    cell.update({"readout": "reasoning" if reasoning_arm else "sampled",
                 "n_draws": len(ok), "n_void": void, "n_truncated": trunc,
                 "n_reasoning": sum(reasoned),
                 "p_yes": statistics.mean(reading) if reading else None,
                 "p_yes_all_draws": statistics.mean(vals) if vals else None,
                 "answer_mass": (1 - void / len(ok)) if ok else None,
                 "usable": usable, "void_fraction": frac,
                 "why_none": None if reading else _why(cell, "no draw parsed")})
    return cell


def paired_scores(texts_or_records, label_first, label_second, order, reasoning_arm=False):
    """Both parties' scores from a comparison answer; the gap is professional minus
    private whichever was listed first."""
    if texts_or_records and isinstance(texts_or_records[0], dict):
        ok, cell = _base(texts_or_records)
        texts = [r.get("raw_text") for r in ok]
        flags = [API.reasoned(r) for r in ok]
        trunc_flags = [API.truncated(r) for r in ok]
    else:
        ok, cell = list(texts_or_records), {"n_asked": len(texts_or_records),
                                           "n_errors": 0}
        texts, flags, trunc_flags = ok, [False] * len(ok), [False] * len(ok)
    pro, priv, gaps, reas, void, trunc = [], [], [], [], 0, 0
    for t, f, tr in zip(texts, flags, trunc_flags):
        a, b = API.parse_pair(t, label_first, label_second)
        if a is None or b is None:
            void += 1
            trunc += tr
            continue
        p, q = (a, b) if order == "pro_first" else (b, a)
        pro.append(p)
        priv.append(q)
        gaps.append(p - q)
        reas.append(f)
    keep = [i for i, f in enumerate(reas) if f] if reasoning_arm else range(len(gaps))
    g = [gaps[i] for i in keep]
    usable, frac = _usable(cell["n_errors"] + void, cell["n_asked"])
    cell.update({"readout": "paired", "order": order, "n_draws": len(ok),
                 "n_void": void, "n_truncated": trunc, "n_reasoning": sum(reas),
                 "score_pro": statistics.mean([pro[i] for i in keep]) if g else None,
                 "score_priv": statistics.mean([priv[i] for i in keep]) if g else None,
                 "gap": statistics.mean(g) if g else None, "draw_gaps": g,
                 "usable": usable, "void_fraction": frac,
                 "why_none": None if g else _why(cell, "no pair of scores parsed")})
    return cell


def local_cell(read):
    """A teacher-forced local read is one deterministic reading of the full distribution."""
    if read.get("void"):
        return {"readout": "teacher_forced", "ev": None, "p_yes": None,
                "why_none": read.get("void_reason"), **read}
    probs = read["dist_norm"]
    out = {"readout": "teacher_forced", **read, "n_draws": 1, "n_void": 0,
           "usable": True}
    if read.get("ev") is not None:
        out["probs"] = probs
        out["mass_0_3"] = sum(probs[:4])
        out["mass_0_3_source"] = "sum of the answer distribution over 0-3"
    out["answer_mass_source"] = "next-token mass on the answers' first tokens"
    return out
