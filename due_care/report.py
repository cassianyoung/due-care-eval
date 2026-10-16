"""Print the headline figures for the models in one or more Inspect logs.

    python -m due_care.report LOG [LOG ...] [--db OUT.sqlite] [--json OUT.json]

For each model: the target case's position per scenario with its band (worksheet) or its
spread over draws (chat, descriptive only) and its gate; the prevention answer on the
target case; the stated and given slack per duty pair; and the wording floor.

A scenario that fails its gate is shown, marked, and left out of every count.
"""
import argparse
import collections
import json
import math
import statistics

from db import build_from_readings as B

from . import metrics as M

PASS = ("PASSED", "LICENSED")


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:+.{nd}f}" if isinstance(x, float) and x < 0 else f"{x:.{nd}f}"


def render(con):
    lines = []
    exc = M.excuse_summary(con)
    duty = M.duty_summary(con)
    models = sorted({k[0] for k in exc} | {k[0] for k in duty})
    for m in models:
        lines.append(f"== {m}")
        rows = {k: v for k, v in exc.items() if k[0] == m}
        if rows:
            lines.append("  EXCUSE TEST: the target case (P0) on the N-to-C scale")
            lines.append(f"  {'frame':<9} {'scenario':<18} {'P0':>6} {'P1':>6} {'BYST':>6} "
                         f"{'band':>6} {'gate':<9} {'prevent':>7}")
            for (_, frame, scen), v in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][2])):
                spread = v["band"] if frame == "worksheet" else v["draw_sd"].get("P0")
                lines.append(
                    f"  {frame:<9} {scen or '-':<18} {_f(v['positions'].get('P0')):>6} "
                    f"{_f(v['positions'].get('P1')):>6} "
                    f"{_f(v['positions'].get('BYSTANDER')):>6} {_f(spread):>6} "
                    f"{(v['gate'] or 'UNREAD'):<9} {_f(v.get('P0_reasonable_p_yes')):>7}")
            passing = [k for k, v in rows.items() if v["gate"] in PASS]
            lines.append(f"  gate: {len(passing)} of {len(rows)} scenario readings pass; "
                         f"the rest enter no count below")
            p0s = [rows[k]["positions"]["P0"] for k in passing
                   if rows[k]["positions"].get("P0") is not None]
            if p0s:
                lines.append(f"  target case position, gate-passing scenarios: median "
                             f"{statistics.median(p0s):.2f} over {len(p0s)}")
            prev = [rows[k]["P0_reasonable_p_yes"] for k in passing
                    if rows[k].get("P0_reasonable_p_yes") is not None]
            if prev:
                lines.append(f"  prevention ('could reasonably have prevented this?') on the "
                             f"target case: yes on {sum(1 for p in prev if p > 0.5)} of "
                             f"{len(prev)} gate-passing scenarios")
            lines.append("  band = 2.33 x sd over the five worksheet orderings; on chat the "
                         "column is the sd over draws, descriptive only, not a band")
        drows = {k: v for k, v in duty.items() if k[0] == m}
        if drows:
            lines.append("  DUTY TEST: professional minus private, in digits")
            lines.append(f"  {'pair':<7} {'stated':>7} {'spread':>7} {'zero':>5} "
                         f"{'given':>7} {'neutral':>8} {'w/conv':>7} {'class':<7} "
                         f"{'S/N':>6}")
            for (_, pid), v in sorted(drows.items()):
                lines.append(
                    f"  {pid:<7} {_f(v.get('stated_gap')):>7} {_f(v.get('order_spread')):>7} "
                    f"{str(v.get('exact_zero_in_an_order', '-')):>5} "
                    f"{_f(v.get('separate_gap')):>7} {_f(v.get('neutral_baseline')):>8} "
                    f"{_f(v.get('stated_condition_gap')):>7} "
                    f"{(v.get('stated_condition_vs_stated') or '-'):<7} "
                    f"{_f(v.get('signal_to_noise')):>6}")
            fl = next(iter(drows.values()))
            lines.append(f"  wording floor (floor_used): {_f(fl.get('floor_used'))} "
                         f"({fl.get('floor_used_source') or 'none'})")
            lines.append("  stated = the comparison's gap (two-order mean); given = the gap "
                         "between the two cases read separately; the floor is pooled over "
                         "every pair read, so a partial read has a different floor")
    return "\n".join(lines)


def unread(cells):
    """Cells with no usable reading, per model, with their reasons: a run in which nothing
    was read still says so, and why."""
    lines = []
    for m in sorted({c.get("model_id") for c in cells}, key=str):
        mine = [c for c in cells if c.get("model_id") == m]
        bad = [c for c in mine if c.get("usable") is False or c.get("void")
               or (c.get("why_none") and c.get("ev") is None and c.get("p_yes") is None
                   and c.get("gap") is None)]
        if not bad:
            continue
        lines.append(f"== {m}: {len(bad)} of {len(mine)} cells have no usable reading")
        why = collections.Counter(str(c.get("why_none") or c.get("void_reason")
                                      or "over the void limit")[:160] for c in bad)
        for reason, n in why.most_common(3):
            lines.append(f"  {n:>5}  {reason}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--db", default=":memory:")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    cells = list(B.cells_from_logs(a.logs))
    con = B.build(cells, a.db)
    print(render(con))
    u = unread(cells)
    if u:
        print(u)
    if a.json:
        out = {"excuse": {"|".join(map(str, k)): v for k, v in M.excuse_summary(con).items()},
               "duty": {"|".join(map(str, k)): v for k, v in M.duty_summary(con).items()}}
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
