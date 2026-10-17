"""Check a dry run's output directory: what scripts/dry_run.sh wrote is complete and
internally consistent.

    python scripts/dry_run_check.py OUT_DIR

It checks the plumbing, not the model: GPT-2 is not expected to pass the gate, and its
gate verdicts are printed rather than checked. Exits 1 if any check fails.
"""
import glob
import hashlib
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from inspect_ai.log import read_eval_log  # noqa: E402

from _t import Tally  # noqa: E402
from due_care import metrics as M  # noqa: E402
from due_care import model_table as MT  # noqa: E402
from due_care.datasets import duty_dataset, excuse_dataset  # noqa: E402

out = sys.argv[1]
T = Tally("dry_run_check")
entry = MT.load("gpt2")
fixture = set(json.load(open(os.path.join(ROOT, "tests", "fixtures",
                                          "worksheet_prompt_sha16.json")))["worksheet"])
# the fixture holds the v1.0 worksheet prompts; bank v1.1 adds the prawns pair's neutral
# sentences, whose worksheet prompts are rendered here from the banks
_v10 = {s.id for s in duty_dataset("worksheet", bank="1.0")}
fixture |= {hashlib.sha256(s.input.encode()).hexdigest()[:16]
            for s in duty_dataset("worksheet") if s.id not in _v10}
want = {"due_care/excuse": len(excuse_dataset("worksheet")),
        "due_care/duty": len(duty_dataset("worksheet"))}

logs = {}
for p in sorted(glob.glob(os.path.join(out, "logs", "*.eval"))):
    log = read_eval_log(p)
    logs.setdefault(log.eval.task, []).append(log)
T.check("one log per test (excuse, duty)",
        sorted(logs) == sorted(want) and all(len(v) == 1 for v in logs.values()),
        {k: len(v) for k, v in logs.items()})

n_read = {}
for task, n in want.items():
    log = (logs.get(task) or [None])[0]
    if log is None:
        T.check(f"{task}: log present", False)
        continue
    T.check(f"{task}: the run completed", log.status == "success", log.status)
    T.check(f"{task}: read on due_care/gpt2", log.eval.model == "due_care/gpt2",
            log.eval.model)
    samples = log.samples or []
    T.check(f"{task}: one sample per cell ({len(samples)} of {n})", len(samples) == n)
    cells = [s.scores["cell_score"].metadata for s in samples if s.scores]
    T.check(f"{task}: every sample was scored ({len(cells)} of {len(samples)})",
            len(cells) == len(samples))
    bad_value = [c["sample_id"] for c in cells
                 if not (c.get("void") or (c.get("ev") is not None and 0 <= c["ev"] <= 9))]
    T.check(f"{task}: every cell has a digit in [0, 9] or is void", not bad_value,
            bad_value[:4])
    no_reason = [c["sample_id"] for c in cells
                 if c.get("void") and not (c.get("void_reason") or c.get("why_none"))]
    T.check(f"{task}: every void cell says why", not no_reason, no_reason[:4])
    T.check(f"{task}: the determinism check passed on every cell",
            all((c.get("determinism") or {}).get("passed") for c in cells))
    loads = {json.dumps(c.get("load"), sort_keys=True) for c in cells}
    load = json.loads(next(iter(loads))) if len(loads) == 1 else {}
    T.check(f"{task}: one load, at the pinned revision",
            len(loads) == 1 and load.get("repo") == entry["checkpoint"]["repo"]
            and load.get("revision") == entry["checkpoint"]["revision"], loads)
    T.check(f"{task}: every weight file was verified against the Hub's digests",
            bool(load.get("weights_verified")), load.get("weights_verified"))
    off = [c["sample_id"] for c in cells if c.get("prompt_sha256_16") not in fixture]
    T.check(f"{task}: every scored prompt is a worksheet prompt the banks render",
            not off, off[:4])
    n_read[task] = len(cells)
    if task == "due_care/duty":
        n_read["duty_cells"] = len({(c.get("pair"), c.get("condition"), c.get("leg"),
                                     c.get("question")) for c in cells})

db = os.path.join(out, "fresh.sqlite")
if os.path.exists(db):
    con = sqlite3.connect(db)
    n_ex = con.execute("SELECT COUNT(*) FROM orderings").fetchone()[0]
    n_du = con.execute("SELECT COUNT(*) FROM duty_cells").fetchone()[0]
    T.check(f"the database holds every worksheet ordering read ({n_ex})",
            n_ex >= n_read.get("due_care/excuse", -1) > 0)
    T.check(f"the database holds one duty cell per cell read, over its orderings ({n_du})",
            n_du == n_read.get("duty_cells", -1))
    summ = M.excuse_summary(con)
    T.check(f"the views give a gate for every scenario read ({len(summ)})",
            len(summ) == 6 and all(v["gate"] for v in summ.values()))
    for (_, frame, scen), v in sorted(summ.items()):
        print(f"  gate  {frame} {scen}: {v['gate']} (a result about GPT-2, not checked)")
else:
    T.check("the database was built", False, db)

rep = os.path.join(out, "report.txt")
T.check("the report names the model and both tests",
        os.path.exists(rep) and "== gpt2" in open(rep).read()
        and "EXCUSE TEST" in open(rep).read() and "DUTY TEST" in open(rep).read())
T.done()
