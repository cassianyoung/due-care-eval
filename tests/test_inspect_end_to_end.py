"""The tasks run end to end through Inspect, offline: the API call is replaced by a stub
that answers from the prompt, and the log is turned into a database and a report. Checks
that what a solver asks is what reaches the request, that every draw is recorded, and
that the report is built from the log alone."""
import contextlib
import io
import math
import os
import tempfile

from _t import ROOT, Tally

import harness.api as API
from inspect_ai import eval as inspect_eval

import due_care._registry  # noqa: F401
from db import build_from_readings as B
from due_care import model_table as MT
from due_care import report as REP
from due_care.tasks.duty_test import duty
from due_care.tasks.excuse_test import excuse

T = Tally("test_inspect_end_to_end")

# ── THE METRICS TAKE THE CURRENT SIGNATURE (list[SampleScore]), SO NO RUN WARNS ──
from inspect_ai._eval.task.results import is_metric_deprecated  # noqa: E402
from due_care import metrics as METRICS  # noqa: E402

for name in ("excuse_headline", "duty_headline"):
    T.check(f"{name} takes list[SampleScore], not the deprecated list[Score]",
            not is_metric_deprecated(getattr(METRICS, name)()))
SENT = []


def stub(wire, messages, label="", key=None):
    SENT.append((wire, messages))
    user = messages[-1]["content"]
    if "Answer in exactly this form" in user:
        a, b = user.rsplit("form: ", 1)[1].replace("<digit>", "").split()
        text = f"{a}6 {b}3"
        top = []
    elif "yes or no" in user.lower():
        text, top = "No", [{"token": "No", "logprob": math.log(0.8)},
                           {"token": "Yes", "logprob": math.log(0.2)}]
    else:
        d = 0 if "Nobody came to any harm" in user else 8 if "intending" in user else 3
        text, top = str(d), [{"token": str(d), "logprob": math.log(0.9)},
                             {"token": str((d + 1) % 10), "logprob": math.log(0.1)}]
    return {"raw_text": text, "reasoning_text_len": 0, "finish_reason": "stop",
            "provider_served": wire["provider"], "model_served": wire["model"],
            "wire_sha": API.wire_sha(wire), "top_tokens": top,
            "reasoning_on_the_wire": wire["reasoning"], "usage": {}, "cost_usd": 0.0,
            "top_k": wire.get("top_k")}


API.call = stub
MT.MODELS = os.path.join(ROOT, "tests", "fixtures", "models")
with tempfile.TemporaryDirectory() as d:
    logs = inspect_eval(excuse(scenarios="bicycle"), model="due_care/example_api",
                        log_dir=d, display="none")
    log = logs[0]
    T.check("the excuse task completed", log.status == "success", log.status)
    n = len(log.samples or [])
    T.check(f"one sample per bicycle cell on the chat frame ({n})", n == 36)
    blame = [w for w, m in SENT if "to blame" in m[-1]["content"]]
    T.check("every BLAME call used the bicycle wire setting with 10 draws per cell",
            len(blame) == 100 and all(w["max_tokens"] == 1 and w["want_logprobs"]
                                      for w in blame), len(blame))
    inputs = {s.input for s in log.samples}
    T.check("every request's user turn is a sample's input, verbatim",
            all(m[-1]["content"] in inputs for _, m in SENT))
    sc = log.samples[0].scores["cell_score"].metadata
    T.check("every scored cell records its draws, providers and wire digest",
            sc["n_draws"] == 10 and sc["providers_served"] and sc["wire_sha"])
    p0 = [s for s in log.samples if s.id == "bicycle|P0|BLAME|chat"][0]
    T.check("the stub's distribution reaches the cell's expected digit",
            abs(p0.scores["cell_score"].value - 3.1) < 1e-9,
            p0.scores["cell_score"].value)
    SENT.clear()
    dlogs = inspect_eval(duty(pairs="prawns,1"), model="due_care/example_api", log_dir=d,
                         display="none")
    T.check("the duty task completed", dlogs[0].status == "success")
    cmp_ = [w for w, m in SENT if "Answer in exactly this form" in m[-1]["content"]]
    T.check("comparison calls used the duty_comparison wire setting",
            cmp_ and all(w["max_tokens"] == 256 and not w["want_logprobs"] for w in cmp_))
    paths = [logs[0].location, dlogs[0].location]
    con = B.build(B.cells_from_logs(paths), os.path.join(d, "fresh.sqlite"))
    n_cells = con.execute("SELECT COUNT(*) FROM cells c JOIN artefacts a ON a.id = "
                          "c.artefact_id WHERE a.path NOT LIKE '%:duty'").fetchone()[0]
    n_duty = con.execute("SELECT COUNT(*) FROM duty_cells").fetchone()[0]
    # prawns and pair 1: 6 + 12 cells under bank v1.0; v1.1 adds, for prawns, 6 neutral
    # cells, 2 full-text joint cells, 2 reference-case cells and 2 CAUSE cells
    T.check(f"the database is built from the logs alone ({n_cells} excuse, {n_duty} duty "
            f"cells)", n_cells == 36 and n_duty == 30)
    n_pn = con.execute("SELECT COUNT(*) FROM duty_cells WHERE pair_id = 'prawns' AND "
                       "condition LIKE 'neutral_%'").fetchone()[0]
    T.check(f"the prawns neutral cells are among them ({n_pn} of 6)", n_pn == 6)
    v10 = inspect_eval(duty(pairs="prawns,1", bank="1.0"), model="due_care/example_api",
                       log_dir=os.path.join(d, "v10"), display="none")
    con10 = B.build(B.cells_from_logs([v10[0].location]),
                    os.path.join(d, "fresh_v10.sqlite"))
    T.check("the same read under bank 1.0 builds the v1.0 18 duty cells, no prawns neutral "
            "text", con10.execute("SELECT COUNT(*) FROM duty_cells").fetchone()[0] == 18
            and con10.execute("SELECT COUNT(*) FROM duty_texts WHERE pair_id = 'prawns' "
                              "AND kind = 'neutral'").fetchone()[0] == 0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        REP.main(paths)
    txt = buf.getvalue()
    T.check("the report prints the bicycle row with its gate", "bicycle" in txt
            and "PASSED" in txt, txt[:400])
    T.check("the report prints the duty pairs", "prawns" in txt and "\n  1 " in txt, txt)
    print(txt)

    # ── A REFUSED ACCOUNT STOPS THE RUN; A CALL THAT FAILS IS A CELL WITH ITS REASON ──
    def refused(wire, messages, label="", key=None):
        raise API.AccountRefused("OpenRouter refused the request (HTTP 401). Check that "
                                 "OPENROUTER_API_KEY is a valid key")

    API.call = refused
    rlog = inspect_eval(excuse(scenarios="bicycle"), model="due_care/example_api",
                        log_dir=d, display="none")[0]
    T.check("a refused key stops the run: the log is an error, not a success",
            rlog.status == "error", rlog.status)
    T.check("and the error names the key",
            "OPENROUTER_API_KEY" in str(getattr(rlog.error, "message", rlog.error)))

    def flaky(wire, messages, label="", key=None):
        raise RuntimeError("call failed after 6 attempts: HTTP 503: busy")

    API.call = flaky
    flog = inspect_eval(excuse(scenarios="bicycle"), model="due_care/example_api",
                        log_dir=d, display="none")[0]
    T.check("calls that fail on the server leave the run complete", flog.status == "success",
            flog.status)
    why = {s.scores["cell_score"].metadata.get("why_none") for s in flog.samples}
    T.check("and every cell says that every call failed, with the error",
            all(w and w.startswith("every call failed: ") and "HTTP 503" in w for w in why),
            sorted(why)[:2])
    API.call = stub
T.done()
