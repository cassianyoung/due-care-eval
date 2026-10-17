"""The excuse test: ten roles on six scenarios, four questions.

    inspect eval due_care/excuse --model due_care/<model-id> -T frame=chat
    inspect eval due_care/excuse --model due_care/<model-id> -T frame=worksheet

On the chat frame every scenario is asked BLAME, CAUSE, REASONABLE and CARELESS
(REASONABLE and CARELESS are not asked of N and I). On the worksheet frame BLAME is read
under each of the five banked orderings.

`scenarios`, `questions` and `roles` narrow the read, for example -T scenarios=bicycle
-T roles=P0,P1. The ten roles are read by default; SUPPLIER, which exists on the two food
scenarios, is read when named (-T roles=SUPPLIER) or with -T roles=all. `cells` reads
exactly the named cells: a comma-separated list of sample ids (for example
bicycle|C|REASONABLE|chat), or the path of a JSON file holding a list of them; a cell
the selection does not render is refused. `prompt_check` names a file of 16-hex prompt
digests; every prompt the read would send is checked against it before the task starts,
and the read is refused if any is missing.
"""
from inspect_ai import Task, task

from harness import frames as F

from ..datasets import (QUESTIONS, SCENARIOS, excuse_dataset, load_cells, preflight,
                        select_cells)
from ..metrics import excuse_headline
from ..scorers import cell_score
from ..solvers import chat_frame, worksheet_frame

ROLES = F.TABLE1_ROLES + ("SUPPLIER",)


def _split(x, default, known, name):
    if x is None:
        return default
    out = tuple(s.strip() for s in x.split(",")) if isinstance(x, str) else tuple(x)
    unknown = sorted(set(out) - set(known))
    if unknown:
        raise ValueError(f"unknown {name} {unknown}; one of {list(known)}")
    return out


@task(name="excuse")
def excuse(frame: str = "chat", scenarios: str | None = None,
           questions: str | None = None, draws: int | None = None,
           roles: str | None = None, cells: str | None = None,
           prompt_check: str | None = None):
    if frame not in ("chat", "worksheet"):
        raise ValueError("frame is chat or worksheet")
    r = ROLES if roles == "all" else _split(roles, F.TABLE1_ROLES, ROLES, "roles")
    ds = excuse_dataset(frame, _split(scenarios, SCENARIOS, SCENARIOS, "scenarios"),
                        _split(questions, QUESTIONS, QUESTIONS, "questions"),
                        roles=tuple(x for x in r if x != "SUPPLIER"),
                        include_supplier="SUPPLIER" in r)
    if cells:
        ds = select_cells(ds, load_cells(cells))
    if len(ds) == 0:
        raise ValueError(f"the selection has no samples on the {frame} frame")
    if prompt_check:
        preflight(ds, prompt_check)
    return Task(dataset=ds,
                solver=chat_frame(draws) if frame == "chat" else worksheet_frame(),
                scorer=cell_score(), metrics=[excuse_headline()], epochs=1,
                metadata={"test": "excuse", "frame": frame})
