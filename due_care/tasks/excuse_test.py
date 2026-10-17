"""The excuse test: ten roles on six scenarios, four questions.

    inspect eval due_care/excuse --model due_care/<model-id> -T frame=chat
    inspect eval due_care/excuse --model due_care/<model-id> -T frame=worksheet

On the chat frame every scenario is asked BLAME, CAUSE, REASONABLE and CARELESS
(REASONABLE and CARELESS are not asked of N and I). On the worksheet frame BLAME is read
under each of the five banked orderings. `scenarios` narrows the read, for example
-T scenarios=bicycle. `prompt_check` names a file of 16-hex prompt digests; every prompt
the read would send is checked against it before the task starts, and the read is
refused if any is missing.
"""
from inspect_ai import Task, task

from ..datasets import QUESTIONS, SCENARIOS, excuse_dataset, preflight
from ..metrics import excuse_headline
from ..scorers import cell_score
from ..solvers import chat_frame, worksheet_frame


def _split(x, default):
    if x is None:
        return default
    return tuple(x.split(",")) if isinstance(x, str) else tuple(x)


@task(name="excuse")
def excuse(frame: str = "chat", scenarios: str | None = None,
           questions: str | None = None, draws: int | None = None,
           prompt_check: str | None = None):
    if frame not in ("chat", "worksheet"):
        raise ValueError("frame is chat or worksheet")
    ds = excuse_dataset(frame, _split(scenarios, SCENARIOS), _split(questions, QUESTIONS))
    if prompt_check:
        preflight(ds, prompt_check)
    return Task(dataset=ds,
                solver=chat_frame(draws) if frame == "chat" else worksheet_frame(),
                scorer=cell_score(), metrics=[excuse_headline()], epochs=1,
                metadata={"test": "excuse", "frame": frame})
