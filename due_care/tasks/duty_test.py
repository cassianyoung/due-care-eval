"""The duty test: six professional-against-private pairs.

    inspect eval due_care/duty --model due_care/<model-id> -T frame=chat

Each pair is read side by side in both orders, each case alone, each case alone with the
convention stated, and each case alone with each of three neutral sentences (prawns has
no neutral sentences). The side-by-side comparison asks for two digits in one answer and
is read on the chat frame only. `pairs` narrows the read, for example -T pairs=prawns,1.
"""
from inspect_ai import Task, task

from ..datasets import DUTY_PAIRS, duty_dataset
from ..metrics import duty_headline
from ..scorers import cell_score
from ..solvers import chat_frame, worksheet_frame


@task(name="duty")
def duty(frame: str = "chat", pairs: str | None = None, draws: int | None = None):
    if frame not in ("chat", "worksheet"):
        raise ValueError("frame is chat or worksheet")
    p = tuple(pairs.split(",")) if isinstance(pairs, str) else (pairs or DUTY_PAIRS)
    ds = duty_dataset(frame, p)
    return Task(dataset=ds,
                solver=chat_frame(draws) if frame == "chat" else worksheet_frame(),
                scorer=cell_score(), metrics=[duty_headline()], epochs=1,
                metadata={"test": "duty", "frame": frame})
