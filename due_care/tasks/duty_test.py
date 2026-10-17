"""The duty test: six professional-against-private pairs.

    inspect eval due_care/duty --model due_care/<model-id> -T frame=chat

Each pair is read side by side in both orders, each case alone, each case alone with the
convention stated, and each case alone with each of three neutral sentences. The
side-by-side comparison asks for two digits in one answer and is read on the chat frame
only.

Bank v1.1 adds, for the prawns pair: its three neutral sentences; a side-by-side prompt
that sets out the two separate cases in full, as the other pairs' do (`joint_full`); a
reference case (`comparand`); and a duty CAUSE question (`cause`). The last three are
read on the chat frame. Where a model has read the full-text prompt, its stated gap
comes from it, and the v1.0 summary prompt is kept beside it (`duty_stated_gap_summary`).
Under `-T bank=1.0` prawns has none of these, as in the study's first reads.

`pairs` and `conditions` narrow the read, for example -T pairs=prawns,1 or
-T pairs=prawns -T conditions=neutral. A condition is a kind (separate, comparison,
stated, neutral, comparand, cause), a group (joint, joint_full) or one condition (bare,
cmp_pro_first, cmp_priv_first, cmp_full_pro_first, cmp_full_priv_first, stated,
neutral_1, neutral_2, neutral_3, comparand). `cells` reads exactly the named cells (a
comma-separated list of sample ids, or a JSON file holding a list); a cell the selection
does not render is refused. `prompt_check` names a file of 16-hex prompt digests; every
prompt the read would send is checked against it before the task starts, and the read is
refused if any is missing.
"""
from inspect_ai import Task, task

from harness import frames as F

from ..datasets import DUTY_PAIRS, duty_dataset, load_cells, preflight, select_cells
from ..metrics import duty_headline
from ..scorers import cell_score
from ..solvers import chat_frame, worksheet_frame


def _split(x):
    # Inspect parses -T pairs=1 as the integer 1
    if isinstance(x, (int, float)):
        x = str(x)
    return tuple(s.strip() for s in x.split(",")) if isinstance(x, str) else \
        tuple(str(s) for s in x)


@task(name="duty")
def duty(frame: str = "chat", pairs: str | None = None, draws: int | None = None,
         conditions: str | None = None, bank: str = F.BANK_LATEST,
         cells: str | None = None, prompt_check: str | None = None):
    if frame not in ("chat", "worksheet"):
        raise ValueError("frame is chat or worksheet")
    p = _split(pairs) if pairs else DUTY_PAIRS
    c = _split(conditions) if conditions else None
    ds = duty_dataset(frame, p, bank=str(bank), conditions=c)
    if cells:
        ds = select_cells(ds, load_cells(cells))
    if len(ds) == 0:
        raise ValueError(f"the selection pairs={p} conditions={c} on bank {bank} "
                         f"has no samples on the {frame} frame")
    if prompt_check:
        preflight(ds, prompt_check)
    return Task(dataset=ds,
                solver=chat_frame(draws) if frame == "chat" else worksheet_frame(),
                scorer=cell_score(), metrics=[duty_headline()], epochs=1,
                metadata={"test": "duty", "frame": frame, "bank_version": str(bank)})
