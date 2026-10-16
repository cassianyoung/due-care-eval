"""The two frames.

`chat_frame`: a system message and a user turn. On an API model the solver takes the
entry's draws for the cell's channel group and keeps every record; on a local model it
renders the model's chat template (with the template adjustments from the model table)
and reads the full answer distribution once, by teacher forcing. Comparison cells on a
local model are sampled, because the answer is two digits.

`worksheet_frame`: the four worked examples with their ratings followed by the case,
ending at "Rating:", read by teacher forcing on local weights only. Each of the five
banked orderings is its own sample.

Draws are taken inside the solver, not as Inspect epochs, so that the rules for void,
truncated and reasoning draws see every draw of a cell together. Run with epochs=1.
"""
from inspect_ai.model import get_model
from inspect_ai.solver import Generate, TaskState, solver

from harness import api as API

from . import model_table


def _api(state):
    api = get_model().api
    if not (hasattr(api, "entry") and hasattr(api, "api_draws")):
        raise TypeError("the due_care frames need a due_care/<model-id> model")
    return api


@solver
def chat_frame(draws: int | None = None):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        api = _api(state)
        md = state.metadata
        e = api.entry
        reading = {"frame": "chat", "model_id": e["id"], "kind": e["kind"]}
        if e["kind"] == "api":
            g, block = model_table.wire_group(e, md["group"])
            n = draws or block["draws"]
            _, block, recs = await api.api_draws(g, md["system"], state.input_text, n,
                                                 state.sample_id)
            reading.update({"group": g, "readout": block["readout"],
                            "wire": block["wire"],
                            "wire_sha": API.wire_sha(block["wire"]),
                            "draws": n, "records": recs})
        elif md.get("question") == "CMPN":
            n = draws or 10
            meta, texts = await api.local_sample(md["system"], state.input_text, n)
            reading.update({"readout": "paired_sampled", "draws": n, "texts": texts,
                            **meta})
        else:
            if "chat" not in e.get("frames", ["chat"]):
                reading.update({"readout": "teacher_forced", "local": {
                    "void": True, "void_reason": "this model is not read on the chat frame"}})
            else:
                prefix = (e.get("template") or {}).get("chat_answer_prefix", "")
                reading.update({"readout": "teacher_forced", "local": await api.local_read(
                    "chat", md["system"], state.input_text, md["answers"], prefix)})
        state.metadata["reading"] = reading
        return state

    return solve


@solver
def worksheet_frame():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        api = _api(state)
        md = state.metadata
        e = api.entry
        reading = {"frame": "worksheet", "model_id": e["id"], "kind": e["kind"],
                   "readout": "teacher_forced", "ordering": md.get("ordering")}
        if e["kind"] != "local":
            reading["local"] = {"void": True,
                                "void_reason": "the worksheet frame reads local weights only"}
        else:
            reading["local"] = await api.local_read(
                "worksheet", None, state.input_text, md["answers"], md["answer_prefix"])
        state.metadata["reading"] = reading
        return state

    return solve
