"""The model table: one JSON entry per model, in `models/`.

An API entry carries one wire setting per channel group (`bicycle`, `scenarios`,
`duty_separate`, `duty_comparison`), each with its readout and draws per cell. A local
entry carries the checkpoint, its pinned Hub revision, an optional adapter, the frames it
is read on, and its chat-template adjustments. New models are added as entries, not code.
"""
import json
import os

from harness import api as API

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("DUE_CARE_MODELS") or os.path.join(ROOT, "models")
GROUPS = ("bicycle", "scenarios", "duty_separate", "duty_comparison")
READOUTS = ("logprob", "sampled", "reasoning")


class EntryError(ValueError):
    pass


def default_draws(readout):
    """Draws per cell for a new entry that does not state them: 20 on a reasoning
    readout, 10 otherwise."""
    return 20 if readout == "reasoning" else 10


def validate(e):
    if e.get("kind") not in ("api", "local"):
        raise EntryError(f"{e.get('id')}: kind must be api or local")
    if e["kind"] == "api":
        for g, block in (e.get("wire") or {}).items():
            if g not in GROUPS:
                raise EntryError(f"{e['id']}: unknown channel group {g}")
            if not block:
                continue
            API.check_wire(block["wire"])
            if block.get("readout") not in READOUTS:
                raise EntryError(f"{e['id']}/{g}: readout must be one of {READOUTS}")
            block.setdefault("draws", default_draws(block["readout"]))
            if not isinstance(block.get("draws"), int) or block["draws"] < 1:
                raise EntryError(f"{e['id']}/{g}: draws must be a positive integer")
            if block["wire"]["want_logprobs"] != (block["readout"] == "logprob"):
                raise EntryError(f"{e['id']}/{g}: a log-prob readout requests log-probs "
                                 f"and no other readout does")
    else:
        ck = e.get("checkpoint") or {}
        if not ck.get("repo") or not ck.get("revision"):
            raise EntryError(f"{e['id']}: a local entry pins a repo and a revision")
        if len(ck["revision"]) != 40:
            raise EntryError(f"{e['id']}: the revision is a full 40-character commit")
        ad = e.get("adapter")
        if ad is not None:
            if bool(ad.get("repo")) == bool(ad.get("path")):
                raise EntryError(f"{e['id']}: an adapter gives a Hub `repo` or a local "
                                 f"`path`, not both and not neither")
            if ad.get("path") and not ad.get("digest"):
                raise EntryError(f"{e['id']}: a local adapter records its `digest`, so a "
                                 f"changed file on disk is refused")
    return e


def load(model_id):
    path = os.path.join(MODELS, f"{model_id}.json")
    if not os.path.exists(path):
        raise EntryError(f"no entry models/{model_id}.json")
    with open(path, encoding="utf-8") as fh:
        e = json.load(fh)
    if e.get("id") != model_id:
        raise EntryError(f"models/{model_id}.json carries id {e.get('id')!r}")
    return validate(e)


def all_entries():
    return {f[:-5]: load(f[:-5]) for f in sorted(os.listdir(MODELS))
            if f.endswith(".json")}


def wire_group(entry, group):
    """The wire block for a channel group, falling back from bicycle to scenarios and
    from duty_separate to scenarios when a new entry declares only one setting."""
    w = entry.get("wire") or {}
    for g in (group, "scenarios"):
        if w.get(g):
            return g, w[g]
    raise EntryError(f"{entry['id']}: no wire setting for {group}")


def local_adapter_dir(entry):
    """A local adapter's directory: an absolute path, or one relative to the directory
    the model entry sits in."""
    p = os.path.expanduser(entry["adapter"]["path"])
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(MODELS, p))
