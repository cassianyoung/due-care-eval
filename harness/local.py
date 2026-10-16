"""The reader for local weights: load, verify, and score a cell by teacher forcing.

THE SCORER. Each of the answers ("0".."9", or "Yes"/"No") is scored by teacher-forced
log-probability over all of its own tokens, in one batched forward pass per cell. The
answer ids are taken in context: they are the difference between tokenising the prompt
and tokenising prompt + prefix + answer, which is exactly what the model is fed, and a
retokenisation that changes the prompt's own ids is refused.

`answer_mass` is the probability that the next token begins some answer, summed over the
distinct first tokens, so it is bounded by 1. `ev` is the expected digit under the
length-normalised answer distribution (`dist_norm`).

THE ANSWER PREFIX belongs to the frame and the template. The declared prefix is tried
first and kept if it carries at least PREFIX_OK of the mass; otherwise " ", "" and "\\n"
are tried and the one with the most mass is used. Both the declared and the used prefix
are recorded on every cell.

CHAT TEMPLATES are rendered with `enable_thinking=False` where the template accepts it.
A template that refuses a system role has the system text folded to the top of the user
turn, and the cell records `system_folded_into_user`. Whatever else a template adds (a
date header, for example) is left as the template serves it.
"""
import hashlib
import json
import os

import numpy as np
import torch

PREFIX_OK = 0.50
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
WEIGHT_SUFFIXES = (".safetensors", ".bin")


class Refused(RuntimeError):
    """A precondition of the read does not hold; nothing was scored."""


# ── WEIGHTS ──────────────────────────────────────────────────────────────────────
def hub_digests(repo_id, revision, token=None):
    """{filename: sha256} for the weight files of one Hub revision (LFS ids are sha256)."""
    from huggingface_hub import HfApi
    info = HfApi(token=token).model_info(repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise Refused(f"{repo_id}: asked for revision {revision}, the Hub resolved {info.sha}")
    return {s.rfilename: (s.lfs or {}).get("sha256") if s.lfs else None
            for s in info.siblings if s.rfilename.endswith(WEIGHT_SUFFIXES)}


# the files a pinned checkpoint is downloaded with; a cached copy is looked up with the same
FETCH_PATTERNS = ["*.json", "*.safetensors", "*.model", "tokenizer*", "*.txt", "*.jinja"]


def fetch_verified(repo_id, revision, token=None):
    """Download one pinned revision and check every weight file against the Hub digests.

    Returns (local_dir, {filename: sha256}). Refuses if any weight file is absent, has no
    published digest, or hashes differently.
    """
    from huggingface_hub import snapshot_download
    want = hub_digests(repo_id, revision, token)
    if not want:
        raise Refused(f"{repo_id}@{revision}: the Hub lists no weight files")
    path = snapshot_download(repo_id, revision=revision, token=token,
                             allow_patterns=FETCH_PATTERNS)
    bad = []
    for f, digest in sorted(want.items()):
        fp = os.path.join(path, f)
        if not os.path.exists(fp):
            if f.endswith(".bin") and any(k.endswith(".safetensors") for k in want):
                continue
            bad.append(f"absent {f}")
            continue
        if not digest:
            bad.append(f"no published digest for {f}")
            continue
        h = hashlib.sha256()
        with open(fp, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != digest:
            bad.append(f"sha256 mismatch {f}")
    if bad:
        raise Refused(f"{repo_id}@{revision} failed verification: {bad[:6]}")
    return path, want


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")


def adapter_digest(adapter_dir):
    """sha256 over the adapter's own files (adapter_config.json and its weights), sorted by
    name, truncated to 16 hex characters. Manifests and other files beside them are not
    part of the adapter and are not hashed."""
    h = hashlib.sha256()
    for f in sorted(os.listdir(adapter_dir)):
        if f in ADAPTER_FILES:
            with open(os.path.join(adapter_dir, f), "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def adapter_base(adapter_dir):
    """The base model the adapter's own config names (`base_model_name_or_path`), or None."""
    path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("base_model_name_or_path")


def _is_hub_id(name):
    """A Hub repository id ("org/name"), as opposed to a local path a trainer recorded."""
    return bool(name) and name.count("/") == 1 and not name.startswith((".", "/", "~"))


def check_adapter(adapter_dir, expect_digest=None, expect_base=None):
    """The adapter's digest and named base, refusing a digest other than the recorded one
    or a Hub base other than the one the entry pins. Cheap: it reads only the adapter."""
    digest = adapter_digest(adapter_dir)
    if expect_digest and digest != expect_digest:
        raise Refused(f"adapter digest {digest} != expected {expect_digest}")
    named = adapter_base(adapter_dir)
    if expect_base and _is_hub_id(named) and named != expect_base:
        raise Refused(f"the adapter was trained on {named}, not {expect_base}")
    return digest, named


def load(model_path, spec, adapter=None, expect_adapter_digest=None,
         trust_remote_code=False, expect_base=None):
    """Load weights at `spec` ("<device>:<dtype>"), merging a LoRA adapter if given.

    The device is part of the instrument and is never inferred: a spec asking for cuda
    on a machine without it is refused. The dtype is recorded from the loaded model.
    An adapter whose config names a Hub base other than `expect_base` is refused before
    any weights are loaded, so an adapter is never merged onto a model it was not
    trained on; a local path in that field (a trainer's working copy) is recorded, not
    compared.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev, _, dtn = spec.partition(":")
    if dev == "cuda" and not torch.cuda.is_available():
        raise Refused("the spec asks for cuda and it is not available")
    if dev == "mps" and not torch.backends.mps.is_available():
        raise Refused("the spec asks for mps and it is not available")
    digest = named_base = None
    if adapter:
        digest, named_base = check_adapter(adapter, expect_adapter_digest, expect_base)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=DTYPES[dtn],
                                                 trust_remote_code=trust_remote_code)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(dev)
    model.eval()
    return tok, model, {"device": dev, "dtype": dtn, "spec": spec,
                        "dtype_from_model": str(next(model.parameters()).dtype),
                        "adapter_digest": digest, "adapter_base": named_base}


def environment():
    """Library and device versions, recorded with every read."""
    import platform
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "cuda_available": torch.cuda.is_available()}
    for lib in ("transformers", "peft", "safetensors", "inspect_ai"):
        try:
            env[lib] = __import__(lib).__version__
        except Exception:
            env[lib] = None
    if torch.cuda.is_available():
        env["device_name"] = torch.cuda.get_device_name(0)
        env["cuda_build"] = torch.version.cuda
        try:
            import subprocess
            env["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20).stdout.strip() or None
        except Exception:
            env["driver"] = None
    return env


# ── SCORING ──────────────────────────────────────────────────────────────────────
def answer_ids(tok, answers, prefix, prompt):
    """Token ids for each answer in context, and the distinct first tokens."""
    base = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = []
    for a in answers:
        full = tok(f"{prompt}{prefix}{a}", add_special_tokens=False)["input_ids"]
        if full[:len(base)] != base:
            raise Refused(f"appending {prefix + a!r} retokenises the prompt")
        ids.append(full[len(base):])
    return ids, sorted({a[0] for a in ids})


def score(tok, model, prompt, answers, prefix, numeric=True):
    """Teacher-forced full-answer scoring for one cell, in one forward pass."""
    ans_ids, first_ids = answer_ids(tok, answers, prefix, prompt)
    dev = next(model.parameters()).device
    p_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    n_p = len(p_ids)
    seqs = [p_ids + a for a in ans_ids]
    width = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    batch = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, sq in enumerate(seqs):
        batch[i, :len(sq)] = torch.tensor(sq)
        mask[i, :len(sq)] = 1
    longest = max(len(a) for a in ans_ids)
    with torch.no_grad():
        logits = model(input_ids=batch.to(dev), attention_mask=mask.to(dev)).logits
        lp = torch.log_softmax(logits[:, n_p - 1: n_p + longest - 1, :].float(), -1).cpu()
        p0 = torch.softmax(logits[0, n_p - 1].float(), -1).cpu()
    raw, norm = [], []
    for i, a in enumerate(ans_ids):
        tot = float(sum(float(lp[i, k, t]) for k, t in enumerate(a)))
        raw.append(tot)
        norm.append(tot / len(a))
    d_raw = torch.softmax(torch.tensor(raw), -1).numpy()
    d_norm = torch.softmax(torch.tensor(norm), -1).numpy()
    ev = float(np.dot(d_norm, np.array([float(x) for x in answers]))) if numeric else None
    top = torch.topk(p0, 5)
    return {"dist_norm": d_norm.tolist(), "dist_raw": d_raw.tolist(), "ev": ev,
            "answer_mass": float(sum(float(p0[t]) for t in first_ids)),
            "entropy": float(-np.sum(d_raw * np.log(d_raw + 1e-12))),
            "n_prompt_tokens": n_p,
            "top5": [[tok.decode([int(i)]), float(p)]
                     for p, i in zip(top.values, top.indices)]}


def render_chat(tok, system, user):
    """(prompt, wrap) through the model's chat template; wrap names what was done."""
    if getattr(tok, "chat_template", None) in (None, ""):
        return None, "NO_CHAT_TEMPLATE"

    def apply(msgs):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    try:
        return apply(msgs), "chat"
    except Exception:
        folded = [{"role": "user", "content": ((system or "") + "\n\n" + user).strip()}]
        return apply(folded), "chat_system_folded"


def read_cell(tok, model, prompt, wrap, answers, declared_prefix):
    """Score one rendered prompt, resolving the answer prefix as described above."""
    numeric = answers[0] == "0"
    tried, best = {}, None
    for cand in [declared_prefix] + [x for x in (" ", "", "\n") if x != declared_prefix]:
        s = score(tok, model, prompt, answers, cand, numeric)
        tried[repr(cand)] = round(s["answer_mass"], 6)
        if best is None or s["answer_mass"] > best[1]["answer_mass"]:
            best = (cand, s)
        if s["answer_mass"] >= PREFIX_OK:
            break
    used, s = best
    return {"void": False, "wrap": wrap,
            "prompt_sha256_16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "system_folded_into_user": wrap == "chat_system_folded",
            "answer_prefix_declared": declared_prefix, "answer_prefix_used": used,
            "answer_mass_by_prefix": tried, "dist_norm": s["dist_norm"],
            "answer_mass": s["answer_mass"], "entropy": s["entropy"], "ev": s["ev"],
            "p_yes": (s["dist_norm"][answers.index("Yes")] if "Yes" in answers else None),
            "n_prompt_tokens": s["n_prompt_tokens"], "top5": s["top5"]}


def determinism_check(tok, model, prompt, wrap, answers, declared_prefix):
    """Two forward passes on the same cell must give identical distributions."""
    a = read_cell(tok, model, prompt, wrap, answers, declared_prefix)
    b = read_cell(tok, model, prompt, wrap, answers, declared_prefix)
    return {"passed": a["dist_norm"] == b["dist_norm"],
            "rule": "two forward passes, identical distributions",
            "prompt_sha256_16": a["prompt_sha256_16"]}
