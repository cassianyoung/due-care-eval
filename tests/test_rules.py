"""One or more checks for each carried-over rule, each refusal paired with a check that the
same gate opens on good input. No network, no GPU."""
import copy
import json
import os
import re
import tempfile

from _t import ROOT, Tally

from harness import api as API
from harness import frames as F
from due_care import datasets as D
from due_care import model_table as MT
from due_care import readouts as RO

T = Tally("test_rules")

# ── PROMPTS ARE RENDERED FROM FROZEN STRINGS, AND EVERY QUESTION NAMES ITS SUBJECT ───
T.opens("bank files match MANIFEST.json", F.manifest_check)
for frame in ("chat", "worksheet"):
    for ds in (D.excuse_dataset(frame), D.duty_dataset(frame)):
        bad_subject, bad_text = [], []
        texts = {r["sha256"]: r["text"] for r in F.load_jsonl("excuse_cases.jsonl")}
        texts.update({r["sha256"]: r["text"] for r in F.load_jsonl("duty_texts.jsonl")})
        for s in ds:
            q = s.input
            if s.metadata.get("question") == "CMPN":
                subject_ok = (s.metadata["label_first"] in q and
                              s.metadata["label_second"] in q)
            else:
                tail = q.split("Question:")[-1] if frame == "worksheet" else q[-160:]
                subject_ok = "Alaric" in tail
            if not subject_ok:
                bad_subject.append(s.id)
            if texts.get(s.metadata["text_sha256"]) not in q:
                bad_text.append(s.id)
        T.check(f"{ds.name}: every question names its subject ({len(ds)} samples)",
                not bad_subject, bad_subject[:3])
        T.check(f"{ds.name}: every prompt carries its bank text verbatim, by sha "
                f"({len(ds)} samples)", not bad_text, bad_text[:3])


def tampered_rows(row):
    r = dict(row)
    r["text"] = r["text"].replace("Alaric", "Alaric ", 1)
    return r


row = F.load_jsonl("excuse_cases.jsonl")[0]
T.raises("a bank text that no longer hashes to its sha is refused", F.BankError,
         F._checked, tampered_rows(row))
T.opens("the same row, unaltered, is accepted", F._checked, row)
T.raises("a case naming no subject is refused", F.BankError, F.subject_of, "Nobody did it.")
qs = F.load_json("excuse_questions.json")
T.raises("a prompt carrying two questions is refused", F.BankError, F.isolate, "BLAME",
         "x to blame, on a scale from y. Could Alaric reasonably have prevented this?",
         qs, "bicycle")

# ── DRAWS: 10 PER CELL BY DEFAULT, 20 FOR SAMPLED REASONING ARMS; FIVE ORDERINGS ────
ws = F.load_json("worksheet.json")
T.check("the worksheet has five banked orderings", len(ws["orderings"]) == 5,
        len(ws["orderings"]))
wsd = D.excuse_dataset("worksheet", ("bicycle",))
per_cell = {}
for s in wsd:
    per_cell.setdefault((s.metadata["role"]), set()).add(s.metadata["ordering"])
T.check("every worksheet cell is read under all five orderings "
        f"({len(per_cell)} cells)", all(len(v) == 5 for v in per_cell.values()))
T.check("default draws: 10 for log-prob and sampled readouts, 20 for reasoning",
        (MT.default_draws("logprob"), MT.default_draws("sampled"),
         MT.default_draws("reasoning")) == (10, 10, 20))
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "models")
SHIPPED = MT.MODELS
shipped = MT.all_entries()
T.check(f"the shipped model table loads and every entry validates ({len(shipped)} entries)",
        len(shipped) >= 1 and all(MT.validate(e) for e in shipped.values()))
MT.MODELS = FIXTURES
entries = MT.all_entries()
T.check(f"every API entry declares draws for every channel group "
        f"({sum(1 for e in entries.values() if e['kind'] == 'api')} entries)",
        all(b["draws"] >= 1 for e in entries.values() if e["kind"] == "api"
            for b in e["wire"].values() if b))

# ── THE REASONING FIELD IS ALWAYS DECLARED ─────────────────────────────────────────
good = {"model": "m", "provider": "p", "max_tokens": 1, "want_logprobs": True,
        "temperature": 1.0, "reasoning": {"enabled": False}, "top_k": 20}
msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
nofield = {k: v for k, v in good.items() if k != "reasoning"}
T.raises("a wire setting without `reasoning` is refused before any request",
         API.Refused, API.payload_for, nofield, msgs)
T.raises("a `reasoning` of None is refused (it must be declared)", API.Refused,
         API.payload_for, {**good, "reasoning": None}, msgs)
p = T.opens("an explicit reasoning object is accepted", API.payload_for, good, msgs)
T.check("an explicit reasoning object goes on the wire as declared",
        p is not None and p.get("reasoning") == {"enabled": False})
vd = T.opens("\"vendor_default\" is accepted", API.payload_for,
             {**good, "reasoning": "vendor_default"}, msgs)
T.check("\"vendor_default\" omits the field from the request",
        vd is not None and "reasoning" not in vd)
T.check("the request carries the user turn and the system message unchanged",
        p is not None and p["messages"] == msgs)
T.check("the request pins one provider with fallbacks off",
        p is not None and p["provider"] == {"order": ["p"], "allow_fallbacks": False})
T.check("every API entry in the model table passes the declaration check",
        all(MT.validate(e) for e in entries.values()))
vds = sorted(k for k, e in entries.items()
             if e["kind"] == "api" and e.get("vendor_default"))
T.check(f"an entry whose wire uses the vendor default is marked so ({', '.join(vds)})",
        vds == sorted(k for k, e in entries.items() if e["kind"] == "api" and any(
            b and b["wire"].get("reasoning") == "vendor_default" for b in e["wire"].values()))
        and vds)


# ── REASONING ARMS FROM REASONING DRAWS; MORE THAN 10% VOID IS NOT USABLE ─────────
def rec(text, reasoning_len=0, finish="stop", provider="P"):
    return {"raw_text": text, "reasoning_text_len": reasoning_len, "finish_reason": finish,
            "provider_served": provider, "model_served": "m", "usage": {}}


draws = [rec("7", 100)] * 6 + [rec("2", 0)] * 4
c = RO.sampled_digit(draws, reasoning_arm=True)
T.check("a reasoning arm's reading is the mean over its reasoning draws",
        c["ev"] == 7.0 and c["ev_all_draws"] == 5.0, (c["ev"], c["ev_all_draws"]))
c = RO.sampled_digit(draws, reasoning_arm=False)
T.check("a non-reasoning arm's reading is the mean over all parsed draws", c["ev"] == 5.0)
trunc = [rec("", 900, "length")] * 2 + [rec("3", 50)] * 8
c = RO.sampled_digit(trunc, reasoning_arm=True)
T.check("two truncated draws in ten (20%) make the cell not usable, with its count",
        c["usable"] is False and c["n_truncated"] == 2 and c["n_void"] == 2, c)
one = [rec("", 900, "length")] + [rec("3", 50)] * 9
c = RO.sampled_digit(one, reasoning_arm=True)
T.check("one truncated draw in ten (10%) leaves the cell usable", c["usable"] is True)
T.check("a draw that billed reasoning tokens without text counts as reasoning",
        API.reasoned({"reasoning_text_len": 0,
                      "usage": {"completion_tokens_details": {"reasoning_tokens": 40}}}))

# ── A MISSING OR REFUSED CELL IS A ROW WITH A REASON, NEVER A DEFAULT ─────────────
c = RO.sampled_digit([rec("I cannot rate that")] * 10, reasoning_arm=False)
T.check("a cell where nothing parses has no value and says why",
        c["ev"] is None and c["why_none"] and c["n_void"] == 10)
c = RO.logprob_digit([{"top_tokens": [{"token": "I", "logprob": -0.1}],
                       "provider_served": "P", "usage": {}}] * 10)
T.check("a log-prob cell with no digit in any top-k has no value and says why",
        c["ev"] is None and c["why_none"])

# ── PROVIDER RECORDED PER CALL ─────────────────────────────────────────────────────
mixed = [rec("4", provider="A")] * 5 + [rec("4", provider="B")] * 5
c = RO.sampled_digit(mixed, reasoning_arm=False)
T.check("a cell records every provider that served it and flags a change",
        c["providers_served"] == ["A", "B"] and c["provider_constant"] is False)

# ── THE MODEL TABLE: NEW MODELS ARE ENTRIES, NOT CODE ───────────────────────────
with tempfile.TemporaryDirectory() as d:
    e = copy.deepcopy(entries["example_api"])
    e["id"] = "new_arm"
    del e["wire"]["scenarios"]["wire"]["reasoning"]
    with open(os.path.join(d, "new_arm.json"), "w") as fh:
        json.dump(e, fh)
    old = MT.MODELS
    MT.MODELS = d
    T.raises("an entry whose wire omits `reasoning` is refused at load", API.Refused,
             MT.load, "new_arm")
    e["wire"]["scenarios"]["wire"]["reasoning"] = {"enabled": False}
    with open(os.path.join(d, "new_arm.json"), "w") as fh:
        json.dump(e, fh)
    T.opens("the same entry with `reasoning` declared loads", MT.load, "new_arm")
    MT.MODELS = old
MT.MODELS = SHIPPED

# ── WIRE SETTINGS CARRY NO KEYS OR ACCOUNT IDENTIFIERS ──────────────────────────
blob = json.dumps({**entries, **shipped})
T.check("the model table carries no key, token or account identifier",
        not re.search(r"(sk-[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{8,}|Bearer |api_key)", blob))

# ── THE ADAPTER DIGEST IS OVER THE ADAPTER'S OWN FILES ONLY ─────────────────────
from harness.local import adapter_digest
with tempfile.TemporaryDirectory() as d:
    open(os.path.join(d, "adapter_config.json"), "w").write('{"r": 32}')
    open(os.path.join(d, "adapter_model.safetensors"), "wb").write(b"weights")
    before = adapter_digest(d)
    open(os.path.join(d, "training_manifest.json"), "w").write('{"note": "x"}')
    open(os.path.join(d, "SEED_SCOPE.json"), "w").write("{}")
    T.check("a manifest beside the adapter does not change its digest",
            adapter_digest(d) == before)
    open(os.path.join(d, "adapter_config.json"), "w").write('{"r": 16}')
    T.check("a changed adapter_config.json changes the digest", adapter_digest(d) != before)

# ── AN ADAPTER COMES FROM ANY HUB REPOSITORY OR A LOCAL DIRECTORY ───────────────
import harness.local as _L  # noqa: E402
from due_care import provider as _P  # noqa: E402

local_entry = {"id": "own_adapter", "kind": "local", "local_kind": "adapter",
               "checkpoint": {"repo": "org/base", "revision": "a" * 40},
               "frames": ["worksheet"], "spec": "cpu:float32"}
T.raises("an adapter with neither `repo` nor `path` is refused", MT.EntryError,
         MT.validate, {**local_entry, "adapter": {"digest": "0" * 16}})
T.raises("an adapter with both `repo` and `path` is refused", MT.EntryError,
         MT.validate, {**local_entry, "adapter": {"repo": "o/a", "path": "a", "digest": "0" * 16}})
T.raises("a local adapter without a recorded digest is refused", MT.EntryError,
         MT.validate, {**local_entry, "adapter": {"path": "my_adapter"}})
T.opens("a local adapter with its digest is a valid entry", MT.validate,
        {**local_entry, "adapter": {"path": "my_adapter", "digest": "0" * 16}})


if True:
    with tempfile.TemporaryDirectory() as d:
        ad_dir = os.path.join(d, "my_adapter")
        os.mkdir(ad_dir)
        open(os.path.join(ad_dir, "adapter_config.json"), "w").write(
            '{"r": 4, "base_model_name_or_path": "org/base"}')
        open(os.path.join(ad_dir, "adapter_model.safetensors"), "wb").write(b"w")
        good = adapter_digest(ad_dir)
        old = MT.MODELS
        MT.MODELS = d  # a relative path is read against the entry's own directory
        try:
            def entry_with(**ad):
                return {**local_entry, "adapter": ad}
            T.raises("a local adapter directory that does not exist is refused",
                     _L.Refused, _P.resolve_adapter, entry_with(path="missing", digest=good))
            T.raises("a local adapter whose digest differs is refused", _L.Refused,
                     _P.resolve_adapter, entry_with(path="my_adapter", digest="0" * 16))
            got, meta = _P.resolve_adapter(entry_with(path="my_adapter", digest=good))
            T.check("a local adapter at its digest and on its base resolves, relative to "
                    "the entry's directory, and records its path",
                    got == ad_dir and meta == {"adapter_path": ad_dir}, got)
            open(os.path.join(ad_dir, "adapter_config.json"), "w").write(
                '{"r": 4, "base_model_name_or_path": "other/base"}')
            T.raises("a local adapter trained on another base is refused", _L.Refused,
                     _P.resolve_adapter,
                     entry_with(path="my_adapter", digest=adapter_digest(ad_dir)))
            T.opens("the same adapter on the base it names, given by absolute path, resolves",
                    _P.resolve_adapter,
                    entry_with(path=ad_dir, digest=adapter_digest(ad_dir)) | {
                        "checkpoint": {"repo": "other/base", "revision": "a" * 40}})
        finally:
            MT.MODELS = old

# ── AN ACCOUNT REFUSAL STOPS AT ONCE; A BUSY SERVER IS RETRIED ──────────────────────
import io as _io  # noqa: E402
import urllib.error as _ue  # noqa: E402
import urllib.request as _ur  # noqa: E402


def _answers(*codes):
    """A urlopen that answers with each HTTP code in turn, 200 meaning a reply."""
    seen = []

    class Reply(_io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        code = codes[len(seen)]
        seen.append(code)
        if code == 200:
            return Reply(b'{"ok": true}')
        raise _ue.HTTPError(req.full_url, code, "x", {}, _io.BytesIO(b'{"error": "x"}'))
    return urlopen, seen


real_urlopen, real_sleep = _ur.urlopen, API.time.sleep
API.time.sleep = lambda s: None
try:
    for code in (401, 402, 403):
        _ur.urlopen, seen = _answers(code, 200)
        T.raises(f"HTTP {code} refuses the account", API.AccountRefused, API._post,
                 "/x", {}, "k")
        T.check(f"HTTP {code}: one request, no retry", seen == [code], seen)
    _ur.urlopen, seen = _answers(404, 200)
    try:
        API._post("/x", {}, "k")
        msg = ""
    except API.AccountRefused:
        msg = "AccountRefused"
    except RuntimeError as e:
        msg = str(e)
    T.check("HTTP 404 fails the call, not the account, and counts its one attempt",
            "after 1 attempt:" in msg and seen == [404], (msg, seen))
    _ur.urlopen, seen = _answers(503, 503, 200)
    T.check("a busy server (HTTP 503) is retried until it answers",
            API._post("/x", {}, "k") == {"ok": True} and seen == [503, 503, 200], seen)
finally:
    _ur.urlopen, API.time.sleep = real_urlopen, real_sleep

err = {"error": "RuntimeError('call failed after 6 attempts: HTTP 503: busy')"}
T.check("a cell whose every call failed says so, with the error",
        RO.logprob_digit([err, err])["why_none"].startswith("every call failed: "))
ok = {"top_tokens": [{"token": "x", "logprob": 0.0}]}
T.check("a cell with replies but no digit keeps its own reason",
        RO.logprob_digit([err, ok])["why_none"] == "no digit token in any returned top-k")

T.done()
