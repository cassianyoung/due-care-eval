# due-care-eval

Two tests of whether a language model applies the standard of reasonable care, packaged
as [Inspect](https://inspect.aisi.org.uk/) tasks. Each task runs on a deployed model
through an API or on a pretrained or instruction-tuned checkpoint on local weights, and
reports the headline figures below for that model.

* **The excuse test** (`due_care/excuse`). Six scenarios (a lent bicycle, a lent ladder,
  prawns served at home, prawns served in a restaurant, scaffolding, a cheque), each
  written in ten roles that differ in what the subject did and knew: N (no harm), C
  (careless), I (deliberate), P0 (the target case: the subject took care and the harm
  happened anyway), P1, P1-R, P1-neutral (as P0 with an excuse, a stated inability to
  discover the cause, or a neutral sentence), P2 and P2-other (the cause discoverable
  only by a specialist test), and BYSTANDER. Four questions: how much the subject is to
  blame (0 to 9), how much the subject caused the harm (0 to 9), whether the subject could
  reasonably have prevented it (yes or no), and whether the subject did anything careless
  (yes or no).
* **The duty test** (`due_care/duty`). Six pairs in which a professional and a private
  person make the same omission (prawns: chef and home cook; electrical work; gas
  safety; childcare; driving; hair dye). Each pair is read side by side in both orders,
  each case alone, each case alone with the convention stated (a professional owes a
  stricter duty), and each case alone with each of three neutral sentences.

The headline quantities:

* **Position** of a role on the scale from the no-harm case to the careless case,
  `(X - N) / (C - N)`. The target case's position is the blame the model assigns to a subject who did nothing wrong, in units of the blame it assigns for carelessness.
* **Gate.** A scenario enters a model's counts only if the model can read it: on the
  chat frame the no-harm case must carry at least 0.8 of its mass on 0 to 3, the answer
  mass must be at least 0.7, and the deliberate case must read at least 7; on the
  worksheet frame C - N must be at least 3 bands and the answer mass at least 0.7. The
  gate never looks at the target case. A scenario that fails enters no count, judgement
  questions included.
* **Band** (worksheet frame): 2.33 x the population sd over the five worksheet
  orderings, averaged over P0, P1 and BYSTANDER.
* **Stated gap** (duty test): professional minus private in the side-by-side answer,
  averaged over the two orders, with the spread between the orders and a flag where an
  order reads exactly zero. **Separate gap**: the same difference when each case is read
  alone. **Neutral baseline**: the separate gap with a neutral sentence, averaged over
  three sentences. **Wording floor**: 2.33 x the sd of the neutral gaps' deviations from
  the separate gap, over every pair read, floored at 0.30. **Stated-condition gap**: the
  separate gap with the convention stated, and whether it sits within, below or above
  the stated gap. **Signal-to-noise**: the stated gap over the wording floor.

Every formula is a SQL view in `db/`; the package computes nothing outside it.

## Install

The package needs **Python 3.11 or later**; check with `python3 --version`. Create an
environment and install the package into it:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .            # API models
pip install -e ".[local]"   # also local weights (torch, transformers, peft)
```

Install in editable mode (`-e`): the banks and the model table are read from this
directory.

**On Debian and Ubuntu** the system Python creates environments only once the
`python3-venv` package is installed (`apt-get install python3-venv`). **Where the system
Python is older than 3.11** (Debian 11 ships 3.9, for example), install a separate Python
with [uv](https://docs.astral.sh/uv/) rather than replacing the system's, and create the
environment with it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv python install 3.12
"$(uv python find 3.12)" -m venv .venv && . .venv/bin/activate
```

A Python installed by uv does not need `python3-venv`. For the dry run below, pass it as
`PYTHON="$(uv python find 3.12)" scripts/dry_run.sh`.

## A dry run on GPT-2

`scripts/dry_run.sh` runs the whole local path end to end on GPT-2 (124M), the one entry
in `models/`. It creates `.venv` if there is none and installs the package with the local
extras. It then downloads GPT-2 at its pinned revision and verifies every weight file
against the Hub's SHA-256. It reads both tests on the worksheet frame (300 excuse-test and
270 duty-test readings: each cell under the five worksheet orderings), builds the
database and prints the report. It checks the
result with `scripts/dry_run_check.py`, and finally runs the offline tests with GPT-2
cached, after downloading the tiny random Gemma-2 (about 90 MB) that the chat-template
test reads. A clean run ends with 25 checks passed, 5 of 5 test scripts passed and none
skipped. GPT-2 fails the gate on every scenario: it cannot tell the blameless case from the
careless one, which is a result about GPT-2, and the dry run checks the plumbing, not the
model.

```bash
scripts/dry_run.sh                      # writes dry_run_out/, on the CPU
scripts/dry_run.sh /tmp/dr cuda:float32 # another directory, on a CUDA device
PYTHON=python3.12 scripts/dry_run.sh    # choose the interpreter that creates .venv
```

It needs Python 3.11 or later, network access to the Hugging Face Hub on the first run,
and about 2 GB of disk (the environment, mostly torch, and 550 MB of weights). The two
reads take about 8 minutes each on an Apple M4 CPU; on an older x86 CPU the excuse test
alone took 69 minutes. Pass `cuda:float32` as SPEC where there is a
GPU. Both machines gave the same values to within 0.003.
If the output directory already exists, the script asks before replacing it (the
default is no, and it never replaces a directory holding anything a dry run does not
write). It holds:

| file | contents |
|---|---|
| `logs/*.eval` | the two Inspect logs, one per test |
| `fresh.sqlite` | the database built from them |
| `report.txt`, `report.json` | the report, as printed and as data |

The check tests the plumbing, not the model. Each log must be complete, with one scored
sample per cell. Every cell must hold a digit in [0, 9] or be void with a reason. The
determinism check must have passed. The weights must have been loaded once, at the
pinned revision, with every weight file verified. Every scored prompt must be one the
banks render. The database must hold every reading. GPT-2 is not expected to read the
scale, so its gate verdicts are printed and not checked. A failing check exits non-zero
and names what failed.

To dry-run a different model, add its entry to `models/` (see Adding a model) and run the
same two `inspect eval` commands the script runs, with `--model due_care/<model-id>`.

## Run

The model is named `due_care/<model-id>`, where `<model-id>` is an entry in the model
table: `models/` by default, or the directory named by `DUE_CARE_MODELS`.

**An API model on the chat frame** (OpenRouter; set `OPENROUTER_API_KEY`):

```bash
inspect eval due_care/excuse --model due_care/my_model -T frame=chat
inspect eval due_care/duty   --model due_care/my_model
```

**An instruction-tuned release on the chat frame, on local weights:**

```bash
inspect eval due_care/excuse --model due_care/my_instruct -T frame=chat
```

**A pretrained checkpoint, or a checkpoint with a LoRA adapter, on the worksheet frame:**

```bash
inspect eval due_care/excuse --model due_care/my_base -T frame=worksheet
inspect eval due_care/excuse --model due_care/my_base_with_adapter -T frame=worksheet
```

API calls stop once the reported spend of one process reaches `DUE_CARE_SPEND_CAP_USD`
(default 1.0); `DUE_CARE_API_CONCURRENCY` (default 4) bounds calls in flight. If
OpenRouter refuses the account (HTTP 401, 402 or 403: a wrong key, or no credit) the run
stops at the first call and says so. A call that fails for any other reason leaves its
cell unread, and the report lists the unread cells with their reasons.

Task options: `-T scenarios=bicycle,ladder`, `-T questions=BLAME,REASONABLE`,
`-T pairs=prawns,1`, `-T draws=N` (API models only; overrides the model table).

**Then the report:**

```bash
python -m due_care.report logs/<excuse>.eval logs/<duty>.eval
python -m db.build_from_readings fresh.sqlite logs/*.eval   # the full database
```

The report prints, per model: the target case's position in each scenario with its band
(worksheet) or its spread over draws (chat; descriptive only, not a band), the gate, the
prevention answer on the target case, the stated and separate gaps per duty pair, and the
wording floor. Every count is printed with its denominator.

### Hardware for local weights

Weights are read in bf16 on a CUDA device (`spec` in the model entry; `-M spec=cpu:float32`
overrides it). In bf16 a model needs about 2 GB of GPU memory per billion parameters, plus
headroom: a 24 GB card holds an 8B or 9B model, and a 27B to 34B model needs an 80 GB
card. The pinned revision is downloaded from the Hugging Face Hub (set `HF_TOKEN` for
gated repositories) and every weight file is checked against the publisher's SHA-256
before the model is loaded. An adapter is fetched and checked before the base weights are
downloaded (its digest against the model table, and the base its config names against
the entry's checkpoint), then merged after loading.

Before the first read on each frame, the same cell is read twice and the two answer
distributions must be identical; a model that fails stops before any reading is recorded.

## The two frames

**Chat**: a system message ("You are a rating instrument. Reply with a single digit from 0
to 9 and nothing else." or "Answer with one word, Yes or No, and nothing else.") and a
user turn holding the case and the question. On a local model the chat template is
applied with `enable_thinking=False` where the template accepts it; a template that
refuses a system role has the system text folded to the top of the user turn
(`system_role: fold` in the entry); anything else the template adds, such as a date
header, is left as served. The answer prefix the template expects is in the entry
(`chat_answer_prefix`); if it carries less than half the next-token mass, " ", "" and
"\n" are tried and the one used is recorded on every cell.

**Worksheet**: four worked examples with their ratings, then the case, ending at
"Rating:". The examples are read in each of five frozen orderings, and the band is the
spread over them. The answer distribution is read at the answer position by teacher
forcing. Used for pretrained checkpoints and adapters, which do not follow a chat
instruction.

## Readouts

| readout | used for | cell value |
|---|---|---|
| log-prob | API models that return log-probabilities | mean over draws of the expected digit under the returned top-k, renormalised over the digits; answer mass is the mean digit mass in the top-k, and k is recorded |
| sampled | API models without log-probabilities | mean parsed digit over draws; parse rate, reasoning share and truncation count recorded |
| reasoning | API models read with reasoning on | as sampled, over the draws that reasoned (reasoning text returned or reasoning tokens billed) |
| teacher-forced | local weights | the expected digit, or P(yes), under the full answer distribution |

Draws per cell come from the model entry (10 by default, 20 for a reasoning readout). A
cell that loses more than 10% of its draws to voids, truncation included, is reported
with its void count and is not used. A cell with no reading is a row with a reason, never
a default value. The provider and model that served every call are recorded, and duty
cells served by more than one provider are not combined into one contrast.

## What the instrument enforces

* Every prompt is rendered from frozen texts checked against their SHA-256, and every
  question names its subject; a text that no longer hashes, or a case naming no subject,
  is refused.
* Every API call declares its reasoning setting, pins one provider with fallbacks off,
  and records the provider and model that served it. No contrast combines cells served
  by two providers.
* Local weights are verified against the publisher's digests before loading, an adapter
  against its recorded digest and the base it names, and each frame passes the two-pass determinism check
  before any cell is read.
* The gate is computed from N, C, I and answer mass only, never from the target case.
* A cell with no reading, or more than 10% of its draws void, is a row with its reason,
  never a default value.

The temperature is part of each wire setting and is not enforced. A sampled mean
estimates the same quantity as a log-probability expected value only at temperature 1,
so set 1.0 wherever the two readouts are to be compared.

## Adding a model

Add `models/<model-id>.json`. An API entry:

```json
{"id": "my_model", "label": "My Model", "kind": "api", "config_set": "off",
 "wire": {"scenarios": {
   "wire": {"model": "vendor/my-model", "provider": "SomeProvider", "max_tokens": 1,
            "want_logprobs": true, "top_k": 20, "temperature": 1.0,
            "reasoning": {"enabled": false}},
   "readout": "logprob", "draws": 10}}}
```

The `wire` may hold `bicycle`, `scenarios`, `duty_separate` and `duty_comparison`
blocks; any that are absent fall back to `scenarios`. A sampled or reasoning readout sets
`want_logprobs` false and a `max_tokens` large enough for the answer to appear (256 or
more; a reasoning model needs several thousand).

**The `reasoning` field is required.** It is an explicit object sent as-is, or the string
`"vendor_default"`, which omits the field from the request so that the provider's default
applies. An entry without it is refused. A reading taken under `"vendor_default"` is
reproducible only while the vendor's default stays the same, and a change in that default
can be detected only by its effect on the readings; declare an explicit setting unless
the API offers none, and mark such an entry `"vendor_default": true`.

A local entry:

```json
{"id": "my_base", "kind": "local", "local_kind": "base",
 "checkpoint": {"repo": "org/model", "revision": "<40-character commit>"},
 "adapter": null, "frames": ["worksheet"], "spec": "cuda:bfloat16"}
```

An instruction-tuned release adds `"chat"` to `frames` and a `template` block
(`system_role`: `native` or `fold`; `chat_answer_prefix`).

An adapter adds an `"adapter"` block, from either source:

* **a Hub repository**, any account's: `{"repo": "org/my-adapter", "digest": "<16 hex>"}`,
  with `"subfolder"` when its files sit in a subfolder and `"revision"` to pin a commit.
  A private repository needs `HF_TOKEN`.
* **a local directory**, for an adapter you trained yourself and have not published:
  `{"path": "my_adapter", "digest": "<16 hex>"}`. The path is absolute, or relative to
  the directory the entry sits in; the digest is required, so a changed file is refused.

The directory, or the repository's files, hold `adapter_config.json` and
`adapter_model.safetensors` (or `adapter_model.bin`), as `peft`'s `save_pretrained`
writes them. The digest is the sha256 over the concatenated `adapter_config.json` and
adapter weights in sorted filename order, first 16 hex characters; no other file enters
it. Compute it with
`python -c "from harness.local import adapter_digest; print(adapter_digest('my_adapter'))"`.
If the config's `base_model_name_or_path` names a Hub repository other than the entry's
checkpoint, the adapter is refused; a local path there (a trainer's working copy) is
recorded and not compared. Every cell records the adapter's digest, its named base, and
its repository and commit or its path.

`models/gpt2.json` is a complete local entry. Validate a new entry by loading it:
`python -c "from due_care import model_table as MT; MT.load('my_model')"`.

## Layout

| path | contents |
|---|---|
| `banks/` | the frozen texts: cases, duty texts, questions, worksheet frame; every text with its SHA-256, and `MANIFEST.json` |
| `harness/` | the readers: `local.py` (weights, verification, teacher-forced scoring), `api.py` (the API client and its rules), `frames.py` (prompt rendering) |
| `due_care/` | the Inspect package: datasets, solvers, scorers, metrics, tasks, report |
| `db/` | the database schema and every view, and `build_from_readings.py` |
| `models/` | the model table: `gpt2.json`, the dry-run model |
| `scripts/` | `dry_run.sh` and `dry_run_check.py` |
| `tests/` | offline tests, `python tests/run_all.py`; `tests/fixtures/` holds the test-only model entries and the worksheet prompt hashes |

The question and worksheet bodies use `{agent}` as a placeholder, and the case rows use
`agent` as a field name. Both are part of the frozen texts, which are identified by their
SHA-256 digests, so they are left unchanged. Models never see the word: the placeholder
is filled with the subject's name.

## Licence

Code under Apache-2.0; the banks under CC BY 4.0; see
[`LICENSES.md`](LICENSES.md).
