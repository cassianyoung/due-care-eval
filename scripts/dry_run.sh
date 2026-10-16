#!/usr/bin/env bash
# The whole local path on GPT-2 (124M): install, download and verify the weights, read
# both tests on the worksheet frame, build the database, print the report, check the
# result, then run the offline tests with the model cached.
#
#     scripts/dry_run.sh [OUT_DIR] [SPEC]
#
# OUT_DIR defaults to dry_run_out/; if it already exists the script asks before
# replacing it (default no). SPEC defaults to the model entry's cpu:float32 (pass
# cuda:float32 or mps:float32 to read on a GPU). Set PYTHON to choose the interpreter
# that creates .venv (3.11 or later). Needs network access to the Hugging Face Hub for
# the first run; about 2 GB of disk for the environment and the weights. Each read takes
# about 8 minutes on an Apple M4 CPU; the excuse read took 69 minutes on an older x86 CPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/dry_run_out}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
SPEC="${2:-}"
PYTHON="${PYTHON:-python3}"
cd "$ROOT"

# ── an earlier run's output: ask, default no ─────────────────────────────────
if [ -e "$OUT" ]; then
  # only what a dry run writes may be replaced, so a mistyped path deletes nothing else
  other="$(find "$OUT" -mindepth 1 -maxdepth 1 ! -name logs ! -name fresh.sqlite \
           ! -name report.txt ! -name report.json -print -quit)"
  if [ -n "$other" ] || [ ! -d "$OUT" ]; then
    echo "refusing: $OUT exists and is not a dry run's output ($other); choose another OUT_DIR" >&2
    exit 1
  fi
  ans=""
  if [ -t 0 ]; then
    read -r -p "Dry run output exists at $OUT. Proceed and replace it? [y/N] " ans || true
  else
    echo "Dry run output exists at $OUT, and there is no terminal to ask: not replacing it." >&2
  fi
  case "$ans" in
    y|Y|yes|YES|Yes) rm -rf "$OUT" ;;
    *) echo "stopped: $OUT left as it was" >&2; exit 1 ;;
  esac
fi

echo "== 1. environment"
# the interpreter is checked before anything is created
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "no interpreter '$PYTHON'; set PYTHON to a Python 3.11 or later (see README, Install)" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
  cat >&2 <<EOF
$("$PYTHON" --version 2>&1) is too old: the package needs Python 3.11 or later.
Install a separate Python with uv and pass it in (see README, Install):
    curl -LsSf https://astral.sh/uv/install.sh | sh && source \$HOME/.local/bin/env
    uv python install 3.12
    PYTHON="\$(uv python find 3.12)" scripts/dry_run.sh
EOF
  exit 1
fi
# a .venv without a working pip is a failed earlier creation: replace it
if [ -e .venv ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  echo "removing .venv: it has no working pip (an earlier creation failed)"
  rm -rf .venv
fi
if [ ! -e .venv ]; then
  if ! "$PYTHON" -m venv .venv; then
    rm -rf .venv
    echo "could not create .venv with $PYTHON; on Debian or Ubuntu install python3-venv," \
         "or use a Python installed by uv (see README, Install)" >&2
    exit 1
  fi
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python - <<'EOF'
import sys
if sys.version_info < (3, 11):
    sys.exit(f".venv runs Python {sys.version.split()[0]}; delete .venv and run again "
             f"with PYTHON set to 3.11 or later")
EOF
pip install --quiet --upgrade pip
pip install --quiet -e ".[local]"
python -c "import inspect_ai, torch, transformers; print('inspect-ai', inspect_ai.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"

# the output directory is made only once the environment is ready
mkdir -p "$OUT/logs"

MARGS=()
if [ -n "$SPEC" ]; then
  MARGS=(-M "spec=$SPEC")
fi

echo "== 2. the excuse test on the worksheet frame (weights downloaded and verified first)"
inspect eval due_care/excuse --model due_care/gpt2 -T frame=worksheet \
  ${MARGS[@]+"${MARGS[@]}"} --log-dir "$OUT/logs" --display plain

echo "== 3. the duty test on the worksheet frame"
inspect eval due_care/duty --model due_care/gpt2 -T frame=worksheet \
  ${MARGS[@]+"${MARGS[@]}"} --log-dir "$OUT/logs" --display plain

echo "== 4. the database and the report"
python -m db.build_from_readings "$OUT/fresh.sqlite" "$OUT"/logs/*.eval
python -m due_care.report "$OUT"/logs/*.eval --json "$OUT/report.json" | tee "$OUT/report.txt"

echo "== 5. checking the dry run"
python scripts/dry_run_check.py "$OUT"

echo "== 6. the offline tests, with GPT-2 now cached"
# the chat-template test reads a tiny random Gemma-2 (about 90 MB, ungated), pinned
python - <<'EOF2'
from huggingface_hub import snapshot_download
snapshot_download("hf-internal-testing/tiny-random-Gemma2ForCausalLM",
                  revision="de7c11b6c25d26ddd1bf4324fcf479b61d18e440")
EOF2
HF_HUB_OFFLINE=1 python tests/run_all.py

echo "dry run complete: logs, database and report in $OUT"
