"""The local-weights path, offline.

1. The worksheet prompts: every worksheet prompt the frames render from the frozen banks
   hashes to an entry in tests/fixtures/worksheet_prompt_sha16.json, and every entry is
   rendered. A change to harness/frames.py or to a bank that alters a prompt fails here.
2. The reader end to end on a small cached model (GPT-2 at the revision models/gpt2.json
   pins, which scripts/dry_run.sh downloads; a tiny random Gemma-2 on chat, whose template
   refuses a system role): the answer distribution sums
   to one, the expected digit lies in [0, 9], the determinism check runs before the first
   read and passes, and a determinism failure stops the read.

Run with HF_HUB_OFFLINE=1 so nothing is downloaded.
"""
import hashlib
import json
import os

from _t import ROOT, Tally

from due_care import model_table as MT
from due_care.datasets import duty_dataset, excuse_dataset

T = Tally("test_local")
WANT = set(json.load(open(os.path.join(ROOT, "tests", "fixtures",
                                       "worksheet_prompt_sha16.json")))["worksheet"])


def sha16(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ── 1. THE WORKSHEET PROMPTS ─────────────────────────────────────────────────────
# the fixture holds the v1.0 prompts; tests/test_bank_v1_1.py covers what v1.1 adds
got = {sha16(s.input) for ds in (excuse_dataset("worksheet"),
                                 duty_dataset("worksheet", bank="1.0"))
       for s in ds}
T.check(f"every rendered worksheet prompt is in the fixture ({len(got)} prompts)",
        got <= WANT, sorted(got - WANT)[:4])
T.check(f"every fixture entry is rendered ({len(WANT)} entries)", WANT <= got,
        sorted(WANT - got)[:4])

try:
    import torch  # noqa: F401
    from transformers import AutoTokenizer
except Exception as e:
    T.skip("the reader checks", f"torch/transformers not installed: {e}")
    T.done()

import harness.local as L  # noqa: E402

# ── 2. THE READER END TO END ON SMALL CACHED MODELS ───────────────────────────────
WS = json.load(open(os.path.join(ROOT, "banks", "worksheet.json")))
ws_samples = [s for s in excuse_dataset("worksheet", ("bicycle",)) if s.metadata["role"] == "P0"]
GPT2 = MT.load("gpt2")["checkpoint"]
try:
    from huggingface_hub import snapshot_download
    gpt2_path = snapshot_download(GPT2["repo"], revision=GPT2["revision"],
                                  allow_patterns=L.FETCH_PATTERNS, local_files_only=True)
    tok, model, meta = L.load(gpt2_path, "cpu:float32")
except Exception as e:
    T.skip("gpt2 worksheet read",
           f"{GPT2['repo']}@{GPT2['revision'][:12]} not cached (scripts/dry_run.sh "
           f"downloads it): {e}")
else:
    p = ws_samples[0].input
    det = L.determinism_check(tok, model, p, "raw", WS["answers"], WS["answer_prefix"])
    T.check("gpt2: two passes on one worksheet cell give identical distributions",
            det["passed"])
    c = L.read_cell(tok, model, p, "raw", WS["answers"], WS["answer_prefix"])
    T.check("gpt2: the worksheet answer distribution sums to one",
            abs(sum(c["dist_norm"]) - 1) < 1e-6)
    T.check("gpt2: the expected digit lies in [0, 9]", 0 <= c["ev"] <= 9, c["ev"])
    T.check("gpt2: answer mass lies in [0, 1] and the prefix is recorded",
            0 <= c["answer_mass"] <= 1 and c["answer_prefix_used"] is not None)
    T.check("gpt2: the scored prompt is the sample's input, by hash",
            c["prompt_sha256_16"] == sha16(p))
    T.raises("a retokenising answer is refused", L.Refused, L.answer_ids, tok,
             ["0"], "", p + " 1")

# a tiny random Gemma-2 (ungated; scripts/dry_run.sh downloads it at this revision); it
# ships no chat template, so it is given a minimal one in Gemma's turn format that, like
# Gemma 2's own, refuses a system role
TINY = ("hf-internal-testing/tiny-random-Gemma2ForCausalLM",
        "de7c11b6c25d26ddd1bf4324fcf479b61d18e440")
TURN_TEMPLATE = (
    "{% for m in messages %}{% if m['role'] == 'system' %}"
    "{{ raise_exception('System role not supported') }}{% endif %}"
    "<start_of_turn>{{ 'model' if m['role'] == 'assistant' else m['role'] }}\n"
    "{{ m['content'] }}<end_of_turn>\n{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")
try:
    from huggingface_hub import snapshot_download
    tiny_path = snapshot_download(TINY[0], revision=TINY[1], local_files_only=True)
    ttok = AutoTokenizer.from_pretrained(tiny_path)
    from transformers import AutoModelForCausalLM
    tmodel = AutoModelForCausalLM.from_pretrained(tiny_path)
    tmodel.eval()
except Exception as e:
    T.skip("tiny Gemma-2 chat read",
           f"{TINY[0]}@{TINY[1][:12]} not cached (scripts/dry_run.sh downloads it): {e}")
else:
    if not getattr(ttok, "chat_template", None):
        ttok.chat_template = TURN_TEMPLATE
    sys_ = "You are a rating instrument. Reply with a single digit from 0 to 9 and nothing else."
    p, wrap = L.render_chat(ttok, sys_, "Alaric lent a bicycle. How much is Alaric to blame?")
    T.check("a template that refuses a system role is folded, and says so",
            wrap == "chat_system_folded" and sys_ in p, wrap)
    c = L.read_cell(ttok, tmodel, p, wrap, [str(i) for i in range(10)], "")
    T.check("tiny Gemma-2: the chat read records the fold and a distribution",
            c["system_folded_into_user"] and abs(sum(c["dist_norm"]) - 1) < 1e-6)

    class Flaky:
        """Stands in for a model whose two passes differ."""
        calls = 0

    real = L.read_cell

    def flaky_read(*a, **k):
        Flaky.calls += 1
        out = real(*a, **k)
        if Flaky.calls % 2 == 0:
            out = dict(out, dist_norm=[x * 0.999 for x in out["dist_norm"]])
        return out

    L.read_cell = flaky_read
    det = L.determinism_check(ttok, tmodel, p, wrap, [str(i) for i in range(10)], "")
    L.read_cell = real
    T.check("a model whose two passes differ fails the determinism check",
            det["passed"] is False)

    # ── AN ADAPTER IS MERGED ONLY ONTO THE BASE IT NAMES, AT ITS RECORDED DIGEST ──
    # a LoRA built here on the tiny model, with non-zero weights so the merge shows
    import tempfile
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        T.skip("adapter merge", f"peft not installed: {e}")
    else:
        with tempfile.TemporaryDirectory() as d:
            torch.manual_seed(0)
            lora = get_peft_model(AutoModelForCausalLM.from_pretrained(tiny_path),
                                  LoraConfig(r=4, target_modules=["q_proj", "v_proj"],
                                             init_lora_weights=False))
            lora.save_pretrained(d)
            cfg_path = os.path.join(d, "adapter_config.json")

            def name_base(name):
                cfg = json.load(open(cfg_path))
                cfg["base_model_name_or_path"] = name
                json.dump(cfg, open(cfg_path, "w"))
                return L.adapter_digest(d)

            dg = name_base(TINY[0])
            words = ttok("Alaric lent a bicycle.", return_tensors="pt")
            with torch.no_grad():
                base_logits = tmodel(**words).logits
            T.opens("an adapter at its digest, naming its base, is merged", L.load,
                    tiny_path, "cpu:float32", adapter=d, expect_adapter_digest=dg,
                    expect_base=TINY[0])
            _, merged, meta = L.load(tiny_path, "cpu:float32", adapter=d,
                                     expect_adapter_digest=dg, expect_base=TINY[0])
            with torch.no_grad():
                moved = (merged(**words).logits - base_logits).abs().max().item()
            T.check("the merged adapter changes the model's output, and the load records "
                    "its digest and base",
                    moved > 1e-4 and meta["adapter_digest"] == dg
                    and meta["adapter_base"] == TINY[0], f"max change {moved:.2e}")
            T.raises("an adapter whose digest differs from the recorded one is refused",
                     L.Refused, L.load, tiny_path, "cpu:float32", adapter=d,
                     expect_adapter_digest="0" * 16, expect_base=TINY[0])
            dg2 = name_base("other-org/other-model")
            T.raises("an adapter naming another Hub base is refused", L.Refused, L.load,
                     tiny_path, "cpu:float32", adapter=d, expect_adapter_digest=dg2,
                     expect_base=TINY[0])
            dg3 = name_base("/workspace/models/tiny")
            T.opens("an adapter naming a local path is merged (the path is recorded)",
                    L.load, tiny_path, "cpu:float32", adapter=d,
                    expect_adapter_digest=dg3, expect_base=TINY[0])
T.done()
