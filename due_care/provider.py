"""The `due_care/<model-id>` model provider for Inspect.

The provider resolves `<model-id>` in the model table. For an API entry it sends calls
through `harness.api` with the entry's wire setting for the channel group; for a local
entry it downloads the pinned revision, verifies every weight file against the
publisher's digests, merges the adapter if one is listed (checking its digest), and
reads cells through `harness.local`.

The solvers call `api_draws`, `local_read` and `local_sample` directly, because a cell
reading needs the per-draw records and the full answer distribution that a plain
`generate()` does not return. `generate()` is implemented so that the provider also works
with any ordinary Inspect solver.

Before the first read on a local model, and once per frame, the same cell is read twice
and the two distributions must be identical; a model that fails stops before any
reading is recorded.
"""
import asyncio
import os
import threading

from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, modelapi

from harness import api as API
from harness import local as LOCAL

from . import model_table

API_CONCURRENCY = int(os.environ.get("DUE_CARE_API_CONCURRENCY", "4"))
# calls stop once the reported spend of this process reaches the cap (US dollars)
SPEND_CAP = float(os.environ.get("DUE_CARE_SPEND_CAP_USD", "1.0"))
_SPENT = {"usd": 0.0, "calls": 0}
_SPENT_LOCK = threading.Lock()


def resolve_adapter(entry, token=None):
    """The entry's adapter directory and its provenance, fetched and checked before any
    base weights are downloaded, so a refused, missing or mismatched adapter surfaces
    first. An adapter comes from any Hub repository (`repo`, with `subfolder` and
    `revision` optional; HF_TOKEN for a private one) or a local directory (`path`)."""
    ad = entry.get("adapter")
    if not ad:
        return None, {}
    if ad.get("path"):
        adapter_dir = model_table.local_adapter_dir(entry)
        if not os.path.isdir(adapter_dir):
            raise LOCAL.Refused(f"no adapter directory at {adapter_dir}")
        meta = {"adapter_path": adapter_dir}
    else:
        from huggingface_hub import snapshot_download
        sub = ad.get("subfolder")
        pre = f"{sub}/" if sub else ""
        # only the files the adapter is loaded from: the recorded digest is over these,
        # and a repository may also hold manifests and card files
        root = snapshot_download(ad["repo"], revision=ad.get("revision"), token=token,
                                 allow_patterns=[f"{pre}adapter_config.json",
                                                 f"{pre}adapter_model.safetensors",
                                                 f"{pre}adapter_model.bin"])
        adapter_dir = os.path.join(root, sub) if sub else root
        # the commit that served is the snapshot directory's name
        meta = {"adapter_repo": ad["repo"], "adapter_revision": os.path.basename(root)}
    LOCAL.check_adapter(adapter_dir, ad.get("digest"), entry["checkpoint"]["repo"])
    return adapter_dir, meta


@modelapi(name="due_care")
class DueCareAPI(ModelAPI):
    def __init__(self, model_name, base_url=None, api_key=None,
                 config=GenerateConfig(), **model_args):
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key,
                         config=config)
        self.entry = model_table.load(model_name)
        self.spec = model_args.get("spec") or self.entry.get("spec") or "cuda:bfloat16"
        self._tok = self._model = None
        self._load_meta = None
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(API_CONCURRENCY)
        self.determinism = {}

    # ── API ENTRIES ────────────────────────────────────────────────────────────
    async def api_draws(self, group, system, user, n, label):
        g, block = model_table.wire_group(self.entry, group)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def one(i):
            with self._slots:
                with _SPENT_LOCK:
                    if _SPENT["usd"] >= SPEND_CAP:
                        return {"error": f"spend cap of ${SPEND_CAP} reached after "
                                         f"{_SPENT['calls']} calls; nothing sent",
                                "wire_sha": API.wire_sha(block["wire"])}
                try:
                    rec = API.call(block["wire"], msgs, f"{label}|d{i + 1}")
                    with _SPENT_LOCK:
                        _SPENT["usd"] += rec.get("cost_usd") or 0.0
                        _SPENT["calls"] += 1
                    return rec
                except API.AccountRefused:
                    raise
                except Exception as exc:
                    return {"error": repr(exc)[:300], "wire_sha": API.wire_sha(block["wire"])}

        recs = await asyncio.gather(*[asyncio.to_thread(one, i) for i in range(n)])
        return g, block, list(recs)

    # ── LOCAL ENTRIES ──────────────────────────────────────────────────────────
    def _ensure_loaded(self):
        with self._lock:
            if self._model is not None:
                return
            ck = self.entry["checkpoint"]
            token = os.environ.get("HF_TOKEN")
            ad = self.entry.get("adapter")
            adapter_dir, adapter_meta = resolve_adapter(self.entry, token)
            path, digests = LOCAL.fetch_verified(ck["repo"], ck["revision"], token)
            self._tok, self._model, meta = LOCAL.load(
                path, self.spec, adapter=adapter_dir,
                expect_adapter_digest=(ad or {}).get("digest"), expect_base=ck["repo"])
            self._load_meta = {**meta, **adapter_meta, "repo": ck["repo"],
                               "revision": ck["revision"],
                               "weights_verified": sorted(digests),
                               "environment": LOCAL.environment()}

    def _render(self, frame, system, text):
        if frame == "worksheet":
            return text, "raw"
        tmpl = self.entry.get("template") or {}
        if tmpl.get("system_role") == "fold":
            folded = ((system or "") + "\n\n" + text).strip()
            p, wrap = LOCAL.render_chat(self._tok, None, folded)
            return p, ("chat_system_folded" if p else wrap)
        return LOCAL.render_chat(self._tok, system, text)

    def _local_read_sync(self, frame, system, text, answers, prefix):
        self._ensure_loaded()
        prompt, wrap = self._render(frame, system, text)
        if prompt is None:
            return {"void": True, "void_reason": wrap}
        with self._lock:
            if frame not in self.determinism:
                self.determinism[frame] = LOCAL.determinism_check(
                    self._tok, self._model, prompt, wrap, answers, prefix)
            if not self.determinism[frame]["passed"]:
                raise LOCAL.Refused(f"determinism check failed on the {frame} frame")
            cell = LOCAL.read_cell(self._tok, self._model, prompt, wrap, answers, prefix)
        cell["determinism"] = self.determinism[frame]
        cell["load"] = {k: v for k, v in self._load_meta.items() if k != "environment"}
        return cell

    async def local_read(self, frame, system, text, answers, prefix):
        return await asyncio.to_thread(self._local_read_sync, frame, system, text,
                                       answers, prefix)

    def _local_sample_sync(self, system, text, n, max_new_tokens, seed):
        import torch
        self._ensure_loaded()
        prompt, wrap = self._render("chat", system, text)
        if prompt is None:
            return {"void": True, "void_reason": wrap}, []
        with self._lock:
            ids = self._tok(prompt, return_tensors="pt").to(self._model.device)
            torch.manual_seed(seed)
            out = self._model.generate(**ids, do_sample=True, temperature=1.0,
                                       max_new_tokens=max_new_tokens,
                                       num_return_sequences=n,
                                       pad_token_id=self._tok.eos_token_id)
        texts = [self._tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True)
                 for o in out]
        return {"void": False, "wrap": wrap, "seed": seed}, texts

    async def local_sample(self, system, text, n, max_new_tokens=32, seed=20260915):
        return await asyncio.to_thread(self._local_sample_sync, system, text, n,
                                       max_new_tokens, seed)

    def environment(self):
        return (self._load_meta or {}).get("environment")

    # ── PLAIN GENERATE ─────────────────────────────────────────────────────────
    async def generate(self, input, tools, tool_choice, config):
        system = "\n".join(m.text for m in input if m.role == "system")
        user = "\n".join(m.text for m in input if m.role == "user")
        if self.entry["kind"] == "api":
            _, _, recs = await self.api_draws("scenarios", system, user, 1, "generate")
            text = recs[0].get("raw_text") or ""
        else:
            _, texts = await self.local_sample(system, user, 1, config.max_tokens or 32)
            text = texts[0] if texts else ""
        return ModelOutput.from_content(model=self.model_name, content=text)
