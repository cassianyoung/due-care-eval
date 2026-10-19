"""An API-only install (`pip install -e .`, no torch) runs an API task.

torch and the other local-weights packages are made unimportable before anything else is
imported, as they are in an environment that never installed them. The package must then
register its tasks and provider, run the excuse and duty tasks on an API entry end to end
(offline: the API call is a stub), and never import `harness.local` or torch; and a
local-weights entry must say which extra it needs rather than fail obscurely."""
import importlib.abc
import math
import os
import sys
import tempfile

BLOCKED = ("torch", "transformers", "peft", "safetensors", "accelerate")


class NoLocalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r} (blocked by the test)",
                                      name=name)
        return None


for m in [m for m in sys.modules if m.split(".")[0] in BLOCKED]:
    del sys.modules[m]
sys.meta_path.insert(0, NoLocalPackages())

from _t import ROOT, Tally  # noqa: E402

T = Tally("test_api_without_torch")
try:
    import torch  # noqa: F401
    T.check("torch is unimportable in this environment", False)
except ImportError:
    T.check("torch is unimportable in this environment", True)

import harness.api as API  # noqa: E402
from inspect_ai import eval as inspect_eval  # noqa: E402

import due_care._registry  # noqa: E402,F401
from due_care import model_table as MT  # noqa: E402
from due_care import provider as P  # noqa: E402
from due_care.tasks.duty_test import duty  # noqa: E402
from due_care.tasks.excuse_test import excuse  # noqa: E402

T.check("the package registers its provider and tasks without torch",
        "harness.local" not in sys.modules and "torch" not in sys.modules)


def stub(wire, messages, label="", key=None):
    user = messages[-1]["content"]
    if "Answer in exactly this form" in user:
        a, b = user.rsplit("form: ", 1)[1].replace("<digit>", "").split()
        text, top = f"{a}6 {b}3", []
    else:
        text, top = "3", [{"token": "3", "logprob": math.log(0.9)},
                          {"token": "4", "logprob": math.log(0.1)}]
    return {"raw_text": text, "reasoning_text_len": 0, "finish_reason": "stop",
            "provider_served": wire["provider"], "model_served": wire["model"],
            "wire_sha": API.wire_sha(wire), "top_tokens": top,
            "reasoning_on_the_wire": wire["reasoning"], "usage": {}, "cost_usd": 0.0,
            "top_k": wire.get("top_k")}


API.call = stub
MT.MODELS = os.path.join(ROOT, "tests", "fixtures", "models")
with tempfile.TemporaryDirectory() as d:
    log = inspect_eval(excuse(scenarios="bicycle"), model="due_care/example_api",
                       log_dir=d, display="none")[0]
    T.check("the excuse task runs on an API entry without torch",
            log.status == "success" and len(log.samples or []) == 36, log.status)
    dlog = inspect_eval(duty(pairs="prawns"), model="due_care/example_api", log_dir=d,
                        display="none")[0]
    T.check("the duty task runs on an API entry without torch",
            dlog.status == "success" and len(dlog.samples or []) > 0, dlog.status)
    T.check("and neither run imported harness.local or torch",
            "harness.local" not in sys.modules and "torch" not in sys.modules)

T.raises("a local-weights entry, without torch, names the extra it needs", ImportError,
         P._local)
try:
    P._local()
except ImportError as exc:
    T.check("the message is the install line", 'pip install -e ".[local]"' in str(exc),
            str(exc))
T.done()
