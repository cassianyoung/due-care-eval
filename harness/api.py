"""The reader for models served through an API (OpenRouter's chat completions endpoint).

Rules the client enforces on every call:

* THE REASONING SETTING IS DECLARED. Every wire setting carries a `reasoning` key whose
  value is either an explicit object sent as-is (for example {"enabled": false} or
  {"effort": "low"}) or the literal string "vendor_default", which means the field is
  deliberately omitted and the provider's default applies. A wire setting without the
  key is refused before any request is made.
* ONE PROVIDER, NO FALLBACK. The request pins a single provider with fallbacks off, and
  the provider and model that actually served are read back from every response.
* NOTHING IS COERCED. A completion that does not parse is recorded as void with the
  reason, never as a digit.

An API returns only the top-k log-probabilities, so for a log-prob readout the expected
value is renormalised over the digit tokens that were returned, and the digit mass and
the k used are recorded.
"""
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

BASE = "https://openrouter.ai/api/v1"
TIMEOUT = 120
ATTEMPTS = 6
DIGITS = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
VENDOR_DEFAULT = "vendor_default"


class Refused(RuntimeError):
    """A precondition of the call does not hold; nothing was sent."""


def api_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise Refused("OPENROUTER_API_KEY is not set")
    return k


def check_wire(wire):
    """Refuse a wire setting that does not declare its reasoning configuration."""
    if "reasoning" not in wire:
        raise Refused("the wire setting does not declare `reasoning`; set an explicit "
                      "object or \"vendor_default\"")
    r = wire["reasoning"]
    if not (isinstance(r, dict) or r == VENDOR_DEFAULT):
        raise Refused(f"`reasoning` must be an object or \"{VENDOR_DEFAULT}\", got {r!r}")
    for k in ("model", "provider", "max_tokens", "want_logprobs", "temperature"):
        if k not in wire:
            raise Refused(f"the wire setting lacks `{k}`")
    return wire


def wire_sha(wire):
    """32-hex digest of a wire setting, keys sorted."""
    return hashlib.sha256(json.dumps(wire, sort_keys=True).encode()).hexdigest()[:32]


def payload_for(wire, messages):
    """The request body for one call. Exposed so tests can inspect what is sent."""
    check_wire(wire)
    p = {"model": wire["model"], "temperature": wire["temperature"],
         "max_tokens": wire["max_tokens"],
         "provider": {"order": [wire["provider"]], "allow_fallbacks": False},
         "usage": {"include": True}, "messages": messages}
    if wire["want_logprobs"]:
        p["logprobs"] = True
        p["top_logprobs"] = wire.get("top_k", 20)
    if wire["reasoning"] != VENDOR_DEFAULT:
        p["reasoning"] = wire["reasoning"]
    return p


class AccountRefused(RuntimeError):
    """OpenRouter refused the account (HTTP 401, 402 or 403): no later call can succeed,
    so the run stops instead of recording every cell as void."""


def _post(path, payload, key):
    body = json.dumps(payload).encode()
    last = None
    tried = 0
    for a in range(ATTEMPTS):
        tried = a + 1
        try:
            req = urllib.request.Request(BASE + path, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code in (401, 402, 403):
                raise AccountRefused(
                    f"OpenRouter refused the request ({last}). Check that "
                    f"OPENROUTER_API_KEY is a valid key and that the account has credit; "
                    f"the run stops here")
            if e.code in (400, 404):
                break
            if e.code == 429:
                # a rate limit is waited out on the same provider; switching provider
                # would put two models' readings in one cell
                time.sleep(8 * (a + 1) ** 2)
                continue
        except Exception as e:
            last = repr(e)
        time.sleep(1.5 * (a + 1))
    raise RuntimeError(f"call failed after {tried} attempt{'s' if tried > 1 else ''}: {last}")


def endpoints(model, key=None):
    """The providers currently serving `model`, with what each supports."""
    req = urllib.request.Request(f"{BASE}/models/{model}/endpoints",
                                 headers={"Authorization": f"Bearer {key or api_key()}"})
    data = json.load(urllib.request.urlopen(req, timeout=60))["data"]
    return [{"provider": e.get("provider_name"),
             "logprobs": "logprobs" in (e.get("supported_parameters") or []),
             "top_logprobs": "top_logprobs" in (e.get("supported_parameters") or [])}
            for e in data["endpoints"]]


def digit_distribution(top):
    """{digit: renormalised probability} over the ASCII digit tokens in a top-k list,
    and the raw digit mass before renormalisation."""
    p = {}
    for t in top:
        s = (t.get("token") or "").strip()
        if s in DIGITS:
            p[int(s)] = p.get(int(s), 0.0) + math.exp(t["logprob"])
    mass = sum(p.values())
    if not p:
        return None, 0.0
    return {d: p.get(d, 0.0) / mass for d in range(10)}, mass


def yes_no_share(top):
    """P(yes) renormalised over the yes/no tokens in a top-k list, and their raw mass."""
    p = {}
    for t in top:
        s = (t.get("token") or "").strip().lower()
        if s in ("yes", "no"):
            p[s] = p.get(s, 0.0) + math.exp(t["logprob"])
    tot = sum(p.values())
    return (p.get("yes", 0.0) / tot if tot else None), tot


def parse_digit(text):
    s = (text or "").strip().strip(".").strip()
    return int(s) if len(s) == 1 and s in DIGITS else None


def parse_yes_no(text):
    s = (text or "").strip().strip(".").strip().lower()
    return s if s in ("yes", "no") else None


def parse_pair(text, label_a, label_b):
    """(score for label_a, score for label_b) from "LABEL=<digit>" pairs, or Nones."""
    out = []
    for lab in (label_a, label_b):
        m = re.search(re.escape(lab) + r"\s*=\s*(\d)", text or "")
        out.append(int(m.group(1)) if m else None)
    return tuple(out)


def reasoned(rec):
    """A draw reasoned if it returned reasoning text OR billed reasoning tokens.
    Some providers bill reasoning tokens without returning the text, so the text alone
    would report real reasoning as absent."""
    if (rec.get("reasoning_text_len") or 0) > 0:
        return True
    ctd = ((rec.get("usage") or {}).get("completion_tokens_details") or {})
    return (ctd.get("reasoning_tokens") or 0) > 0


def truncated(rec):
    """The completion was cut off by max_tokens (the answer never appeared, or the
    provider reports a length stop)."""
    return rec.get("finish_reason") == "length" or (
        not (rec.get("raw_text") or "").strip() and rec.get("reasoning_text_len", 0) > 0)


def call(wire, messages, label="", key=None):
    """One call. Returns the full record; never raises on an unparseable answer."""
    payload = payload_for(wire, messages)
    t0 = time.time()
    r = _post("/chat/completions", payload, key or api_key())
    ch = (r.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    text = msg.get("content") or ""
    reasoning_text = msg.get("reasoning") or ""
    lp = ch.get("logprobs") or {}
    content = lp.get("content") or []
    top = (content[0].get("top_logprobs") if content else None) or []
    if not top and isinstance(lp.get("top_logprobs"), list) and lp["top_logprobs"]:
        first = lp["top_logprobs"][0]
        if isinstance(first, dict):
            top = [{"token": t, "logprob": v} for t, v in first.items()]
    usage = r.get("usage") or {}
    return {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "label": label,
            "model_asked": wire["model"], "provider_asked": wire["provider"],
            "provider_served": r.get("provider"), "model_served": r.get("model"),
            "wire_sha": wire_sha(wire),
            "reasoning_on_the_wire": payload.get("reasoning", VENDOR_DEFAULT),
            "max_tokens": wire["max_tokens"],
            "top_k": payload.get("top_logprobs"),
            "finish_reason": ch.get("finish_reason"),
            "raw_text": text[:400], "reasoning_text_len": len(reasoning_text),
            "top_tokens": [{"token": t.get("token"), "logprob": t.get("logprob")}
                           for t in top[:20]],
            "usage": usage, "cost_usd": float(usage.get("cost") or 0.0),
            "latency_s": round(time.time() - t0, 2)}
