"""Inspect datasets built from the frozen banks at import time.

Each sample is one cell. Its input is the exact text the model reads (the user turn on
the chat frame, the filled worksheet on the worksheet frame); its metadata names the
cell and carries the SHA-256 of the bank text it was built from, which is re-checked
against the bank before the sample is made.
"""
from inspect_ai.dataset import MemoryDataset, Sample

from harness import frames as F

SCENARIOS = ("bicycle", "ladder", "prawns_home", "prawns_restaurant", "scaffolding",
             "cheque")
QUESTIONS = ("BLAME", "CAUSE", "REASONABLE", "CARELESS")
DUTY_PAIRS = ("prawns", "1", "3", "4", "5", "7")
SEPARATE_ALARIC = "Alaric"


class PromptCheckRefused(F.BankError):
    """A prompt the read would send is not in the pre-flight file; nothing was sent."""


def preflight(dataset, path):
    """Refuse the read, before any call, unless every prompt it would send hashes to a
    digest in `path` (see harness.frames.prompt_sha16 for the form hashed). Returns the
    number of prompts checked."""
    want = F.load_prompt_check(path)
    missing = [s.id for s in dataset
               if F.prompt_sha16(s.metadata.get("system"), s.input) not in want]
    if missing:
        raise PromptCheckRefused(
            f"pre-flight: {len(missing)} of {len(dataset)} prompts are not in {path}, "
            f"so nothing is sent: {missing[:5]}")
    return len(dataset)


def _cases():
    F.manifest_check()
    return F.load_jsonl("excuse_cases.jsonl")


def excuse_dataset(frame="chat", scenarios=SCENARIOS, questions=QUESTIONS,
                   roles=F.TABLE1_ROLES, include_supplier=False):
    """Samples for the excuse test. The worksheet frame asks BLAME only."""
    qs = F.load_json("excuse_questions.json")
    ws = F.load_json("worksheet.json")
    samples = []
    for c in _cases():
        if c["scenario"] not in scenarios:
            continue
        if c["role"] not in roles and not (include_supplier and c["role"] == "SUPPLIER"):
            continue
        group = "bicycle" if c["scenario"] == "bicycle" else "scenarios"
        if frame == "worksheet":
            if "BLAME" not in questions:
                continue
            for o in ws["orderings"]:
                body = F.render_worksheet(o, c["text"])
                samples.append(Sample(
                    id=f"{c['scenario']}|{c['role']}|BLAME|{o['id']}", input=body,
                    metadata={"test": "excuse", "scenario": c["scenario"],
                              "role": c["role"], "question": "BLAME", "frame": "worksheet",
                              "ordering": o["id"], "text_sha256": c["sha256"],
                              "frame_sha256": o["sha256"], "system": None,
                              "answers": ws["answers"],
                              "answer_prefix": ws["answer_prefix"], "group": group}))
            continue
        for q in questions:
            if q in ("REASONABLE", "CARELESS") and c["role"] in F.NO_PROBE_ROLES:
                continue
            system, user = F.build_chat(qs, c, q)
            _, _, body_sha = F.question_body(qs, q, c["scenario"])
            answers = (qs["CAUSE"] if q == "CAUSE" else qs[q])["answers"]
            samples.append(Sample(
                id=f"{c['scenario']}|{c['role']}|{q}|chat", input=user,
                metadata={"test": "excuse", "scenario": c["scenario"], "role": c["role"],
                          "question": q, "frame": "chat", "ordering": "chat",
                          "text_sha256": c["sha256"], "frame_sha256": body_sha,
                          "system": system, "answers": answers,
                          "answer_prefix": "", "group": group}))
    return MemoryDataset(samples, name=f"excuse_{frame}")


# The duty conditions a read can be narrowed to: a kind names every condition of that
# kind, and a condition names itself.
DUTY_KINDS = ("separate", "comparison", "stated", "neutral")
DUTY_CONDITIONS = ("bare", "cmp_pro_first", "cmp_priv_first", "cmp_full_pro_first",
                   "cmp_full_priv_first", "stated", "neutral_1", "neutral_2", "neutral_3")
# names for groups of conditions: the banked joint prompt in both orders, and the
# full-text joint prompt (bank v1.1, prawns) in both orders
DUTY_CONDITION_GROUPS = {"joint": ("cmp_pro_first", "cmp_priv_first"),
                         "joint_full": ("cmp_full_pro_first", "cmp_full_priv_first")}


def _condition_of(t):
    if t["kind"] == "comparison":
        return f"cmp_{t['variant']}"
    return {"separate": "bare", "stated": "stated"}.get(t["kind"], t["variant"])


def _check_selection(name, wanted, known):
    unknown = sorted(set(wanted) - set(known))
    if unknown:
        raise ValueError(f"unknown {name} {unknown}; one of {list(known)}")


def duty_dataset(frame="chat", pairs=DUTY_PAIRS, legs=DUTY_KINDS, bank=F.BANK_LATEST,
                 conditions=None):
    """Samples for the duty test.

    Separate, stated and neutral cells ask the BLAME question of one case (subject
    Alaric); comparison cells put both cases side by side in one of two orders and ask
    for both scores. The comparison needs two digits in one answer, so it is read on the
    chat frame only.

    `bank` is the bank version ("1.0" or "1.1"); v1.1 adds the prawns pair's neutral
    sentences and leaves every v1.0 text as it was. `conditions` narrows the read to some
    conditions, each a kind (separate, comparison, stated, neutral), a group (joint: the
    banked joint prompt in both orders; joint_full: the full-text joint prompt in both
    orders) or a condition (bare, cmp_pro_first, cmp_priv_first, cmp_full_pro_first,
    cmp_full_priv_first, stated, neutral_1, neutral_2, neutral_3). An unknown pair or
    condition is refused rather than read as an empty selection.
    """
    texts = F.load_duty_texts(bank)
    _check_selection("pairs", pairs, DUTY_PAIRS)
    _check_selection("legs", legs, DUTY_KINDS)
    if conditions is not None:
        _check_selection("conditions", conditions,
                         DUTY_KINDS + DUTY_CONDITIONS + tuple(DUTY_CONDITION_GROUPS))
        conditions = tuple(conditions) + tuple(
            c for g in conditions for c in DUTY_CONDITION_GROUPS.get(g, ()))
    qs = F.load_json("excuse_questions.json")
    dq = F.load_json("duty_questions.json")
    ws = F.load_json("worksheet.json")
    samples = []
    for t in texts:
        if t["pair"] not in pairs or t["kind"] not in legs:
            continue
        cond = _condition_of(t)
        if conditions is not None and t["kind"] not in conditions \
                and cond not in conditions:
            continue
        meta = {"test": "duty", "pair": t["pair"], "leg": t["leg"], "kind": t["kind"],
                "condition": cond, "text_sha256": t["sha256"], "frame": frame,
                "bank_version": bank}
        if t["kind"] == "comparison":
            if frame != "chat":
                continue
            samples.append(Sample(
                id=f"{t['pair']}|cmp|{t['variant']}", input=t["text"],
                metadata={**meta, "condition": f"cmp_{t['variant']}", "question": "CMPN",
                          # the full-text prompt's variants are full_pro_first and
                          # full_priv_first; the order is what follows the prefix
                          "order": t["variant"].removeprefix("full_"),
                          "label_first": t["label_first"],
                          "label_second": t["label_second"],
                          "system": dq["comparison"]["system"],
                          "group": "duty_comparison"}))
            continue
        case = {"scenario": "duty", "text": t["text"]}
        if F.subject_of(t["text"]) != SEPARATE_ALARIC:
            raise F.BankError(f"duty case {t['pair']}/{t['leg']} does not name Alaric")
        if frame == "worksheet":
            for o in ws["orderings"]:
                samples.append(Sample(
                    id=f"{t['pair']}|{t['leg']}|{cond}|{o['id']}",
                    input=F.render_worksheet(o, t["text"]),
                    metadata={**meta, "question": "BLAME", "ordering": o["id"],
                              "system": None, "answers": ws["answers"],
                              "answer_prefix": ws["answer_prefix"],
                              "group": "duty_separate"}))
            continue
        system, _, _ = F.question_body(qs, "BLAME", "bicycle")
        user = qs["BLAME"]["body"].format(item=t["text"], agent=SEPARATE_ALARIC)
        samples.append(Sample(
            id=f"{t['pair']}|{t['leg']}|{cond}|chat", input=user,
            metadata={**meta, "question": "BLAME", "ordering": "chat", "system": system,
                      "answers": qs["BLAME"]["answers"], "answer_prefix": "",
                      "group": "duty_separate"}))
    return MemoryDataset(samples, name=f"duty_{frame}")
