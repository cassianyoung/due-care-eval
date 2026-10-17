"""Prompt rendering from the frozen banks.

Every prompt is built from strings in `banks/`, and every text is checked against its
SHA-256 before it is used. A question never shares a context with another question:
`isolate` checks the built prompt carries its own question stem and no other.
"""
import hashlib
import json
import os

BANKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "banks")

SUBJECT_NAMES = ("Alaric",)
TABLE1_ROLES = ("N", "C", "I", "P0", "P1", "P1_R", "P1_neutral", "P2", "P2_other",
                "BYSTANDER")
# REASONABLE and CARELESS are not asked of the no-harm and deliberate cases
NO_PROBE_ROLES = ("N", "I")
STEMS = {"BLAME": "to blame, on a scale from",
         "REASONABLE": "reasonably have prevented this?",
         "CARELESS": "do anything careless?"}


class BankError(ValueError):
    """A bank text that does not hash to its recorded SHA-256, or a malformed row."""


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _checked(row, field="text"):
    if sha256(row[field]) != row["sha256"]:
        raise BankError(f"text does not hash to its recorded sha256: {row}")
    return row


def load_json(name):
    with open(os.path.join(BANKS, name), encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(name):
    with open(os.path.join(BANKS, name), encoding="utf-8") as fh:
        return [_checked(json.loads(l)) for l in fh if l.strip()]


BANK_VERSIONS = ("1.0", "1.1")
BANK_LATEST = "1.1"
# what each bank version adds to the duty texts of the one before; v1.0's
# duty_texts.jsonl is frozen and never edited, so a v1.0 run renders what it always did
DUTY_TEXT_FILES = {"1.0": ("duty_texts.jsonl",),
                   "1.1": ("duty_texts.jsonl", "duty_texts_v1_1.jsonl")}


def check_version(version):
    if version not in BANK_VERSIONS:
        raise BankError(f"unknown bank version {version!r}; one of {BANK_VERSIONS}")
    return version


def manifest_check(version=BANK_LATEST):
    """Every bank file of `version` hashes to the value MANIFEST.json records for it."""
    man = load_json("MANIFEST.json")
    names = man["versions"][check_version(version)]["files"]
    bad = []
    for name, want in ((n, man["files"][n]) for n in names):
        with open(os.path.join(BANKS, name), "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != want:
            bad.append(name)
    if bad:
        raise BankError(f"bank files differ from MANIFEST.json: {bad}")
    return man


# the duty CAUSE questions of each bank version: none in v1.0
DUTY_QUESTION_FILES = {"1.0": (), "1.1": ("duty_questions_v1_1.json",)}


def load_duty_cause(version=BANK_LATEST):
    """{pair: CAUSE question} for a bank version, each body checked against its sha256."""
    manifest_check(version)
    out = {}
    for name in DUTY_QUESTION_FILES[check_version(version)]:
        for pid, q in load_json(name)["cause"].items():
            if sha256(q["body"]) != q["sha256"]:
                raise BankError(f"{name}: the {pid} CAUSE question does not hash to its "
                                f"sha256")
            out[pid] = q
    return out


def load_duty_texts(version=BANK_LATEST):
    """The duty texts of a bank version, each with the bank file it came from, every
    text checked against its sha256."""
    manifest_check(version)
    rows = []
    for name in DUTY_TEXT_FILES[version]:
        rows += [{**r, "bank_file": name} for r in load_jsonl(name)]
    return rows


def prompt_sha16(system, user):
    """16-hex sha256 of a prompt, in the form the study hashed it: the chat messages as
    sent to an API (json.dumps of the system and user turns), or the bare text where
    there is no system turn (the worksheet)."""
    if system is None:
        body = user
    else:
        body = json.dumps([{"role": "system", "content": system},
                           {"role": "user", "content": user}])
    return sha256(body)[:16]


def load_prompt_check(path):
    """The set of 16-hex prompt digests in a pre-flight file: a JSON list, or an object
    whose list values are digests (non-list values, such as a `what` note, are
    ignored)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    vals = data if isinstance(data, list) else [
        x for v in data.values() if isinstance(v, list) for x in v]
    if not vals or not all(isinstance(x, str) and len(x) == 16 for x in vals):
        raise BankError(f"{path}: not a list of 16-hex prompt digests")
    return set(vals)


def subject_of(text):
    """The one subject named in a case. Refuses zero and refuses two."""
    found = [n for n in SUBJECT_NAMES if n in text]
    if len(found) != 1:
        raise BankError(f"a case must name exactly one subject; found {found}: {text[:80]!r}")
    return found[0]


def question_body(questions, channel, scenario):
    """(system, body, body_sha256) for a channel; CAUSE is worded per scenario."""
    if channel == "CAUSE":
        q = questions["CAUSE"]["per_scenario"][scenario]
    else:
        q = questions[channel]
    if sha256(q["body"]) != q["sha256"]:
        raise BankError(f"{channel} question body does not hash to its sha256")
    return q["system"], q["body"], q["sha256"]


def cause_stem(body):
    """The CAUSE question's own stem: the words from 'cause' to 'on a scale from'."""
    i = body.index(" cause ")
    return body[i + 1: body.index(", on a scale from") + len(", on a scale from")]


def build_chat(questions, case, channel):
    """(system, user_turn) for one case and channel, rendered as the API arms read it.

    BLAME and CAUSE carry the case inside the question body; REASONABLE and CARELESS
    follow the case after a blank line.
    """
    system, body, _ = question_body(questions, channel, case["scenario"])
    subject = subject_of(case["text"])
    if "{item}" in body:
        user = body.format(item=case["text"], agent=subject)
    else:
        user = f"{case['text']}\n\n{body.format(agent=subject)}"
    isolate(channel, user, questions, case["scenario"])
    return system, user


def isolate(channel, prompt, questions, scenario):
    stems = dict(STEMS)
    stems["CAUSE"] = cause_stem(questions["CAUSE"]["per_scenario"][scenario]["body"])
    if stems[channel] not in prompt:
        raise BankError(f"{channel}: its own question is not in its prompt")
    for other, st in stems.items():
        if other != channel and st in prompt:
            raise BankError(f"{channel}: the prompt also carries {other}'s question")


def render_worksheet(ordering, text):
    """The worksheet body for one ordering, with the case and its subject filled in."""
    if sha256(ordering["body"]) != ordering["sha256"]:
        raise BankError(f"{ordering['id']} does not hash to its sha256")
    return ordering["body"].format(item=text, agent=subject_of(text))
