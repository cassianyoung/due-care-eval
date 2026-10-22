"""A wheel install (`pip install .`, or from PyPI) reads its banks and model entries.

Until v1.1's re-tag the wheel shipped the code packages and no package data, so a
non-editable install failed on the first bank read with FileNotFoundError; every other
test passed because it runs from the checkout. This test:

1. builds the wheel from this tree (offline, no build isolation) and checks it holds every
   bank file listed in banks/MANIFEST.json, byte for byte, and every model entry, under
   due_care/banks and due_care/models, and no top-level banks/ or models/;
2. installs it with --no-deps into an empty target and, in a fresh interpreter whose path
   is that target plus the environment's dependencies and NOT this checkout, loads every
   bank through harness.frames, checks the manifest, loads every model entry, and checks
   both directories resolve inside the target;
3. the paired refusal: a wheel built from a copy of this tree with v1.1's packaging as it
   shipped (code packages only) must fail step 2 with FileNotFoundError, so the test can
   tell the broken wheel from a working one.

Needs pip and setuptools in the environment (the build is offline). Without them the
checks are SKIPPED BY NAME, never passed quietly.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

from _t import ROOT, Tally

T = Tally("test_wheel_install")
IGNORE = shutil.ignore_patterns(".git", ".venv*", "*.egg-info", "__pycache__", "build", "dist")

PROBE = r'''
import json, os, sys
target = sys.argv[1]
import harness.frames as F
import due_care.model_table as MT
out = {"banks": F.BANKS, "models": MT.MODELS, "frames_file": F.__file__}
man = json.load(open(os.path.join(F.BANKS, "MANIFEST.json")))
names = sorted(man["files"])
out["bank_files"] = len(names)
out["bank_reads"] = sum(1 for n in names if open(os.path.join(F.BANKS, n), "rb").read())
F.load_json("excuse_questions.json"); F.load_json("worksheet.json")
F.manifest_check()          # raises BankError if any bank file differs from the manifest
out["manifest_check"] = True
out["models"] = len([f for f in os.listdir(MT.MODELS) if f.endswith(".json")])
out["inside"] = all(os.path.realpath(p).startswith(os.path.realpath(target))
                    for p in (F.BANKS, MT.MODELS, F.__file__))
print(json.dumps(out))
'''


def have_build_tools():
    try:
        import setuptools  # noqa: F401
        import pip  # noqa: F401
        return True
    except ImportError:
        return False


def build_wheel(src, out):
    r = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                        "--no-build-isolation", "-q", "-w", out, src],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])
    whl = [f for f in os.listdir(out) if f.endswith(".whl")]
    if len(whl) != 1:
        raise RuntimeError(f"expected one wheel, got {whl}")
    return os.path.join(out, whl[0])


def probe(whl, tmp):
    target = os.path.join(tmp, "site")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q",
                        "--target", target, whl], capture_output=True, text=True)
    if r.returncode:
        return None, r.stderr[-800:]
    # the target first, then the environment's own site-packages for the dependencies;
    # the checkout is not on the path, and the probe runs from outside it
    deps = [p for p in sys.path if p.endswith("site-packages")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([target] + deps),
               HF_HUB_OFFLINE="1")
    env.pop("DUE_CARE_MODELS", None)
    script = os.path.join(tmp, "probe.py")
    open(script, "w").write(PROBE)
    r = subprocess.run([sys.executable, "-S", script, target], cwd=tmp, env=env,
                       capture_output=True, text=True)
    if r.returncode:
        return None, (r.stderr or r.stdout)[-800:]
    return json.loads(r.stdout.strip().splitlines()[-1]), ""


def main():
    if not have_build_tools():
        for lab in ("wheel holds every bank file and model entry",
                    "wheel install reads banks and model entries from the install",
                    "a wheel with v1.1's packaging fails the same probe"):
            T.skip(lab, "pip or setuptools not installed in this environment")
        return
    man = json.load(open(os.path.join(ROOT, "banks", "MANIFEST.json")))
    models = sorted(f for f in os.listdir(os.path.join(ROOT, "models")) if f.endswith(".json"))
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        shutil.copytree(ROOT, src, ignore=IGNORE)
        whl = build_wheel(src, os.path.join(tmp, "dist"))
        z = zipfile.ZipFile(whl)
        names = set(z.namelist())
        missing = [f for f in man["files"] if f"due_care/banks/{f}" not in names]
        differ = [f for f in man["files"] if f"due_care/banks/{f}" in names and
                  z.read(f"due_care/banks/{f}") != open(os.path.join(ROOT, "banks", f), "rb").read()]
        mmiss = [f for f in models if f"due_care/models/{f}" not in names]
        toplevel = sorted({n.split("/")[0] for n in names} & {"banks", "models"})
        T.check(f"wheel holds every bank file in the manifest ({len(man['files'])}), byte for "
                f"byte, and every model entry ({len(models)}), under due_care/",
                not missing and not differ and not mmiss,
                f"missing {missing[:3]}, differ {differ[:3]}, models missing {mmiss[:3]}")
        T.check("wheel installs no top-level banks/ or models/", not toplevel, str(toplevel))
        got, err = probe(whl, os.path.join(tmp, "a"))
        if got is None:
            T.check("wheel install reads banks and model entries from the install", False, err)
        else:
            T.check("wheel install reads banks and model entries from the install "
                    f"({got['bank_reads']} of {got['bank_files']} bank files, "
                    f"{got['models']} model entries, all inside the install)",
                    got["inside"] and got["bank_reads"] == got["bank_files"] ==
                    len(man["files"]) and got["models"] == len(models)
                    and got["manifest_check"], json.dumps(got))
        # the paired refusal: the same tree with v1.1's packaging
        # copied fresh from the checkout: `src` now holds the first build's build/, which
        # setuptools would reuse, shipping the data files the configuration lacks
        bad = os.path.join(tmp, "bad")
        shutil.copytree(ROOT, bad, ignore=IGNORE)
        pp = os.path.join(bad, "pyproject.toml")
        txt = open(pp).read()
        # restore v1.1's packaging as it shipped: the code packages only, no package-dir
        # and no package-data tables (with the packages declared, setuptools ships their
        # data files even without the package-data table, so removing that alone is not
        # the broken configuration)
        out_, skip = [], False
        for line in txt.splitlines(keepends=True):
            if line.startswith("["):
                skip = line.strip() in ("[tool.setuptools.package-dir]",
                                        "[tool.setuptools.package-data]")
            if skip:
                continue
            if line.startswith("packages = "):
                line = 'packages = ["due_care", "due_care.tasks", "harness", "db"]\n'
            out_.append(line)
        txt2 = "".join(out_)
        if txt2 == txt or "due_care.banks" in txt2:
            T.check("a wheel with v1.1's packaging fails the same probe", False,
                    "could not restore v1.1's packaging")
            return
        open(pp, "w").write(txt2)
        got2, err2 = probe(build_wheel(bad, os.path.join(tmp, "dist_bad")),
                           os.path.join(tmp, "b"))
        T.check("a wheel with v1.1's packaging fails the same probe (the test discriminates)",
                got2 is None and "FileNotFoundError" in err2, json.dumps(got2) or err2[-200:])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                       # noqa: BLE001
        T.check("test ran", False, f"{type(e).__name__}: {e}")
    T.done()
