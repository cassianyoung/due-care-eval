"""Run every offline test script and report each one's result.

    HF_HUB_OFFLINE=1 python tests/run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = [os.path.join(HERE, f) for f in ("test_rules.py", "test_db_bridge.py",
                                            "test_inspect_end_to_end.py", "test_local.py",
                                            "test_bank_v1_1.py", "test_v1_1_fill.py",
                                            "test_api_without_torch.py")]
SCRIPTS.append(os.path.join(ROOT, "db", "test_views.py"))
env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH=ROOT)
failed = []
for s in SCRIPTS:
    r = subprocess.run([sys.executable, s], cwd=os.path.dirname(s), env=env,
                       capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if l.strip()][-1:] or ["(no output)"]
    print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {os.path.relpath(s, ROOT)}: {tail[0]}")
    if r.returncode:
        failed.append(s)
        print(r.stdout[-2000:], r.stderr[-2000:])
print(f"{len(SCRIPTS) - len(failed)} of {len(SCRIPTS)} test scripts passed")
sys.exit(1 if failed else 0)
