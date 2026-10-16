"""A minimal test tally shared by the test scripts: each check prints PASS, FAIL or SKIP
with what it checked, and the script exits non-zero if any check failed."""
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class Tally:
    def __init__(self, name):
        self.name, self.passed, self.failed, self.skipped = name, 0, 0, 0

    def check(self, label, ok, detail=""):
        if ok:
            self.passed += 1
            print(f"  PASS  {label}")
        else:
            self.failed += 1
            print(f"  FAIL  {label}  {detail}")

    def skip(self, label, why):
        self.skipped += 1
        print(f"  SKIP  {label}  ({why})")

    def raises(self, label, exc, fn, *a, **k):
        try:
            fn(*a, **k)
        except exc:
            self.check(label, True)
            return
        except Exception as e:
            self.check(label, False, f"raised {type(e).__name__}: {e}")
            return
        self.check(label, False, "did not raise")

    def opens(self, label, fn, *a, **k):
        try:
            out = fn(*a, **k)
        except Exception as e:
            self.check(label, False, f"refused: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
        self.check(label, True)
        return out

    def done(self):
        print(f"{self.name}: {self.passed} passed, {self.failed} failed, "
              f"{self.skipped} skipped")
        sys.exit(1 if self.failed else 0)
