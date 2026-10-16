"""Imported by Inspect through the package's entry point, so that the `due_care` model
provider and the `due_care/excuse` and `due_care/duty` tasks are registered."""
from .provider import DueCareAPI  # noqa: F401
from .tasks.duty_test import duty  # noqa: F401
from .tasks.excuse_test import excuse  # noqa: F401
