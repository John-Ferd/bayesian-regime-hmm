"""Record the software versions behind a results file, so a number can be traced
back to the exact Python and library versions that produced it."""
from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ("numpy", "pandas", "yfinance")


def environment(packages=PACKAGES) -> dict:
    """Python version, platform and installed versions of the libraries that matter here."""
    env = {"python": platform.python_version(), "platform": platform.platform(terse=True)}
    for name in packages:
        try:
            env[name] = version(name)
        except PackageNotFoundError:
            pass
    return env
