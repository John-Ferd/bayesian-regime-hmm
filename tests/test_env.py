import platform

from bayes_hmm.env import environment


def test_environment_records_python_and_numpy():
    env = environment()
    assert env["python"] == platform.python_version()
    assert env["numpy"].count(".") >= 1


def test_environment_skips_packages_that_are_not_installed():
    env = environment(("numpy", "not-a-real-package-xyz"))
    assert "numpy" in env and "not-a-real-package-xyz" not in env
