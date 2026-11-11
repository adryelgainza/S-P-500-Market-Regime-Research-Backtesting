"""Verify the supported interpreter and exact active dependency pins."""

from importlib import metadata
from pathlib import Path
import platform
import tomllib

from packaging.requirements import Requirement

from .archive import file_hash


def verify_environment(project):
    project = Path(project)
    expected_python = (project/".python-version").read_text(encoding="utf-8").strip()
    if platform.python_version() != expected_python:
        raise ValueError(f"Use the locked Python version {expected_python}")
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise ValueError("The current release lock supports Windows x86-64")
    with (project/"pyproject.toml").open("rb") as stream:
        declared_version = tomllib.load(stream)["project"]["version"]
    package_version = metadata.version("regime-research")
    from . import __version__
    if package_version != declared_version or __version__ != declared_version:
        raise ValueError("Reinstall the local package to match pyproject.toml")
    packages = {}
    for line in (project/"requirements.lock").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        actual = metadata.version(requirement.name)
        if not requirement.specifier.contains(actual, prereleases=True):
            raise ValueError(f"Dependency differs from lock: {requirement.name}")
        packages[requirement.name] = actual
    return {"Python": platform.python_version(), "Package_Version": package_version, "Platform": platform.platform(), "Packages": packages,
            "Lock_SHA256": file_hash(project/"requirements.lock")}
