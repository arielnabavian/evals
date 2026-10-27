"""Skill libraries as values: checking one, and storing each version as an immutable, content-addressed directory.

The skill format and its checks are the skill-library formula's (ported from harness-optimizer's skill-library formula):
a library is a directory of ``<name>/SKILL.md`` (plus any files a skill needs), and a skill whose frontmatter has no
description can never be loaded.

A study stores every version in its artifact store (``<record>/artifacts``), the default library too, and a version's
value is its path there, ``skills/<hash>``: relative, so a record means the same wherever it is moved. ``resolve`` turns
a value into the path a factory receives; a value that is an absolute path (records written before 2026-10-24) is used
as it is.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

KIND = "skills"  # the folder of the artifact store that holds skill-library versions


def check_library(path: str) -> str | None:
    """Why ``path`` is not a usable skill library, if it is not."""
    from .._reflection import skill_dirs_in, validate_skill_dir

    if not os.path.isdir(path):
        return f"no skill library at {path}"
    problems = [
        f"{os.path.basename(os.path.normpath(d))}: {'; '.join(p)}"
        for d in skill_dirs_in(path)
        if (p := validate_skill_dir(d))
    ]
    return "; ".join(problems) or None


def digest(path: str | Path) -> str:
    """A content hash of a directory: every file's relative path and bytes."""
    root = Path(path)
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(root)).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def store(members: dict[str, str], artifacts: str | Path) -> str:
    """Write a library (``{name: skill directory}``) as one version in a study's artifact store, under
    ``artifacts/skills/<hash>/``, and return its value: ``skills/<hash>``, relative to ``artifacts``. The same library
    always lands in the same directory, which is never changed afterwards."""

    def fill(tmp: Path) -> None:
        for name, src in sorted(members.items()):
            shutil.copytree(src, tmp / name)

    return _store(fill, artifacts)


def store_directory(path: str | Path, artifacts: str | Path) -> str:
    """Store the library at ``path`` (every file in it, as it is) as one version in a study's artifact store, and return
    its value (see ``store``)."""

    def fill(tmp: Path) -> None:
        shutil.copytree(path, tmp, dirs_exist_ok=True)

    return _store(fill, artifacts)


def _store(fill: Callable[[Path], None], artifacts: str | Path) -> str:
    into = Path(artifacts) / KIND
    into.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=into) as tmp:
        fill(Path(tmp))
        name = digest(tmp)
        target = into / name
        if not target.exists():
            os.replace(tmp, target)
            os.makedirs(tmp)  # TemporaryDirectory removes what it made
    return f"{KIND}/{name}"


def resolve(value: str | None, artifacts: str | Path | None) -> str | None:
    """The path a skill-library value names, as a factory receives it: a stored version (``skills/<hash>``, relative to
    a study's artifact store) under ``artifacts``; an absolute path (as records written before 2026-10-24 hold), a
    library the study has not stored, or None, as it is."""
    if not value or artifacts is None or os.path.isabs(value):
        return value
    path = Path(artifacts) / value
    return str(path.resolve()) if path.exists() else value
