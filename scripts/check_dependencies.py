#!/usr/bin/env python3
"""Fail if the runtime package imports anything outside the standard library.

The zero-dependency claim is a promise to users: it is what lets the service run
on a stock interpreter with nothing to audit or patch. That promise is easy to
break accidentally, so CI enforces it by parsing the AST rather than trusting
review.

Scope: ``src/`` only. The scripts and tests are developer tooling and are allowed
to use whatever they like -- though in practice they are standard library only
too.

Usage:
    python scripts/check_dependencies.py [--package-dir src] [--package riskscore]
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import List, Sequence, Set, Tuple

#: Packages that are unambiguously not part of the standard library, used as the
#: fallback check on interpreters without ``sys.stdlib_module_names`` (below
#: Python 3.10).
#:
#: Why a denylist rather than a stdlib list: the previous fallback collected
#: ``sys.modules``, which by the time this script runs contains the interpreter's
#: own startup imports *and* anything installed in site-packages. On a GitHub
#: Actions runner that made ``ast`` and ``pathlib`` look like third-party
#: packages, so the check failed on Python 3.9 while passing everywhere else. An
#: allowlist copied from CPython 3.9 would be long and would rot silently; a
#: denylist of realistic dependencies is short, obviously correct, and cannot
#: produce false positives. The precise check still runs on every modern
#: interpreter.
KNOWN_THIRD_PARTY: Tuple[str, ...] = (
    "attr", "attrs", "azure", "bcrypt", "beautifulsoup4", "bs4", "boto3", "botocore",
    "celery", "click", "coverage", "cv2", "dateutil", "django", "docx", "environs",
    "fastapi", "flask", "gensim", "google", "grpc", "httpx", "jinja2", "joblib",
    "jwt", "keras", "lightgbm", "lxml", "matplotlib", "mypy", "mysql", "nltk",
    "numpy", "openai", "openpyxl", "pandas", "passlib", "pillow", "PIL", "plotly",
    "psycopg2", "pydantic", "pyodbc", "pytest", "pytz", "redis", "requests",
    "scipy", "seaborn", "selenium", "setuptools", "sklearn", "sqlalchemy", "starlette",
    "statsmodels", "tensorflow", "torch", "tqdm", "transformers", "typing_extensions",
    "uvicorn", "xgboost", "yaml",
)


def local_module_names(package_dir: Path) -> Set[str]:
    """Top-level module names that belong to this project, not to PyPI."""
    names = {"riskscore"}
    for entry in package_dir.iterdir():
        if entry.is_dir() and (entry / "__init__.py").exists():
            names.add(entry.name)
        elif entry.suffix == ".py" and entry.stem != "__init__":
            names.add(entry.stem)
    return names


def imported_top_level_modules(tree: ast.AST) -> List[Tuple[str, int]]:
    """Return ``(module, lineno)`` for every top-level module imported."""
    found: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # ``from . import x`` has module=None and level=1: a relative import,
            # which can never reach a third-party package.
            if node.level and node.level > 0:
                continue
            module = (node.module or "").split(".")[0]
            if module:
                found.append((module, node.lineno))
    return found


def check(package_dir: Path, *, allow: Sequence[str] = ()) -> Tuple[List[str], str]:
    """Return ``(violations, mode)`` where mode describes how the check was done."""
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib:
        mode = f"stdlib_module_names (Python {sys.version_info.major}.{sys.version_info.minor})"
        allowed: Set[str] = set(stdlib)
    else:
        mode = (
            f"known-third-party denylist (Python "
            f"{sys.version_info.major}.{sys.version_info.minor} has no sys.stdlib_module_names)"
        )
        allowed = set()

    known_third_party = set(KNOWN_THIRD_PARTY)
    project_modules = local_module_names(package_dir)
    explicitly_allowed = set(allow) | project_modules

    violations: List[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{path}: cannot parse ({exc})")
            continue
        for module, lineno in imported_top_level_modules(tree):
            if module in explicitly_allowed:
                continue
            if stdlib:
                if module not in allowed:
                    violations.append(f"{path}:{lineno}: third-party import {module!r}")
            elif module in known_third_party:
                violations.append(f"{path}:{lineno}: third-party import {module!r}")
    return violations, mode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package-dir", default="src", help="directory holding the runtime package")
    parser.add_argument("--allow", action="append", default=[], help="extra module to permit (repeatable)")
    args = parser.parse_args(argv)

    package_dir = Path(args.package_dir)
    if not package_dir.is_dir():
        print(f"error: {package_dir} is not a directory", file=sys.stderr)
        return 2

    violations, mode = check(package_dir, allow=args.allow)
    if violations:
        print("third-party imports found in the runtime package:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        print(
            "\nThe runtime is deliberately dependency-free. Move the import into "
            "scripts/ or tests/, or add the dependency to pyproject.toml and update "
            "the README's zero-dependency claim.",
            file=sys.stderr,
        )
        return 1

    total = sum(1 for _ in package_dir.rglob("*.py"))
    print(f"{package_dir}/ imports only the standard library ({total} modules checked)")
    print(f"check mode: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
