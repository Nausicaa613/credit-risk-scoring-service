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


def check(package_dir: Path, *, allow: Sequence[str] = ()) -> List[str]:
    """Return a list of human-readable violations (empty means clean)."""
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    if not stdlib:
        # Python < 3.10 has no stdlib_module_names. Fall back to the documented
        # module list rather than silently passing.
        stdlib = set(sys.builtin_module_names) | {
            name for name in sys.modules if not name.startswith("_")
        }
    allowed = set(allow) | local_module_names(package_dir) | stdlib

    violations: List[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{path}: cannot parse ({exc})")
            continue
        for module, lineno in imported_top_level_modules(tree):
            if module not in allowed:
                violations.append(f"{path}:{lineno}: third-party import {module!r}")
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-dir", default="src", help="directory holding the runtime package")
    parser.add_argument("--allow", action="append", default=[], help="extra module to permit (repeatable)")
    args = parser.parse_args(argv)

    package_dir = Path(args.package_dir)
    if not package_dir.is_dir():
        print(f"error: {package_dir} is not a directory", file=sys.stderr)
        return 2

    violations = check(package_dir, allow=args.allow)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
