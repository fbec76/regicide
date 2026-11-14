"""Import-boundary lint: ml/envs/** must stay pure Gymnasium + numpy (ADR-002).

All SB3 / VecEnv / torch usage is confined to ml/trainers/**; the env layer is
enforced dependency-free of stable_baselines3, sb3_contrib and torch via this
AST-level check (catches direct imports AND from-imports, including inside
functions where a surface-level grep would miss them).
"""

import ast
import os

FORBIDDEN_ROOTS = ("stable_baselines3", "sb3_contrib", "torch")

# tests/ml/test_boundaries.py -> repo root -> ml/envs
ENVS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ml", "envs")
)


def _iter_envs_modules():
    for dirpath, _dirnames, filenames in os.walk(ENVS_DIR):
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _forbidden_imports(tree: ast.AST) -> list:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_ROOTS:
                    found.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_ROOTS:
                found.append(f"from {node.module} import ...")
    return found


def test_envs_modules_import_no_sb3_torch():
    modules = list(_iter_envs_modules())
    assert modules, f"no .py modules found under {ENVS_DIR} — path check is broken"
    violations = []
    for path in modules:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        for imp in _forbidden_imports(tree):
            violations.append(f"{os.path.relpath(path)}: {imp}")
    assert not violations, (
        "ml/envs must not import stable_baselines3/sb3_contrib/torch:\n"
        + "\n".join(violations)
    )
