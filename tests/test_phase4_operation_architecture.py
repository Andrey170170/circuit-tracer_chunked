"""Architecture boundaries for decomposed NNSight Phase 4 operations."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from circuit_tracer.attribution.nnsight.phases.phase4 import run_phase4


MODULES = (
    "phase4_frontier.py",
    "phase4_influence.py",
    "phase4_batches.py",
    "phase4_storage.py",
    "phase4_diagnostics.py",
    "phase4_cleanup.py",
)
ROOT = Path(__file__).parents[1] / "circuit_tracer/attribution/nnsight/phases"


def test_phase4_functions_are_bounded() -> None:
    assert len(inspect.getsource(run_phase4).splitlines()) <= 180
    for name in MODULES:
        tree = ast.parse((ROOT / name).read_text())
        oversized = {
            node.name: node.end_lineno - node.lineno + 1
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.end_lineno is not None
            and node.end_lineno - node.lineno + 1 > 180
        }
        assert oversized == {}


def test_phase4_operations_do_not_import_orchestrator() -> None:
    for name in MODULES:
        tree = ast.parse((ROOT / name).read_text())
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "circuit_tracer.attribution.nnsight.phases.phase4" not in imports


def test_phase4_operation_dependency_graph_is_acyclic() -> None:
    module_names = {name.removesuffix(".py") for name in MODULES}
    dependencies = {name: set() for name in module_names}
    for module_name in module_names:
        tree = ast.parse((ROOT / f"{module_name}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                dependency = node.module.rsplit(".", 1)[-1]
                if dependency in module_names:
                    dependencies[module_name].add(dependency)

    visited: set[str] = set()
    active: set[str] = set()

    def visit(module_name: str) -> None:
        assert module_name not in active, f"Phase 4 dependency cycle at {module_name}"
        if module_name in visited:
            return
        active.add(module_name)
        for dependency in dependencies[module_name]:
            visit(dependency)
        active.remove(module_name)
        visited.add(module_name)

    for module_name in module_names:
        visit(module_name)
