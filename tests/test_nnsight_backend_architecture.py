"""Architecture bounds for the decomposed NNSight backend."""

from __future__ import annotations

import ast
from pathlib import Path

BACKEND_ROOT = Path("circuit_tracer/attribution/nnsight")
OWNED_MODULES = {
    "backend": BACKEND_ROOT / "backend.py",
    "execution": BACKEND_ROOT / "execution.py",
    "preparation": BACKEND_ROOT / "preparation.py",
    "run_scope": BACKEND_ROOT / "run_scope.py",
}


def _module_tree(name: str) -> ast.Module:
    return ast.parse(OWNED_MODULES[name].read_text(), filename=str(OWNED_MODULES[name]))


def _decision_complexity(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    decision_nodes = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.IfExp,
        ast.Match,
    )
    return 1 + sum(isinstance(node, decision_nodes) for node in ast.walk(function))


def _function_metrics(name: str) -> dict[str, tuple[int, int]]:
    metrics = {}
    for node in ast.walk(_module_tree(name)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            metrics[node.name] = (
                node.end_lineno - node.lineno + 1,
                _decision_complexity(node),
            )
    return metrics


def _owned_imports(name: str) -> set[str]:
    imports = set()
    for node in ast.walk(_module_tree(name)):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        prefix = "circuit_tracer.attribution.nnsight."
        if node.module.startswith(prefix):
            imported = node.module.removeprefix(prefix).split(".", maxsplit=1)[0]
            if imported in OWNED_MODULES:
                imports.add(imported)
    return imports


def test_backend_coordinator_and_owned_operations_remain_bounded() -> None:
    bounds = {
        "backend": (80, 8),
        "execution": (150, 8),
        "preparation": (120, 8),
        "run_scope": (80, 12),
    }
    violations = []
    for module, (max_lines, max_complexity) in bounds.items():
        for function, (lines, complexity) in _function_metrics(module).items():
            if lines > max_lines or complexity > max_complexity:
                violations.append(
                    f"{module}.{function}: {lines} lines/{complexity} complexity "
                    f"(max {max_lines}/{max_complexity})"
                )
    assert not violations, "backend operation grew past its architecture bound:\n" + "\n".join(
        violations
    )


def test_backend_dependency_direction_is_one_way() -> None:
    assert _owned_imports("backend") == {"execution", "preparation", "run_scope"}
    assert _owned_imports("execution") == {"preparation", "run_scope"}
    assert _owned_imports("preparation") == set()
    assert _owned_imports("run_scope") == set()


def test_backend_has_no_private_compatibility_exports() -> None:
    forbidden = {
        "_raise_cleanup_failures",
        "_resolve_phase0_activation_threshold_compare_mode",
        "_resolve_telemetry_max_events",
        "_resolve_phase4_anomaly_debug_enabled",
    }
    definitions = {
        node.name
        for node in _module_tree("backend").body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert definitions.isdisjoint(forbidden)
