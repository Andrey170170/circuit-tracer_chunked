"""Architecture bounds for the decomposed NNSight Phase 0."""

from __future__ import annotations

import ast
from pathlib import Path


PHASE_ROOT = Path("circuit_tracer/attribution/nnsight/phases")
COORDINATOR = PHASE_ROOT / "phase0.py"
OPERATION_MODULES = sorted(PHASE_ROOT.glob("phase0_*.py"))
PHASE0_MODULES = [COORDINATOR, *OPERATION_MODULES]


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _function_sizes(path: Path) -> dict[str, int]:
    return {
        node.name: node.end_lineno - node.lineno + 1
        for node in ast.walk(_tree(path))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _phase0_imports(path: Path) -> set[str]:
    prefix = "circuit_tracer.attribution.nnsight.phases."
    return {
        node.module.removeprefix(prefix)
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith(f"{prefix}phase0")
    }


def test_phase0_coordinator_and_operations_remain_bounded() -> None:
    coordinator_sizes = _function_sizes(COORDINATOR)
    assert coordinator_sizes["run_phase0"] <= 120

    violations = []
    for path in PHASE0_MODULES:
        for function, lines in _function_sizes(path).items():
            if lines > 180:
                violations.append(f"{path.name}.{function}: {lines} lines")
    assert not violations, "Phase 0 operation grew past 180 lines:\n" + "\n".join(violations)


def test_phase0_operation_dependencies_are_one_way() -> None:
    operation_names = {path.stem for path in OPERATION_MODULES}
    assert _phase0_imports(COORDINATOR) == operation_names

    dependencies = {path.stem: _phase0_imports(path) for path in OPERATION_MODULES}
    assert dependencies == {
        "phase0_activation": set(),
        "phase0_cleanup": set(),
        "phase0_context": {"phase0_tokens"},
        "phase0_evidence": set(),
        "phase0_tokens": set(),
    }


def test_phase0_operations_use_only_the_typed_observer_boundary() -> None:
    forbidden_modules = {
        "circuit_tracer.observability.human_logs",
        "circuit_tracer.observability.recorder",
        "circuit_tracer.observability.resources",
        "circuit_tracer.utils.telemetry",
    }
    violations = []
    for path in PHASE0_MODULES:
        tree = _tree(path)
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        raw_recorder_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and "telemetry_recorder" in node.id
        }
        if imports & forbidden_modules or raw_recorder_names:
            violations.append(path.name)
    assert not violations, "Phase 0 bypassed the typed observer boundary: " + ", ".join(violations)
