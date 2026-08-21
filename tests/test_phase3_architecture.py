"""Architecture boundaries for the decomposed NNSight Phase 3 runtime."""

from __future__ import annotations

import ast
from pathlib import Path


PHASES = Path("circuit_tracer/attribution/nnsight/phases")
ENTRYPOINT = PHASES / "phase3.py"
OWNERS = sorted(PHASES.glob("phase3_*.py"))
FORBIDDEN_PHASE_DEPENDENCIES = {
    f"circuit_tracer.attribution.nnsight.phases.phase{phase}"
    for phase in (0, 2, 4, 5)
}
FORBIDDEN_RUNTIME_DEPENDENCIES = {
    "circuit_tracer.attribution.nnsight.execution",
    "circuit_tracer.attribution.nnsight.preparation",
    "circuit_tracer.attribution.nnsight.session",
    "circuit_tracer.tracing",
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def test_phase3_entrypoint_is_a_small_domain_orchestrator() -> None:
    tree = _tree(ENTRYPOINT)
    run_phase3 = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_phase3"
    )
    assert run_phase3.end_lineno is not None
    assert run_phase3.end_lineno - run_phase3.lineno + 1 <= 150
    calls = {
        node.func.id
        for node in ast.walk(run_phase3)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "run_logit_batches",
        "package_phase3_replay_evidence",
        "select_phase3_frontier",
    } <= calls


def test_phase3_operations_remain_bounded_and_acyclic() -> None:
    violations: list[str] = []
    forbidden = FORBIDDEN_PHASE_DEPENDENCIES | FORBIDDEN_RUNTIME_DEPENDENCIES
    for path in [ENTRYPOINT, *OWNERS]:
        tree = _tree(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.end_lineno is not None
                size = node.end_lineno - node.lineno + 1
                if size > 200:
                    violations.append(f"{path}:{node.lineno} {node.name} is {size} lines")
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "circuit_tracer.attribution.nnsight.phases.phase3":
                    violations.append(f"{path}:{node.lineno} imports the Phase 3 facade")
                if any(
                    node.module == prefix or node.module.startswith(f"{prefix}.")
                    for prefix in forbidden
                ):
                    violations.append(f"{path}:{node.lineno} imports {node.module}")
    assert not violations, "Phase 3 ownership violations:\n" + "\n".join(violations)
