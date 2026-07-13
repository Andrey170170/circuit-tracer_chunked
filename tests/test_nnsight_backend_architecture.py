"""Architecture bounds for the decomposed NNSight backend."""

from __future__ import annotations

import ast
from pathlib import Path

BACKEND_ROOT = Path("circuit_tracer/attribution/nnsight")
PHASE_ROOT = BACKEND_ROOT / "phases"
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


def test_execution_does_not_rebind_frozen_prepared_collections() -> None:
    violations = [
        node.lineno
        for node in ast.walk(_module_tree("execution"))
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Attribute)
        and node.target.attr == "offload_handles"
    ]
    assert not violations, (
        "PreparedBackend is frozen; mutate offload_handles with extend rather than "
        f"rebinding it via augmented assignment at lines {violations}"
    )


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


def test_phases_depend_only_on_typed_observability_boundary() -> None:
    forbidden_modules = {
        "circuit_tracer.observability.human_logs",
        "circuit_tracer.observability.recorder",
        "circuit_tracer.observability.resources",
        "circuit_tracer.utils.telemetry",
    }
    violations = []
    for path in sorted(PHASE_ROOT.glob("phase[0-5].py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        bad_imports = sorted(imports & forbidden_modules)
        raw_recorder_names = sorted(
            {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and "telemetry_recorder" in node.id
            }
        )
        if bad_imports or raw_recorder_names:
            violations.append(
                f"{path.name}: imports={bad_imports}, raw_recorders={raw_recorder_names}"
            )
    assert not violations, "phase observability boundary violations:\n" + "\n".join(
        violations
    )


def test_canonical_nnsight_modules_cannot_reach_raw_telemetry_recorder() -> None:
    paths = [
        *BACKEND_ROOT.rglob("*.py"),
        Path("circuit_tracer/attribution/context_nnsight.py"),
        Path("circuit_tracer/replacement_model/attribution_setup.py"),
        Path("circuit_tracer/replacement_model/replacement_model_nnsight.py"),
        *Path("circuit_tracer/transcoder").glob("*.py"),
    ]
    violations = []
    for path in sorted(set(paths)):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        bad_imports = sorted(
            imports
            & {
                "circuit_tracer.observability.human_logs",
                "circuit_tracer.observability.recorder",
                "circuit_tracer.observability.resources",
                "circuit_tracer.utils.telemetry",
            }
        )
        raw_names = sorted(
            {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and "TelemetryRecorder" in node.id
            }
        )
        recorder_accesses = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "recorder"
        ]
        if bad_imports or raw_names or recorder_accesses:
            violations.append(
                f"{path}: imports={bad_imports}, raw_names={raw_names}, "
                f"recorder_accesses={recorder_accesses}"
            )
    assert not violations, "raw telemetry capability leaks:\n" + "\n".join(violations)


def test_terminal_lifecycle_is_not_owned_by_nnsight_run_scope() -> None:
    source = OWNED_MODULES["run_scope"].read_text()
    assert "attribute.done" not in source
    assert "attribute.failed" not in source
    assert "close_export" not in source
    assert "attach_exception" not in source


def test_observability_has_no_private_observer_compatibility_alias() -> None:
    lifecycle = Path("circuit_tracer/observability/lifecycle.py").read_text()
    tree = ast.parse(lifecycle)
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    assert "_TelemetryObserver" not in assigned_names
