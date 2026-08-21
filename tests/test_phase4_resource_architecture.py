"""Architecture boundary for Phase-4 resource probes."""

from __future__ import annotations

import ast
from pathlib import Path


NNSIGHT_ROOT = Path("circuit_tracer/attribution/nnsight")
CANONICAL_POLICY_AND_PHASES = [
    NNSIGHT_ROOT / "phase4_policy.py",
    *sorted((NNSIGHT_ROOT / "phases").glob("phase[0-5].py")),
]


def test_canonical_policy_and_phase_modules_do_not_access_cuda_resources_directly() -> None:
    violations = []
    for path in CANONICAL_POLICY_AND_PHASES:
        tree = ast.parse(path.read_text(), filename=str(path))
        cuda_accesses = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "torch"
            and node.value.attr == "cuda"
        ]
        if cuda_accesses:
            violations.append(f"{path}: torch.cuda at lines {cuda_accesses}")

    assert not violations, "CUDA resources must use typed observability probes:\n" + "\n".join(
        violations
    )
