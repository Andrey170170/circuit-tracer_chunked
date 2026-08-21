"""Direct contracts for extracted NNSight observability helpers."""

import logging
import torch
from circuit_tracer.attribution.nnsight import telemetry
from circuit_tracer.observability import human_logs
from circuit_tracer.observability.lifecycle import TelemetryObserver


def test_human_log_rendering_keeps_existing_message_shape(caplog):
    logger = logging.getLogger("nnsight-observability-test")
    with caplog.at_level(logging.INFO, logger=logger.name):
        human_logs._log_batch_profile(
            logger, "Phase 3", 2, 4, 1.25, {"calls": 1}, {"calls": 3}, None, None
        )
    assert caplog.messages == ["Phase 3 batch 2/4 in 1.25s | ctx[calls=2]"]


def test_cross_cluster_runtime_snapshot_keeps_hash_and_scalar_shapes():
    class Diagnostics:
        def get_diagnostic_snapshot(self):
            return {"batches": 2}

    summary, stream = telemetry._build_cross_cluster_runtime_snapshot(
        observer=TelemetryObserver.create(),
        device=torch.device("cpu"), ctx=Diagnostics()
    )
    assert summary["ctx_diagnostic_snapshot"] == {"batches": 2}
    assert len(summary["ctx_diagnostic_snapshot_hash"]) == 16
    assert stream["ctx_diagnostic_snapshot_hash"] == summary["ctx_diagnostic_snapshot_hash"]
