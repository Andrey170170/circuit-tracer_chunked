"""Direct contracts for extracted NNSight observability helpers."""

import logging
import torch
from circuit_tracer.attribution import attribute_nnsight
from circuit_tracer.attribution.nnsight import telemetry
from circuit_tracer.observability import exception_export, human_logs


def test_attribute_nnsight_reexports_extracted_helpers():
    assert (
        attribute_nnsight._attach_telemetry_export_to_exception
        is exception_export._attach_telemetry_export_to_exception
    )
    assert attribute_nnsight._log_batch_profile is human_logs._log_batch_profile
    assert (
        attribute_nnsight._build_phase4_refresh_substage_telemetry
        is telemetry._build_phase4_refresh_substage_telemetry
    )
    assert (
        attribute_nnsight._record_cross_cluster_checkpoint
        is telemetry._record_cross_cluster_checkpoint
    )


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
        device=torch.device("cpu"), ctx=Diagnostics()
    )
    assert summary["ctx_diagnostic_snapshot"] == {"batches": 2}
    assert len(summary["ctx_diagnostic_snapshot_hash"]) == 16
    assert stream["ctx_diagnostic_snapshot_hash"] == summary["ctx_diagnostic_snapshot_hash"]
