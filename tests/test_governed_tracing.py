from __future__ import annotations

from dataclasses import replace

import pytest

from circuit_tracer.governor import RECORDED_PROVIDER_PROFILES, ResourceEnvelope
from circuit_tracer.tracing import (
    AttributionProblem,
    PlanningRefusedError,
    TraceRequest,
    TraceSemantics,
    open_session,
    resolve_trace_request,
    trace_batch,
)
from circuit_tracer.transcoder.provider import TranscoderCapabilities


GIB = 1024**3


class Provider:
    def __init__(self, profile) -> None:
        identity = profile.identity
        dimensions = profile.dimensions
        self.n_layers = dimensions.n_layers
        self.d_model = dimensions.d_model
        self.d_transcoder = dimensions.d_features
        self.capabilities = TranscoderCapabilities(
            architecture=identity.architecture,
            checkpoint_format="fixture",
            supports_decoder_chunk_cache=True,
            decoder_output_topology=identity.decoder_topology.value,
        )

    def create_decoder_block_cache(self, max_bytes=None, *, fingerprint=None):
        return (max_bytes, fingerprint)

    def clear_decoder_block_cache(self, cache) -> None:
        del cache


class Model:
    backend = "nnsight"

    def __init__(self, profile) -> None:
        self.transcoders = Provider(profile)


def envelope(*, disk: int = 100 * GIB, vram: int = 141 * GIB) -> ResourceEnvelope:
    return ResourceEnvelope(
        total_vram_bytes=vram,
        host_budget_bytes=800 * GIB,
        file_cache_allowance_bytes=64 * GIB,
        local_disk_bytes=disk,
        scratch_disk_bytes=disk,
        walltime_seconds=10**30,
    )


def request(profile, *, batch: int, prompt_tokens: int = 16) -> TraceRequest:
    return TraceRequest(
        problem=AttributionProblem(model=Model(profile), prompt=list(range(prompt_tokens))),
        semantics=TraceSemantics(
            source_batch_size=batch,
            feature_batch_size=batch,
            logit_batch_size=batch,
            max_feature_nodes=100,
        ),
    )


def test_governed_clt_compiles_to_equivalent_explicit_c2_plan() -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    selected = request(profile, batch=1000)
    governed = resolve_trace_request(
        selected, resources=envelope(), provider_profile=profile
    )
    explicit = resolve_trace_request(replace(selected, execution=governed.execution))

    assert governed.semantic_fingerprint == explicit.semantic_fingerprint
    assert governed.execution_fingerprint == explicit.execution_fingerprint
    assert governed.semantics.source_batch_size == 1000
    assert governed.execution.session.capacity == 128
    assert governed.execution.session.source_microbatch_max_rows == 128
    assert governed.execution.session.phase3_microbatch_max_rows == 128
    assert governed.execution.session.phase4_microbatch_max_rows == 128
    assert governed.execution.session.decoder_cache.max_bytes == 8 * GIB
    assert governed.execution.decoder.fetch_chunk_size == 4096
    assert governed.execution.replay.feature_window == 4
    assert governed.execution.replay.error_vector_prefetch_lookahead == 2
    assert governed.execution.storage.exact_encoder_residency == "lazy"


@pytest.mark.parametrize(
    ("disk", "expected"),
    [(2_000_000, "full_file"), (900_000, "column_tiled_v1"), (1, "none_recompute")],
)
def test_constrained_envelopes_select_validated_storage_rungs(disk, expected) -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    plan = resolve_trace_request(
        request(profile, batch=1000),
        resources=envelope(disk=disk),
        provider_profile=profile,
    )
    actual = (
        plan.execution.storage.retention
        if plan.execution.storage.retention == "none_recompute"
        else plan.execution.storage.full_retention_backend
    )
    assert actual == expected
    if expected == "column_tiled_v1":
        assert plan.execution.storage.feature_column_tile_size == 2048


def test_provider_mismatch_and_planning_refusal_fail_closed() -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_clt_b1000_c4096_cache8"]
    selected = request(profile, batch=1000)
    selected.problem.model.transcoders.capabilities = replace(
        selected.problem.model.transcoders.capabilities, architecture="plt"
    )
    with pytest.raises(ValueError, match="provider profile mismatch"):
        resolve_trace_request(selected, resources=envelope(), provider_profile=profile)

    admitted = request(profile, batch=1000)
    with pytest.raises(PlanningRefusedError, match="planning refused"):
        resolve_trace_request(
            admitted, resources=envelope(vram=GIB), provider_profile=profile
        )


def test_batch_and_session_pin_and_propagate_governed_inputs(monkeypatch) -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_plt_b128_c4096_cache0"]
    resources = envelope()
    selected = request(profile, batch=128)

    def execute(problem, plan, *, observer, forward_overrides, execution_identity):
        del observer, forward_overrides
        execution_identity.mark_requested_as_effective()
        return (problem.prompt[0], plan.execution.session.capacity)

    monkeypatch.setattr(
        "circuit_tracer.attribution.nnsight.backend.run_nnsight_trace", execute
    )
    results = trace_batch(
        [selected], resources=resources, provider_profile=profile
    )
    assert results[0].output == (0, 64)

    session = open_session(
        selected, resources=resources, provider_profile=profile
    )
    assert session.resources is resources
    assert session.provider_profile is profile
    assert session.trace().output == (0, 64)
    session.close()


@pytest.mark.parametrize("api", [resolve_trace_request, trace_batch, open_session])
def test_public_governed_inputs_must_be_paired(api) -> None:
    profile = RECORDED_PROVIDER_PROFILES["granite_h200_1b_plt_b128_c4096_cache0"]
    selected = request(profile, batch=128)
    value = [selected] if api is trace_batch else selected
    with pytest.raises(ValueError, match="must be supplied together"):
        api(value, resources=envelope())
