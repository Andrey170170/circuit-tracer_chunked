"""Domain preparation for NNSight attribution setup."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Literal, cast

import torch
from nnsight import save

from circuit_tracer.attribution.context_nnsight import AttributionContext
from circuit_tracer.attribution.nnsight.context_state import (
    AttributionTensorState,
    ContextExecutionPolicy,
    ContextNumericPolicy,
    DecoderRuntime,
)
from circuit_tracer.attribution.sparsification import SparsificationConfig
from circuit_tracer.observability.events import MemorySnapshot, TraceObserver
from circuit_tracer.transcoder.provider import (
    TranscoderCapabilities,
    get_transcoder_capabilities,
    provider_fingerprint,
    require_exact_chunked_provider,
)


EncoderResidency = Literal["lazy", "active_cpu", "active_pinned_cpu"]


@dataclass(frozen=True)
class AttributionSetupInput:
    tokens: torch.Tensor
    phase0_tokens: torch.Tensor
    prefix_view_length: int | None

    @classmethod
    def resolve(
        cls, tokens: torch.Tensor, prefix_view_length: int | None
    ) -> "AttributionSetupInput":
        if tokens.ndim != 1:
            raise ValueError("Tokens must be a 1D tensor")
        prefix = None if prefix_view_length is None else int(prefix_view_length)
        if prefix is not None and (prefix <= 0 or prefix > int(tokens.numel())):
            raise ValueError(
                "prefix_view_length must be in [1, token_count] "
                f"({prefix} not in [1, {int(tokens.numel())}])"
            )
        phase0_tokens = tokens if prefix is None else tokens[:prefix].contiguous()
        return cls(tokens=tokens, phase0_tokens=phase0_tokens, prefix_view_length=prefix)


@dataclass(frozen=True)
class Phase0ActivationCapture:
    mlp_inputs: torch.Tensor
    mlp_outputs: torch.Tensor
    logits: torch.Tensor
    elapsed_seconds: float

    @classmethod
    def run(cls, model, tokens: torch.Tensor, trace_event) -> "Phase0ActivationCapture":
        start = time.perf_counter()
        if callable(trace_event):
            trace_event(
                "phase0.setup.trace_start",
                backend="nnsight",
                token_count=int(tokens.numel()),
            )
        with model.trace(tokens):
            mlp_inputs = []
            mlp_outputs = []
            for feature_input_loc, feature_output_loc in zip(
                model.feature_input_locs, model.feature_output_locs
            ):
                mlp_inputs.append(feature_input_loc.output)
                mlp_outputs.append(
                    model.model_adapter.normalize_feature_output(feature_output_loc.output)
                )
            saved_inputs = save(torch.cat(mlp_inputs, dim=0))
            saved_outputs = save(torch.cat(mlp_outputs, dim=0))
            logits = save(model.output.logits)
        elapsed = time.perf_counter() - start
        if callable(trace_event):
            trace_event(
                "phase0.setup.trace_done",
                backend="nnsight",
                elapsed_s=f"{elapsed:.2f}",
                mlp_in_shape=tuple(saved_inputs.shape),
                mlp_out_shape=tuple(saved_outputs.shape),
            )
        return cls(saved_inputs, saved_outputs, logits, elapsed)


@dataclass(frozen=True)
class EncoderResidencyPlan:
    requested: EncoderResidency
    effective: EncoderResidency
    fallback_reason: str | None
    materialize_during_phase0: bool
    stage_on_cpu: bool | None

    @classmethod
    def resolve(
        cls,
        *,
        requested: str,
        exact_chunked_decoder: bool,
        capabilities: TranscoderCapabilities,
        stage_on_cpu: bool | None,
    ) -> "EncoderResidencyPlan":
        normalized = str(requested).strip().lower()
        allowed = {"lazy", "active_cpu", "active_pinned_cpu"}
        if normalized not in allowed:
            raise ValueError(
                f"exact_encoder_residency must be one of: {', '.join(sorted(allowed))} "
                f"(got {requested!r})"
            )
        resolved = cast(EncoderResidency, normalized)
        effective = resolved
        fallback_reason = None
        supported = bool(exact_chunked_decoder and capabilities.supports_exact_encoder_residency)
        if effective != "lazy" and not supported:
            effective = "lazy"
            fallback_reason = (
                "active encoder residency requires exact encoder-residency provider support; "
                "falling back to lazy execution"
            )
        materialize = bool(exact_chunked_decoder and effective != "lazy")
        return cls(
            requested=resolved,
            effective=effective,
            fallback_reason=fallback_reason,
            materialize_during_phase0=materialize,
            stage_on_cpu=True if materialize else stage_on_cpu,
        )


@dataclass(frozen=True)
class AttributionSetupOptions:
    sparsification: SparsificationConfig | None
    retain_full_logits: bool
    chunked_feature_replay_window: int
    error_vector_prefetch_lookahead: int
    stage_encoder_vectors_on_cpu: bool | None
    stage_error_vectors_on_cpu: bool | None
    row_subchunk_size: int | None
    exact_encoder_residency: EncoderResidency
    internal_precision_requested: str | None
    resolved_dtype_map: dict[str, str] | None
    decoder_chunk_cache: object | None
    decoder_cache_fingerprint: object | None
    decoder_active_row_residency: bool
    decoder_active_row_max_bytes: int
    phase0_decoder_row_ranges: bool = False


@dataclass(frozen=True)
class AttributionSetupOperation:
    """Own post-capture feature decomposition and context construction."""

    model: object
    setup_input: AttributionSetupInput
    capture: Phase0ActivationCapture
    options: AttributionSetupOptions
    setup_started_at: float
    phase0_input_fingerprints: dict[str, object] | None
    trace_observer: TraceObserver | None

    def run(self) -> AttributionContext:
        model = self.model
        transcoders = model.transcoders  # type: ignore[attr-defined]
        trace_event = getattr(transcoders, "emit_trace_event", None)
        component_start = time.perf_counter()
        if callable(trace_event):
            trace_event("phase0.setup.components_start", backend="nnsight")
        exact_chunked = require_exact_chunked_provider(transcoders)
        residency = EncoderResidencyPlan.resolve(
            requested=self.options.exact_encoder_residency,
            exact_chunked_decoder=exact_chunked,
            capabilities=get_transcoder_capabilities(transcoders),
            stage_on_cpu=self.options.stage_encoder_vectors_on_cpu,
        )
        component_kwargs = {
            "sparsification": self.options.sparsification,
        }
        if exact_chunked:
            # Keep the Phase-0/Phase-3 CUDA allocation history identical to lazy
            # residency. Active encoders are selected directly into their owned
            # host tier below instead of first building a full occurrence table
            # on CUDA and copying it back to CPU.
            component_kwargs["materialize_encoder_vecs"] = False
        if get_transcoder_capabilities(transcoders).supports_active_decoder_row_residency:
            component_kwargs["decoder_active_row_residency"] = (
                self.options.decoder_active_row_residency
            )
            component_kwargs["decoder_active_row_max_bytes"] = (
                self.options.decoder_active_row_max_bytes
            )
        if get_transcoder_capabilities(transcoders).supports_phase0_decoder_row_ranges:
            component_kwargs["phase0_decoder_row_ranges"] = (
                self.options.phase0_decoder_row_ranges
            )
        components = transcoders.compute_attribution_components(  # type: ignore[attr-defined]
            self.capture.mlp_inputs,
            model.zero_positions,  # type: ignore[attr-defined]
            **component_kwargs,
        )
        if residency.materialize_during_phase0:
            if components.chunked_decoder_state is None:
                raise RuntimeError("active encoder residency requires chunked decoder state")
            materialize_rows = getattr(transcoders, "materialize_encoder_rows", None)
            if not callable(materialize_rows):
                raise RuntimeError(
                    "active encoder residency requires provider encoder-row materialization"
                )
            chunked_state = components.chunked_decoder_state
            components = replace(
                components,
                encoder_vectors=materialize_rows(
                    chunked_state["source_layers"],
                    chunked_state["feature_ids"],
                    device=torch.device("cpu"),
                ),
            )
        if components.decoder_row_seed is not None:
            components = replace(
                components,
                decoder_row_seed=replace(
                    components.decoder_row_seed, source_fingerprint=provider_fingerprint(transcoders)
                ),
            )
        component_seconds = time.perf_counter() - component_start
        if callable(trace_event):
            trace_event(
                "phase0.setup.components_done",
                backend="nnsight",
                elapsed_s=f"{component_seconds:.2f}",
                active_features=components.active_feature_count,
            )

        error_start = time.perf_counter()
        if callable(trace_event):
            trace_event("phase0.setup.error_start", backend="nnsight")
        error_vectors = self.capture.mlp_outputs - components.reconstruction
        error_vectors[:, model.zero_positions] = 0  # type: ignore[attr-defined]
        token_vectors = model.embed_weight[self.setup_input.phase0_tokens].detach()  # type: ignore[attr-defined]
        full_logits = self.capture.logits if self.options.retain_full_logits else None
        retained_logits = self.capture.logits
        if exact_chunked and not self.options.retain_full_logits:
            retained_logits = retained_logits[:, -1:, :].contiguous()
        error_seconds = time.perf_counter() - error_start
        if callable(trace_event):
            trace_event(
                "phase0.setup.error_done", backend="nnsight", elapsed_s=f"{error_seconds:.2f}"
            )

        try:
            memory_device = torch.device(str(model.device))  # type: ignore[attr-defined]
        except (TypeError, RuntimeError, ValueError):
            memory_device = None
        memory_before = None
        if residency.materialize_during_phase0 and self.trace_observer is not None:
            memory_before = cast(
                dict[str, object],
                self.trace_observer.observe(MemorySnapshot(memory_device)),
            )
        context = AttributionContext(
            tensor_state=AttributionTensorState(
                activation_matrix=components.activation_matrix,
                logits=retained_logits,
                full_logits=full_logits,
                error_vectors=error_vectors,
                token_vectors=token_vectors,
                decoder_vectors=components.decoder_vectors,
                encoder_vectors=components.encoder_vectors,
                encoder_to_decoder_map=components.encoder_to_decoder_map,
                decoder_locations=components.decoder_locations,
            ),
            execution_policy=ContextExecutionPolicy.resolve(
                chunked_decoder_state=components.chunked_decoder_state,
                encoder_vectors=components.encoder_vectors,
                error_vectors=error_vectors,
                exact_encoder_residency=residency.effective,
                stage_encoder_vectors_on_cpu=residency.stage_on_cpu,
                stage_error_vectors_on_cpu=self.options.stage_error_vectors_on_cpu,
                error_vector_prefetch_lookahead=self.options.error_vector_prefetch_lookahead,
                chunked_feature_replay_window=self.options.chunked_feature_replay_window,
                row_subchunk_size=self.options.row_subchunk_size,
            ),
            decoder_runtime=DecoderRuntime.resolve(
                provider=transcoders if exact_chunked else None,
                chunked_state=components.chunked_decoder_state,
                chunk_cache=self.options.decoder_chunk_cache,
                cache_fingerprint=self.options.decoder_cache_fingerprint,
                decoder_row_seed=components.decoder_row_seed,
                decoder_row_seed_refusal_reason=(components.decoder_row_seed_refusal_reason),
                decoder_row_seed_estimated_bytes=(components.decoder_row_seed_estimated_bytes),
            ),
            numeric_policy=ContextNumericPolicy(
                materialized_encoder_vectors_during_phase0=residency.materialize_during_phase0,
                internal_precision_requested=self.options.internal_precision_requested,
                resolved_dtype_map=self.options.resolved_dtype_map,
            ),
        )
        memory_after_stage = None
        if residency.materialize_during_phase0 and self.trace_observer is not None:
            memory_after_stage = cast(
                dict[str, object],
                self.trace_observer.observe(MemorySnapshot(memory_device)),
            )

        context.setup_diagnostic_stats = {
            "backend": "nnsight",
            "token_count": int(self.setup_input.tokens.numel()),
            "phase0_token_count": int(self.setup_input.phase0_tokens.numel()),
            "prefix_view_length": self.setup_input.prefix_view_length,
            "trace_seconds": self.capture.elapsed_seconds,
            "component_seconds": component_seconds,
            "error_seconds": error_seconds,
            "setup_total_seconds": time.perf_counter() - self.setup_started_at,
            "mlp_in_shape": tuple(self.capture.mlp_inputs.shape),
            "mlp_out_shape": tuple(self.capture.mlp_outputs.shape),
            "reconstruction_shape": tuple(components.reconstruction.shape),
            "active_features": components.active_feature_count,
            "logit_retention": context.logit_retention,
            "chunked_feature_replay_window": self.options.chunked_feature_replay_window,
            "error_vector_prefetch_lookahead": self.options.error_vector_prefetch_lookahead,
            "stage_encoder_vecs_on_cpu": self.options.stage_encoder_vectors_on_cpu,
            "stage_encoder_vecs_on_cpu_effective": residency.stage_on_cpu,
            "stage_error_vectors_on_cpu": self.options.stage_error_vectors_on_cpu,
            "row_subchunk_size": self.options.row_subchunk_size,
            "exact_encoder_residency_requested": residency.requested,
            "exact_encoder_residency_effective": residency.effective,
            "exact_encoder_residency_fallback_reason": residency.fallback_reason,
            "exact_encoder_staging_destination": context.exact_encoder_staging_destination,
            "exact_encoder_materialized_during_phase0": residency.materialize_during_phase0,
            "active_encoder_shape": tuple(context.encoder_vecs.shape),
            "active_encoder_bytes": int(
                context.encoder_vecs.numel() * context.encoder_vecs.element_size()
            ),
            "exact_encoder_pinned_requested": context.exact_encoder_pinned_requested,
            "exact_encoder_pinned_effective": context.exact_encoder_pinned_effective,
            "exact_encoder_pinning_success": context.exact_encoder_pinning_success,
            "exact_encoder_pinning_failure_reason": context.exact_encoder_pinning_failure_reason,
            "exact_encoder_gpu_memory_before_stage": memory_before,
            "exact_encoder_gpu_memory_after_stage": memory_after_stage,
            "exact_encoder_gpu_memory_after_free": memory_after_stage,
            "internal_precision_requested": self.options.internal_precision_requested,
            "resolved_dtype_map": self.options.resolved_dtype_map,
            "phase0_pre_clt_input_fingerprints": self.phase0_input_fingerprints,
        }
        context.sparsification_stats = components.sparsification_stats
        return context
