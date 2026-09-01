from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from nnsight import NNsight
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig

from circuit_tracer.replacement_model.model_adapter import Gemma3NNSightModelAdapter
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.transcoder.provider import TranscoderCapabilities
from circuit_tracer.verification import (
    FeatureNode,
    FeatureValue,
    InterventionExecutionRequest,
    InterventionSemantics,
    InterventionVariant,
    OrderingAdmissionMode,
    PreactivationIntervention,
    RuntimeExecutionStatus,
    RuntimeRefusal,
    TargetState,
    TraceIdentity,
    VariantKind,
    diagnose_propagated_ordering,
)
from circuit_tracer.verification.nnsight_runtime import (
    NNSightInterventionRuntime,
    QualificationExecutionCapture,
)
from circuit_tracer.verification.ordering_qualification import (
    Gemma3PLTEagerHookOracle,
    OrderingQualificationRequest,
    OrderingQualificationVerdict,
    qualify_nnsight_ordering,
    validate_ordering_qualification_receipt,
    validate_serialized_ordering_qualification_receipt,
)


class _IdentityPLTProvider:
    capabilities = TranscoderCapabilities(
        architecture="plt",
        checkpoint_format="tiny-test",
        decoder_output_topology="same_layer",
    )
    d_transcoder = 8
    skip_connection = False

    def __init__(self) -> None:
        self.encoded_shapes: list[tuple[int, tuple[int, ...]]] = []

    def encode_layer(self, values, layer, *, apply_activation_function):
        del apply_activation_function
        self.encoded_shapes.append((layer, tuple(values.shape)))
        return values.clone()

    @staticmethod
    def apply_activation_function_to_feature(layer, feature, values):
        del layer
        mask = torch.nn.functional.one_hot(
            torch.tensor(feature, device=values.device), num_classes=8
        )
        return torch.relu(values.unsqueeze(-1) * mask)[..., feature]

    @staticmethod
    def _get_decoder_vectors(layer, feature_ids):
        del layer
        vectors = torch.zeros(
            len(feature_ids), 8, dtype=torch.bfloat16, device=feature_ids.device
        )
        vectors[:, 0] = 2
        vectors[:, 1] = -1
        return vectors


class _ShapeSensitivePLTProvider(_IdentityPLTProvider):
    """Expose the native full-sequence versus selected-position shape drift."""

    def encode_layer(self, values, layer, *, apply_activation_function):
        del apply_activation_function
        self.encoded_shapes.append((layer, tuple(values.shape)))
        return values + values.shape[-2]


class _TinyGemma3PLTHost:
    backend = "nnsight"
    verification_intervened_capture_ordering_qualified = False
    zero_positions = slice(0, 0)
    skip_transcoder = False

    def __init__(self, *, n_layers: int = 2) -> None:
        torch.manual_seed(7)
        text = Gemma3TextConfig(
            vocab_size=17,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=n_layers,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            layer_types=["sliding_attention"] * n_layers,
            use_cache=False,
            attention_dropout=0.0,
        )
        text._attn_implementation = "eager"
        config = Gemma3Config(
            text_config=text,
            vision_config={
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 14,
                "patch_size": 14,
            },
            image_token_index=16,
        )
        config._attn_implementation = "eager"
        self.model = NNsight(
            Gemma3ForConditionalGeneration(config).eval().to(dtype=torch.bfloat16)
        )
        self.config = config
        self.cfg = SimpleNamespace(n_layers=n_layers)
        self.n_layers = n_layers
        self.transcoders = _IdentityPLTProvider()
        self.dtype = torch.bfloat16
        self.device = torch.device("cpu")
        self.model_adapter = Gemma3NNSightModelAdapter.create(
            architecture="Gemma3ForConditionalGeneration", has_chat_template=False
        )

    def trace(self):
        return self.model.trace()

    @contextmanager
    def _verification_probe_scope(self):
        yield

    @contextmanager
    def zero_softcap(self):
        yield

    def _verification_tokens(self, prompt_token_ids):
        return torch.tensor((prompt_token_ids,), dtype=torch.long)

    @property
    def output(self):
        return SimpleNamespace(logits=self.model.output.logits)

    def get_feature_input_loc(self, layer):
        return self.model.model.language_model.layers[layer].pre_feedforward_layernorm

    def get_feature_output_loc(self, layer):
        return self.model.model.language_model.layers[layer].post_feedforward_layernorm

    @property
    def feature_input_locs(self):
        return tuple(self.get_feature_input_loc(layer) for layer in range(self.n_layers))

    @property
    def feature_output_locs(self):
        return tuple(self.get_feature_output_loc(layer) for layer in range(self.n_layers))

    @property
    def attention_locs(self):
        return self._attention_locations()

    def _attention_locations(self):
        layers = self.model.model.language_model.layers
        for layer in range(self.n_layers):
            yield (
                layers[layer]
                .self_attn.source.attention_interface_0.source.nn_functional_dropout_0
            )

    @property
    def layernorm_scale_locs(self):
        ordinary = [
            self._ordinary_norm_locations(name)
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            )
        ]
        attention = [
            self._attention_norm_locations(name)
            for name in ("q_norm", "k_norm")
        ]
        return ordinary + attention + [self._final_norm_locations()]

    def _ordinary_norm_locations(self, name):
        layers = self.model.model.language_model.layers
        for layer in range(self.n_layers):
            yield (
                getattr(layers[layer], name)
                .source.self__norm_0.source.torch_rsqrt_0
            )

    def _attention_norm_locations(self, name):
        layers = self.model.model.language_model.layers
        for layer in range(self.n_layers):
            yield (
                getattr(layers[layer].self_attn, name)
                .source.self__norm_0.source.torch_rsqrt_0
            )

    def _final_norm_locations(self):
        yield self.model.model.language_model.norm.source.self__norm_0.source.torch_rsqrt_0

    _verification_encode_layers = NNSightReplacementModel._verification_encode_layers
    _verification_save_feature_values = staticmethod(
        NNSightReplacementModel._verification_save_feature_values
    )
    _verification_save_selected_feature_inputs = (
        NNSightReplacementModel._verification_save_selected_feature_inputs
    )
    _verification_capture_baseline = NNSightReplacementModel._verification_capture_baseline
    _verification_freeze_attention = NNSightReplacementModel._verification_freeze_attention
    _verification_freeze_layernorm_group = (
        NNSightReplacementModel._verification_freeze_layernorm_group
    )
    _verification_compute_skip_diffs = NNSightReplacementModel._verification_compute_skip_diffs
    _verification_freeze_feature_output = (
        NNSightReplacementModel._verification_freeze_feature_output
    )
    _verification_inject = NNSightReplacementModel._verification_inject
    _verification_run_variant = NNSightReplacementModel._verification_run_variant
    _verification_release = NNSightReplacementModel._verification_release
    _verification_health_check = NNSightReplacementModel._verification_health_check


def _request() -> OrderingQualificationRequest:
    source = FeatureNode(0, 1, 0)
    downstream = FeatureNode(1, 1, 1)
    predicted = (FeatureValue(downstream, 0.0),)
    variants = (
        InterventionVariant(
            "no_op", VariantKind.NO_OP, InterventionSemantics.DIRECT_FROZEN, (), 0.0
        ),
        InterventionVariant(
            "direct",
            VariantKind.DIRECT_DOUBLE,
            InterventionSemantics.DIRECT_FROZEN,
            (PreactivationIntervention(source, 5.0, 0.0, 5.0),),
            None,
            predicted,
        ),
        InterventionVariant(
            "propagated",
            VariantKind.NECESSITY_HIGH,
            InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
            (PreactivationIntervention(source, 5.0, 0.0, 5.0),),
            None,
            predicted,
        ),
    )
    execution = InterventionExecutionRequest(
        TraceIdentity(
            "tiny-ordering", "graph", "provider", "semantic", "execution", (1, 2, 3), 3, 4
        ),
        TargetState(),
        variants,
        (downstream,),
        120.0,
        1.0,
        1.0,
        1.0,
    )
    return OrderingQualificationRequest.from_execution_requests(
        execution_requests=(execution,),
        scope={"model_family": "gemma3", "provider_architecture": "plt"},
        provenance={"repeat_index": 1},
    )


class _StaticRuntime:
    def __init__(self, result, *, capture_result=None) -> None:
        self.result = result
        self.capture_result = capture_result or result

    def evaluate(self, request):
        del request
        return self.result

    def evaluate_for_ordering_qualification(
        self, request, *, selected_feature_input_nodes
    ):
        del request
        baseline_values = (
            {
                item.node: item.preactivation
                for item in self.capture_result.baseline.feature_values
            }
            if self.capture_result.baseline is not None
            else {}
        )
        runs = {"baseline": baseline_values}
        for observation in self.capture_result.observations:
            values = dict(baseline_values)
            values.update(
                (item.node, item.preactivation)
                for item in (
                    *observation.downstream_feature_values,
                    *observation.intervention_feature_values,
                )
            )
            runs[observation.variant_id] = values
        captures = {}
        for run_id, values in runs.items():
            vectors = []
            for layer, position in sorted(
                {(node.layer, node.position) for node in selected_feature_input_nodes}
            ):
                vector = torch.zeros(8, dtype=torch.bfloat16)
                for node in selected_feature_input_nodes:
                    if node.layer == layer and node.position == position:
                        vector[node.feature] = values.get(node, baseline_values.get(node, 0.0))
                vectors.append((layer, position, vector))
            captures[run_id] = tuple(vectors)
        return QualificationExecutionCapture(self.result, captures)


class _CaptureMutatingRuntime(_StaticRuntime):
    def __init__(self, result, mutation: str) -> None:
        super().__init__(result)
        self.mutation = mutation

    def evaluate_for_ordering_qualification(
        self, request, *, selected_feature_input_nodes
    ):
        capture = super().evaluate_for_ordering_qualification(
            request,
            selected_feature_input_nodes=selected_feature_input_nodes,
        )
        values = list(capture.selected_feature_inputs["baseline"])
        layer, position, vector = values[0]
        if self.mutation == "missing":
            values.pop()
        elif self.mutation == "duplicate":
            values.append((layer, position, vector.clone()))
        elif self.mutation == "wrong_shape":
            values[0] = (layer, position, vector.unsqueeze(0))
        elif self.mutation == "nonfinite":
            changed = vector.clone()
            changed[0] = torch.nan
            values[0] = (layer, position, changed)
        elif self.mutation == "shift":
            values[0] = (layer, position, vector + 1)
        else:
            raise AssertionError(self.mutation)
        capture.selected_feature_inputs["baseline"] = tuple(values)
        return capture


def test_model_backed_gate_qualifies_independent_oracle_against_production() -> None:
    receipt = qualify_nnsight_ordering(_TinyGemma3PLTHost(), _request())

    assert receipt.result.verdict is OrderingQualificationVerdict.QUALIFIED
    assert receipt.result.comparisons
    assert all(comparison.passed for comparison in receipt.result.comparisons)
    assert receipt.to_dict()["status"] == "qualified"
    assert (
        receipt.to_dict()["qualification_claim"]
        == "intervened_forward_capture_ordering_with_joint_canonical_feature_projection"
    )
    assert receipt.request_evidence["qualification_policy"] == {
        "propagated_mutation": "graph_pinned_preactivation_v1",
        "feature_input_capture": "selected_feature_input_vectors_v1",
        "feature_projection": "joint_canonical_provider_preactivation_v1",
        "native_feature_values": "diagnostic_only_v1",
    }
    assert receipt.qualification_fingerprint.startswith("sha256:")
    validate_ordering_qualification_receipt(_request(), receipt)
    validate_serialized_ordering_qualification_receipt(receipt.to_dict())


def test_gate_jointly_projects_selected_hidden_vectors_despite_native_shape_drift() -> None:
    host = _TinyGemma3PLTHost()
    host.transcoders = _ShapeSensitivePLTProvider()

    receipt = qualify_nnsight_ordering(host, _request())

    assert receipt.result.verdict is OrderingQualificationVerdict.QUALIFIED
    assert receipt.to_dict()["qualification_claim"] == (
        "intervened_forward_capture_ordering_with_joint_canonical_feature_projection"
    )
    assert receipt.request_evidence["qualification_policy"] == {
        "propagated_mutation": "graph_pinned_preactivation_v1",
        "feature_input_capture": "selected_feature_input_vectors_v1",
        "feature_projection": "joint_canonical_provider_preactivation_v1",
        "native_feature_values": "diagnostic_only_v1",
    }
    assert receipt.result.native_feature_diagnostics
    assert any(not item.passed for item in receipt.result.native_feature_diagnostics)
    assert all(item.passed for item in receipt.result.comparisons)
    canonical_shape = 4 * (1 + len(_request().execution.variants))
    joint_calls = [
        (layer, shape)
        for layer, shape in host.transcoders.encoded_shapes
        if shape[1] == canonical_shape
    ]
    assert joint_calls == [(0, (1, canonical_shape, 8)), (1, (1, canonical_shape, 8))]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_shape", "nonfinite"])
def test_gate_refuses_invalid_selected_feature_input_capture(mutation: str) -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)

    receipt = qualify_nnsight_ordering(
        host,
        request,
        selective_runtime=_CaptureMutatingRuntime(production, mutation),
        oracle_runtime=_StaticRuntime(production),
    )

    assert receipt.result.verdict is OrderingQualificationVerdict.REFUSED
    assert receipt.result.reasons == ("canonical_projection_refused",)


def test_gate_rejects_mutated_selective_hidden_vector() -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)

    receipt = qualify_nnsight_ordering(
        host,
        request,
        selective_runtime=_CaptureMutatingRuntime(production, "shift"),
        oracle_runtime=_StaticRuntime(production),
    )

    assert receipt.result.verdict is OrderingQualificationVerdict.REJECTED
    assert any(
        reason.startswith("comparison_failed:baseline.feature_input")
        for reason in receipt.result.reasons
    )


class _RecordingQualificationHost(_TinyGemma3PLTHost):
    def __init__(self) -> None:
        super().__init__()
        self.qualification_plans = []

    def _verification_run_variant(self, prompt_token_ids, plan, baseline_state, **kwargs):
        self.qualification_plans.append(plan)
        return NNSightReplacementModel._verification_run_variant(
            self,
            prompt_token_ids,
            plan,
            baseline_state,
            **kwargs,
        )


def test_public_gate_executes_graph_pinned_propagated_mutation_in_selective_runtime() -> None:
    host = _RecordingQualificationHost()
    request = _request()

    qualify_nnsight_ordering(host, request)

    propagated = tuple(
        plan for plan in host.qualification_plans if plan.variant_id == "propagated"
    )
    assert len(propagated) == 2
    assert all(plan.interventions[0].exact_graph_delta == 5.0 for plan in propagated)
    assert all(plan.retain_live_source_for_schedule for plan in propagated)


def test_qualification_refuses_propagated_mutation_without_graph_evidence() -> None:
    request = _request()
    variants = list(request.execution.variants)
    propagated = variants[2]
    source = propagated.interventions[0].node
    variants[2] = replace(
        propagated,
        interventions=(PreactivationIntervention(source, 5.0),),
    )

    with pytest.raises(
        ValueError,
        match="qualification propagated mutation requires graph baseline and delta",
    ):
        OrderingQualificationRequest(
            replace(request.execution, variants=tuple(variants)),
            scope=request.scope,
        )


def test_propagation_diagnostic_runs_common_delta_through_production_engines() -> None:
    receipt = diagnose_propagated_ordering(_TinyGemma3PLTHost(), _request())

    assert receipt.result.status == "complete"
    assert receipt.result.layer_comparisons
    assert receipt.result.first_common_delta_material_divergence_layer is None
    assert all(
        item.native_injection_identity.max_abs_error == 0.0
        for item in receipt.result.source_writes
    )
    assert all(
        item.common_injection_identity.max_abs_error == 0.0
        for item in receipt.result.source_writes
    )


def test_receipt_validators_reject_typed_and_serialized_tampering() -> None:
    request = _request()
    receipt = qualify_nnsight_ordering(_TinyGemma3PLTHost(), request)

    with pytest.raises(ValueError, match="qualification fingerprint"):
        validate_ordering_qualification_receipt(
            request,
            replace(receipt, qualification_fingerprint="sha256:tampered"),
        )

    payload = receipt.to_dict()
    payload["request_evidence"]["variants"][1]["variant_id"] = "tampered"
    with pytest.raises(ValueError, match="request fingerprint"):
        validate_serialized_ordering_qualification_receipt(payload)

    with pytest.raises(ValueError, match="must use current schema"):
        validate_ordering_qualification_receipt(
            request,
            replace(receipt, schema_version=1),
        )

    payload = receipt.to_dict()
    payload["request_evidence"]["qualification_policy"]["propagated_mutation"] = (
        "native_live_activation_v1"
    )
    with pytest.raises(ValueError, match="propagated mutation policy"):
        validate_serialized_ordering_qualification_receipt(payload)

    payload = receipt.to_dict()
    payload["request_evidence"]["execution_policy"]["deadline_seconds"] = 1.0
    with pytest.raises(ValueError, match="request fingerprint"):
        validate_serialized_ordering_qualification_receipt(payload)


def test_offline_validator_preserves_historical_v1_receipt_verification() -> None:
    request_evidence = {
        "scope": {},
        "scope_bindings": {},
        "tolerance": {},
        "identity": {},
        "target": {},
        "variants": [],
        "observed_downstream_nodes": [],
        "execution_policy": {},
    }
    legacy = {
        "schema": "nnsight_intervened_forward_ordering_qualification",
        "schema_version": 1,
        "qualification_claim": "intervened_forward_capture_ordering_only",
        "status": "qualified",
        "scope": {},
        "scope_bindings": {},
        "result": {"verdict": "qualified"},
        "provenance": {},
        "request_evidence": request_evidence,
        "request_fingerprint": (
            "sha256:9c8a665dc460b277afdf9f19d344a2f1440bcbdec1ac5b7bd7d760d09134182c"
        ),
        "qualification_fingerprint": (
            "sha256:7e26ddf409ab4610e7d83a66b2655d7a0f84755ce4dbaf796af2a56646ef4742"
        ),
        "evidence_fingerprint": (
            "sha256:52e104f3719fd4703be3e83df6b79499efa2530c357a08a67fdad5e9efb4e11e"
        ),
    }

    validate_serialized_ordering_qualification_receipt(legacy)


def test_offline_validator_preserves_historical_v2_receipt_verification() -> None:
    payload = qualify_nnsight_ordering(_TinyGemma3PLTHost(), _request()).to_dict()
    payload["schema_version"] = 2
    payload["qualification_claim"] = (
        "intervened_forward_capture_ordering_with_graph_pinned_mutations_only"
    )
    payload["request_evidence"]["qualification_policy"] = {
        "propagated_mutation": "graph_pinned_preactivation_v1"
    }
    payload["result"].pop("native_feature_diagnostics")

    def fingerprint(value) -> str:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    payload["request_fingerprint"] = fingerprint(payload["request_evidence"])
    science_request = {
        key: payload["request_evidence"][key]
        for key in (
            "scope",
            "tolerance",
            "identity",
            "target",
            "variants",
            "observed_downstream_nodes",
            "qualification_policy",
        )
    }
    qualification_evidence = {
        "schema": payload["schema"],
        "schema_version": 2,
        "qualification_claim": payload["qualification_claim"],
        "science_request_fingerprint": fingerprint(science_request),
        "scope": payload["scope"],
        "result": payload["result"],
    }
    payload["qualification_fingerprint"] = fingerprint(qualification_evidence)
    payload["evidence_fingerprint"] = fingerprint(
        {
            **qualification_evidence,
            "request_fingerprint": payload["request_fingerprint"],
            "qualification_fingerprint": payload["qualification_fingerprint"],
            "provenance": payload["provenance"],
        }
    )

    validate_serialized_ordering_qualification_receipt(payload)


def test_eager_oracle_encodes_only_requested_layers_and_positions() -> None:
    host = _TinyGemma3PLTHost(n_layers=3)

    result = Gemma3PLTEagerHookOracle(host).evaluate(_request().execution)

    assert result.status is RuntimeExecutionStatus.COMPLETE
    assert {layer for layer, _shape in host.transcoders.encoded_shapes} == {0, 1}
    assert all(shape[1] == 1 for _layer, shape in host.transcoders.encoded_shapes)


def test_refusal_receipt_persists_bounded_exception_diagnostic() -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)
    refused = replace(
        production,
        status=RuntimeExecutionStatus.REFUSED,
        baseline=None,
        observations=(),
        refusal=RuntimeRefusal("runtime_failure", "RuntimeError: " + "x" * 10_000),
    )

    receipt = qualify_nnsight_ordering(
        host,
        request,
        selective_runtime=_StaticRuntime(refused),
        oracle_runtime=_StaticRuntime(production),
    )

    assert receipt.result.verdict is OrderingQualificationVerdict.REFUSED
    diagnostic = next(
        item
        for item in receipt.to_dict()["result"]["refusal_diagnostics"]
        if item["runtime"] == "selective"
    )
    assert diagnostic["exception_type"] == "RuntimeError"
    assert diagnostic["message"].endswith("...[truncated]")
    assert len(diagnostic["message"]) <= 2_048


def test_gate_rejects_capture_mutated_to_pre_intervention_values() -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)
    assert production.baseline is not None
    observations = list(production.observations)
    index = next(i for i, item in enumerate(observations) if item.variant_id == "propagated")
    observations[index] = replace(
        observations[index],
        target_value=production.baseline.target_value,
        raw_target_logits=production.baseline.raw_target_logits,
        downstream_feature_values=tuple(
            value
            for value in production.baseline.feature_values
            if value.node in request.execution.observed_downstream_nodes
        ),
    )

    receipt = qualify_nnsight_ordering(
        host,
        request,
        selective_runtime=_StaticRuntime(
            replace(production, observations=tuple(observations))
        ),
    )

    assert receipt.result.verdict is OrderingQualificationVerdict.REJECTED
    assert any("variant.propagated" in reason for reason in receipt.result.reasons)


def test_science_fingerprint_excludes_project_provenance() -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)
    runtime = _StaticRuntime(production)

    first = qualify_nnsight_ordering(
        host, request, selective_runtime=runtime, oracle_runtime=runtime
    )
    second = qualify_nnsight_ordering(
        host,
        replace(request, provenance={"repeat_index": 2, "created_at": "later"}),
        selective_runtime=runtime,
        oracle_runtime=runtime,
    )
    rebound = qualify_nnsight_ordering(
        host,
        replace(
            request,
            execution=replace(
                request.execution,
                deadline_seconds=60.0,
                predicted_variant_seconds=2.0,
            ),
            scope_bindings={"ordering_mechanism": "same-science-different-binding"},
        ),
        selective_runtime=runtime,
        oracle_runtime=runtime,
    )

    assert first.qualification_fingerprint == second.qualification_fingerprint
    assert first.qualification_fingerprint == rebound.qualification_fingerprint
    assert first.request_fingerprint != rebound.request_fingerprint
    assert first.evidence_fingerprint != second.evidence_fingerprint
    assert first.evidence_fingerprint != rebound.evidence_fingerprint


def test_gate_rejects_vacuous_zero_effect_runtime() -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)
    assert production.baseline is not None
    baseline_by_node = {
        value.node: value for value in production.baseline.feature_values
    }
    variant_by_id = {
        variant.variant_id: variant for variant in request.execution.variants
    }
    zero_effect = replace(
        production,
        observations=tuple(
            replace(
                observation,
                target_value=production.baseline.target_value,
                raw_target_logits=production.baseline.raw_target_logits,
                downstream_feature_values=tuple(
                    baseline_by_node[node]
                    for node in request.execution.observed_downstream_nodes
                ),
                intervention_feature_values=tuple(
                    baseline_by_node[item.node]
                    for item in variant_by_id[observation.variant_id].interventions
                ),
            )
            for observation in production.observations
        ),
    )
    runtime = _StaticRuntime(zero_effect)

    receipt = qualify_nnsight_ordering(
        host, request, selective_runtime=runtime, oracle_runtime=runtime
    )

    assert receipt.result.verdict is OrderingQualificationVerdict.REJECTED
    assert set(receipt.result.reasons) == {
        "non_vacuous_effect_missing:direct_frozen",
        "non_vacuous_effect_missing:propagated_frozen_attention",
    }


def test_gate_rejects_baseline_downstream_capture_despite_objective_movement() -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)
    assert production.baseline is not None
    baseline_by_node = {
        value.node: value for value in production.baseline.feature_values
    }
    stale_downstream = tuple(
        baseline_by_node[node]
        for node in request.execution.observed_downstream_nodes
    )
    stale = replace(
        production,
        observations=tuple(
            replace(observation, downstream_feature_values=stale_downstream)
            if observation.variant_id != "no_op"
            else observation
            for observation in production.observations
        ),
    )
    assert any(
        observation.raw_target_logits != production.baseline.raw_target_logits
        for observation in stale.observations
        if observation.variant_id != "no_op"
    )
    runtime = _StaticRuntime(stale)

    receipt = qualify_nnsight_ordering(
        host, request, selective_runtime=runtime, oracle_runtime=runtime
    )

    assert receipt.result.verdict is OrderingQualificationVerdict.REJECTED
    assert set(receipt.result.reasons) == {
        "non_vacuous_effect_missing:direct_frozen",
        "non_vacuous_effect_missing:propagated_frozen_attention",
    }


def test_gate_rejects_no_op_that_does_not_recover_baseline() -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)
    observations = list(production.observations)
    no_op = observations[0]
    observations[0] = replace(
        no_op,
        target_value=no_op.target_value + 1.0,
        raw_target_logits=(no_op.raw_target_logits[0] + 1.0, no_op.raw_target_logits[1]),
    )
    mutated = replace(production, observations=tuple(observations))
    runtime = _StaticRuntime(mutated)

    receipt = qualify_nnsight_ordering(
        host, request, selective_runtime=runtime, oracle_runtime=runtime
    )

    assert receipt.result.verdict is OrderingQualificationVerdict.REJECTED
    assert any("no_op_baseline" in reason for reason in receipt.result.reasons)


def test_gate_rejects_near_zero_stale_downstream_capture() -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)
    assert production.baseline is not None
    downstream = request.execution.observed_downstream_nodes[0]

    def feature_values(value: float):
        return tuple(
            replace(item, preactivation=value) if item.node == downstream else item
            for item in production.baseline.feature_values
        )

    baseline = replace(production.baseline, feature_values=feature_values(0.0))
    oracle = replace(
        production,
        baseline=baseline,
        observations=tuple(
            replace(
                observation,
                downstream_feature_values=(FeatureValue(downstream, 0.01),),
            )
            if observation.variant_id != "no_op"
            else replace(
                observation,
                downstream_feature_values=(FeatureValue(downstream, 0.0),),
            )
            for observation in production.observations
        ),
    )
    selective = replace(
        oracle,
        observations=tuple(
            replace(
                observation,
                downstream_feature_values=(FeatureValue(downstream, 0.0),),
            )
            if observation.variant_id != "no_op"
            else observation
            for observation in oracle.observations
        ),
    )

    receipt = qualify_nnsight_ordering(
        host,
        request,
        selective_runtime=_StaticRuntime(selective),
        oracle_runtime=_StaticRuntime(oracle),
    )

    assert receipt.result.verdict is OrderingQualificationVerdict.REJECTED
    assert any(
        reason.startswith("comparison_failed:direct.canonical_feature")
        for reason in receipt.result.reasons
    )


def test_gate_rejects_near_zero_objective_difference_beyond_two_bf16_ulps() -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)
    assert production.baseline is not None

    def with_objective(value: float):
        return replace(
            production,
            baseline=replace(
                production.baseline,
                target_value=value,
                raw_target_logits=(value, 0.0),
            ),
            observations=tuple(
                replace(
                    observation,
                    target_value=value,
                    raw_target_logits=(value, 0.0),
                )
                for observation in production.observations
            ),
        )

    receipt = qualify_nnsight_ordering(
        host,
        request,
        selective_runtime=_StaticRuntime(with_objective(0.001)),
        oracle_runtime=_StaticRuntime(with_objective(0.0)),
    )

    assert receipt.result.verdict is OrderingQualificationVerdict.REJECTED
    assert any(
        reason.startswith("comparison_failed:baseline.target_logit")
        for reason in receipt.result.reasons
    )


def test_native_feature_difference_is_diagnostic_only() -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)
    assert production.baseline is not None
    source = request.execution.variants[1].interventions[0].node

    def with_source_feature(value: float):
        def rewrite(values):
            return tuple(
                replace(item, preactivation=value) if item.node == source else item
                for item in values
            )

        return replace(
            production,
            baseline=replace(
                production.baseline,
                feature_values=rewrite(production.baseline.feature_values),
            ),
            observations=tuple(
                replace(
                    observation,
                    intervention_feature_values=rewrite(
                        observation.intervention_feature_values
                    ),
                )
                for observation in production.observations
            ),
        )

    receipt = qualify_nnsight_ordering(
        host,
        request,
        selective_runtime=_StaticRuntime(
            with_source_feature(0.001), capture_result=production
        ),
        oracle_runtime=_StaticRuntime(
            with_source_feature(0.0), capture_result=production
        ),
    )

    assert receipt.result.verdict is OrderingQualificationVerdict.QUALIFIED
    assert any(
        item.path.startswith("baseline.feature") and not item.passed
        for item in receipt.result.native_feature_diagnostics
    )


def test_gate_refuses_incomplete_cleanup() -> None:
    host = _TinyGemma3PLTHost()
    request = _request()
    production = NNSightInterventionRuntime(
        host, ordering_admission_mode=OrderingAdmissionMode.CANDIDATE_SMOKE
    ).evaluate(request.execution)

    receipt = qualify_nnsight_ordering(
        host,
        request,
        selective_runtime=_StaticRuntime(
            replace(production, cleanup_completed=False)
        ),
        oracle_runtime=_StaticRuntime(production),
    )

    assert receipt.result.verdict is OrderingQualificationVerdict.REFUSED
    assert "selective_cleanup_incomplete" in receipt.result.reasons


def test_public_constructor_accepts_exact_project_scope_mapping() -> None:
    execution = _request().execution
    project_scope = {
        "model_name": "google/gemma-3-12b-it",
        "provider_family": "gemmascope2-plt-12b-small-affine",
        "feature_input_hook": "mlp.hook_in",
        "feature_output_hook": "hook_mlp_out",
        "model_dtype": "bfloat16",
        "skip_connection": False,
        "ordering_mechanism": "forward_ordered_same_invoke_capture_v1",
    }

    request = OrderingQualificationRequest.from_execution_requests(
        execution_requests=(execution,),
        scope=project_scope,
        provenance={"repeat_index": 2},
    )

    assert request.scope.model_family == "gemma3"
    assert request.scope.transcoder_architecture == "plt"
    assert request.scope_bindings == project_scope


def test_public_constructor_reduces_production_variants_to_one_matched_case() -> None:
    base = _request().execution
    no_op, direct, propagated = base.variants
    invalid_source = FeatureNode(0, 1, 2)
    later_source = FeatureNode(1, 1, 2)
    invalid_downstream = FeatureNode(0, 2, 3)
    later_downstream = FeatureNode(3, 1, 3)
    variants = (
        no_op,
        InterventionVariant(
            "invalid_same_layer_direct",
            VariantKind.DIRECT_DOUBLE,
            InterventionSemantics.DIRECT_FROZEN,
            (PreactivationIntervention(invalid_source, 2.0, 1.0, 1.0),),
            None,
            (FeatureValue(invalid_downstream, 0.1),),
        ),
        InterventionVariant(
            "later_direct",
            VariantKind.DIRECT_DOUBLE,
            InterventionSemantics.DIRECT_FROZEN,
            (PreactivationIntervention(later_source, 2.0, 1.0, 1.0),),
            None,
            (FeatureValue(later_downstream, 0.2),),
        ),
        InterventionVariant(
            "later_propagated",
            VariantKind.NECESSITY_HIGH,
            InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
            (PreactivationIntervention(later_source, 0.0),),
            None,
            (FeatureValue(later_downstream, 0.2),),
        ),
        direct,
        propagated,
        InterventionVariant(
            "unmatched_propagated",
            VariantKind.NECESSITY_HIGH,
            InterventionSemantics.PROPAGATED_FROZEN_ATTENTION,
            (PreactivationIntervention(invalid_source, 0.0),),
            None,
            (FeatureValue(later_downstream, 0.2),),
        ),
    )
    observed = tuple(
        sorted(
            {
                value.node
                for variant in variants
                for value in variant.predicted_downstream_feature_deltas
            }
        )
    )
    production = replace(base, variants=variants, observed_downstream_nodes=observed)

    request = OrderingQualificationRequest.from_execution_requests(
        execution_requests=(production,),
        scope={"model_family": "gemma3", "provider_architecture": "plt"},
        provenance={"full_input_artifact": "unchanged.json"},
    )

    assert tuple(variant.variant_id for variant in request.execution.variants) == (
        "no_op",
        "direct",
        "propagated",
    )
    assert request.execution.observed_downstream_nodes == base.observed_downstream_nodes
    assert request.execution.identity == production.identity
    assert request.execution.target == production.target
    assert request.execution.deadline_seconds == production.deadline_seconds
    assert request.execution.predicted_variant_seconds == production.predicted_variant_seconds
    assert len(production.variants) == 7
