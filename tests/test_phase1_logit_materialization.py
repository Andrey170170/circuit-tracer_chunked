from types import SimpleNamespace

from circuit_tracer.attribution.context_nnsight import AttributionContext


def _context(logit_retention: str) -> AttributionContext:
    ctx = AttributionContext.__new__(AttributionContext)
    ctx.logit_retention = logit_retention
    ctx.phase1_logit_materialization_metadata = {}
    return ctx


def _model_with_forward(forward):
    return SimpleNamespace(_model=SimpleNamespace(forward=forward))


def test_phase1_uses_final_token_logits_when_model_supports_current_kwarg() -> None:
    def forward(input_ids=None, logits_to_keep=0):
        return input_ids, logits_to_keep

    ctx = _context("last_token")

    assert ctx.resolve_phase1_invoke_kwargs(_model_with_forward(forward)) == {
        "logits_to_keep": 1
    }
    assert ctx.phase1_logit_materialization_metadata == {
        "requested": "last_token",
        "effective": "last_token",
        "model_forward_kwarg": "logits_to_keep",
        "fallback_reason": None,
    }


def test_phase1_supports_legacy_num_logits_to_keep_kwarg() -> None:
    def forward(input_ids=None, num_logits_to_keep=0):
        return input_ids, num_logits_to_keep

    ctx = _context("last_token")

    assert ctx.resolve_phase1_invoke_kwargs(_model_with_forward(forward)) == {
        "num_logits_to_keep": 1
    }


def test_phase1_preserves_full_logits_when_required() -> None:
    def forward(input_ids=None, logits_to_keep=0):
        return input_ids, logits_to_keep

    ctx = _context("full")

    assert ctx.resolve_phase1_invoke_kwargs(_model_with_forward(forward)) == {}
    assert ctx.phase1_logit_materialization_metadata["effective"] == "full"
    assert (
        ctx.phase1_logit_materialization_metadata["fallback_reason"]
        == "full_logits_required"
    )


def test_phase1_falls_back_when_model_cannot_slice_logits() -> None:
    def forward(input_ids=None):
        return input_ids

    ctx = _context("last_token")

    assert ctx.resolve_phase1_invoke_kwargs(_model_with_forward(forward)) == {}
    assert ctx.phase1_logit_materialization_metadata["effective"] == "full"
    assert (
        ctx.phase1_logit_materialization_metadata["fallback_reason"]
        == "model_forward_logit_slice_unsupported"
    )
