from circuit_tracer.attribution.nnsight.resource_sampling import (
    should_sample_batch_resources,
    should_sample_phase4_resources,
)


def _sample(index: int, *, retain_graph: bool = True) -> bool:
    return should_sample_batch_resources(
        phase_label="phase4_features",
        phase_batch_index=index,
        retain_graph=retain_graph,
    )


def test_phase4_resource_sampling_covers_transition_periodic_and_final_batches() -> None:
    assert [_sample(index) for index in range(1, 5)] == [True, True, True, False]
    assert _sample(31) is False
    assert _sample(32) is True
    assert _sample(64) is True
    assert _sample(65, retain_graph=False) is True


def test_non_phase4_batches_retain_full_resource_sampling() -> None:
    assert should_sample_batch_resources(
        phase_label="phase3_logits",
        phase_batch_index=17,
        retain_graph=True,
    )


def test_phase4_refresh_sampling_covers_transition_and_periodic_evidence() -> None:
    assert [
        should_sample_phase4_resources(sample_index=index)
        for index in range(1, 5)
    ] == [True, True, True, False]
    assert should_sample_phase4_resources(sample_index=32)
    assert should_sample_phase4_resources(sample_index=33) is False
    assert should_sample_phase4_resources(sample_index=33, final=True)
