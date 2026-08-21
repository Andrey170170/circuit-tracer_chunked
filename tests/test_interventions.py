import gc

import pytest
import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer import ReplacementModel
from tests.test_attributions_gemma import (
    load_dummy_gemma_model,
    patch_tokenizer_special_ids,
)


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


def test_intervention_generate_returns_step_by_vocab_logits_on_cpu() -> None:
    cfg = HookedTransformerConfig(
        d_model=8,
        d_head=4,
        n_layers=1,
        n_ctx=16,
        n_heads=2,
        d_mlp=16,
        d_vocab=16,
        d_vocab_out=16,
        device="cpu",
        tokenizer_name="gpt2",
        init_weights=False,
    )
    model = load_dummy_gemma_model(cfg)
    generated_steps = 2

    with patch_tokenizer_special_ids(model, [0]):
        _, logits, _ = model.feature_intervention_generate(
            torch.tensor([[1, 2, 3]]),
            [],
            freeze_attention=False,
            return_activations=False,
            max_new_tokens=generated_steps,
            do_sample=False,
            stop_at_eos=False,
            verbose=False,
        )

    assert logits.shape == (generated_steps, cfg.d_vocab_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_return_activations_tl():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

    s = "The National Digital Analytics Group (ND"

    interventions = [(21, 7, 5066, 0.0)]

    logits_with_activations, activations = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.cfg.n_layers),  # type:ignore
        return_activations=True,
    )

    logits_without_activations, no_activations = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.cfg.n_layers),  # type:ignore
        return_activations=False,
    )

    assert torch.allclose(
        logits_with_activations, logits_without_activations, atol=1e-6, rtol=1e-5
    ), "Logits should be identical regardless of return_activations setting"

    assert activations is not None, "Activations should be returned when return_activations=True"
    assert no_activations is None, "Activations should be None when return_activations=False"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_return_activations_nnsight():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")

    s = "The National Digital Analytics Group (ND"

    interventions = [(21, 7, 5066, 0.0)]

    logits_with_activations, activations = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
        return_activations=True,
    )

    logits_without_activations, no_activations = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
        return_activations=False,
    )

    assert torch.allclose(
        logits_with_activations, logits_without_activations, atol=1e-6, rtol=1e-5
    ), "Logits should be identical regardless of return_activations setting"

    assert activations is not None, "Activations should be returned when return_activations=True"
    assert no_activations is None, "Activations should be None when return_activations=False"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_generate_return_activations_tl():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

    s = "Fait: Michael Jordan joue au"

    interventions = [(20, slice(6, None), 1454, 0), (20, slice(6, None), 341, 272)]

    with model.zero_softcap():
        str_with_acts, logits_with_activations, activations = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.cfg.n_layers),  # type:ignore
            return_activations=True,
            do_sample=False,
        )

        str_without_acts, logits_without_activations, no_activations = (
            model.feature_intervention_generate(
                s,
                interventions,
                constrained_layers=range(model.cfg.n_layers),  # type:ignore
                return_activations=False,
                do_sample=False,
            )
        )

    assert str_with_acts == str_without_acts, (
        f"Generated strings should be identical, but got {str_with_acts}, {str_without_acts}"
    )

    assert torch.allclose(
        logits_with_activations, logits_without_activations, atol=1e-6, rtol=1e-5
    ), "Logits should be identical regardless of return_activations setting"

    assert activations is not None, "Activations should be returned when return_activations=True"
    assert no_activations is None, "Activations should be None when return_activations=False"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_generate_return_activations_nnsight():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")

    s = "Fait: Michael Jordan joue au"

    interventions = [(20, slice(6, None), 1454, 0), (20, slice(6, None), 341, 272)]

    with model.zero_softcap():
        str_with_acts, logits_with_activations, activations = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
            return_activations=True,
            do_sample=False,
        )

        str_without_acts, logits_without_activations, no_activations = (
            model.feature_intervention_generate(
                s,
                interventions,
                constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
                return_activations=False,
                do_sample=False,
            )
        )

    assert str_with_acts == str_without_acts, (
        f"Generated strings should be identical, but got {str_with_acts}, {str_without_acts}"
    )

    assert torch.allclose(
        logits_with_activations, logits_without_activations, atol=1e-6, rtol=1e-5
    ), "Logits should be identical regardless of return_activations setting"

    assert activations is not None, "Activations should be returned when return_activations=True"
    assert no_activations is None, "Activations should be None when return_activations=False"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_sparse_tl():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

    s = "The National Digital Analytics Group (ND"

    interventions = [(21, 7, 5066, 0.0)]

    logits_dense, activations_dense = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.cfg.n_layers),  # type:ignore
        sparse=False,
        return_activations=True,
    )

    logits_sparse, activations_sparse = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.cfg.n_layers),  # type:ignore
        sparse=True,
        return_activations=True,
    )

    assert torch.allclose(logits_dense, logits_sparse, atol=1e-6, rtol=1e-5), (
        "Logits should be identical regardless of sparse setting"
    )

    assert torch.allclose(
        activations_dense,  # type:ignore
        activations_sparse.to_dense(),  # type:ignore
        atol=1e-6,
        rtol=1e-5,
    ), "Activations should be identical regardless of sparse setting"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_sparse_nnsight():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")

    s = "The National Digital Analytics Group (ND"

    interventions = [(21, 7, 5066, 0.0)]

    logits_dense, activations_dense = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
        sparse=False,
        return_activations=True,
    )

    logits_sparse, activations_sparse = model.feature_intervention(
        s,
        interventions,
        constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
        sparse=True,
        return_activations=True,
    )

    assert torch.allclose(logits_dense, logits_sparse, atol=1e-6, rtol=1e-5), (
        "Logits should be identical regardless of sparse setting"
    )

    assert torch.allclose(
        activations_dense,  # type:ignore
        activations_sparse.to_dense(),  # type:ignore
        atol=1e-6,
        rtol=1e-5,
    ), "Activations should be identical regardless of sparse setting"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_generate_sparse_tl():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

    s = "Fait: Michael Jordan joue au"

    interventions = [(20, slice(6, None), 1454, 0), (20, slice(6, None), 341, 272)]

    with model.zero_softcap():
        str_dense, logits_dense, activations_dense = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.cfg.n_layers),  # type:ignore
            sparse=False,
            do_sample=False,
            return_activations=True,
        )

        str_sparse, logits_sparse, activations_sparse = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.cfg.n_layers),  # type:ignore
            sparse=True,
            do_sample=False,
            return_activations=True,
        )

    assert str_dense == str_sparse, (
        f"Generated strings should be identical, but got {str_dense}, {str_sparse}"
    )

    assert torch.allclose(logits_dense, logits_sparse, atol=1e-6, rtol=1e-5), (
        "Logits should be identical regardless of sparse setting"
    )

    assert torch.allclose(
        activations_dense,  # type:ignore
        activations_sparse.to_dense(),  # type:ignore
        atol=1e-6,
        rtol=1e-5,
    ), "Activations should be identical regardless of sparse setting"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_intervention_generate_sparse_nnsight():
    model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="nnsight")

    s = "Fait: Michael Jordan joue au"

    interventions = [(20, slice(6, None), 1454, 0), (20, slice(6, None), 341, 272)]

    with model.zero_softcap():
        str_dense, logits_dense, activations_dense = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
            sparse=False,
            do_sample=False,
            return_activations=True,
        )

        str_sparse, logits_sparse, activations_sparse = model.feature_intervention_generate(
            s,
            interventions,
            constrained_layers=range(model.config.num_hidden_layers),  # type:ignore
            sparse=True,
            do_sample=False,
            return_activations=True,
        )

    assert str_dense == str_sparse, (
        f"Generated strings should be identical, but got {str_dense}, {str_sparse}"
    )

    assert torch.allclose(logits_dense, logits_sparse, atol=1e-6, rtol=1e-5), (
        "Logits should be identical regardless of sparse setting"
    )

    assert torch.allclose(
        activations_dense,  # type:ignore
        activations_sparse.to_dense(),  # type:ignore
        atol=1e-6,
        rtol=1e-5,
    ), "Activations should be identical regardless of sparse setting"


if __name__ == "__main__":
    torch.manual_seed(42)
    test_intervention_return_activations_tl()
    test_intervention_return_activations_nnsight()
    test_intervention_generate_return_activations_tl()
    test_intervention_generate_return_activations_nnsight()

    test_intervention_sparse_tl()
    test_intervention_sparse_nnsight()
    test_intervention_generate_sparse_tl()
    test_intervention_generate_sparse_nnsight()
