from types import SimpleNamespace

import pytest

from tests.test_attributions_gemma import patch_tokenizer_special_ids


class _Tokenizer:
    @property
    def all_special_ids(self) -> list[int]:
        return [1, 2]


def test_special_id_patch_restores_class_property_after_failure() -> None:
    tokenizer = _Tokenizer()
    model = SimpleNamespace(tokenizer=tokenizer)
    original_property = type(tokenizer).all_special_ids

    with pytest.raises(RuntimeError, match="stop inside patched scope"):
        with patch_tokenizer_special_ids(model, [0]):  # type: ignore[arg-type]
            assert tokenizer.all_special_ids == [0]
            raise RuntimeError("stop inside patched scope")

    assert type(tokenizer).all_special_ids is original_property
    assert tokenizer.all_special_ids == [1, 2]
