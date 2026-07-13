from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "circuit_tracer"


def test_production_has_no_legacy_tracing_entrypoints_or_references() -> None:
    assert not (PACKAGE_ROOT / "runtime.py").exists()
    assert not (PACKAGE_ROOT / "attribution" / "attribute.py").exists()
    assert not (PACKAGE_ROOT / "attribution" / "attribute_nnsight.py").exists()

    forbidden = (
        "attribute_nnsight",
        "circuit_tracer.runtime",
        "circuit_tracer.attribution.attribute import",
    )
    stale = {
        path.relative_to(PACKAGE_ROOT): token
        for path in PACKAGE_ROOT.rglob("*.py")
        for token in forbidden
        if token in path.read_text()
    }
    assert stale == {}
