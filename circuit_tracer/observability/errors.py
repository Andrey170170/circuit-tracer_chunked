"""Exception serialization that cannot replace the primary failure."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
from typing import Any


def _guarded_text(error: BaseException, method: str) -> str:
    try:
        value = str(error) if method == "str" else repr(error)
    except BaseException as formatting_error:  # pragma: no cover - adversarial objects
        return (
            f"<unavailable: {method} raised "
            f"{type(formatting_error).__module__}.{type(formatting_error).__qualname__}>"
        )
    return value


def _safe_structured_details(error: BaseException) -> dict[str, object] | None:
    """Copy bounded primitive details without trusting exception attributes."""

    try:
        details = getattr(error, "details", None)
    except BaseException:  # pragma: no cover - adversarial objects
        return None
    if not isinstance(details, Mapping):
        return None

    safe: dict[str, object] = {}
    try:
        items = details.items()
        for raw_key, value in islice(items, 64):
            if not isinstance(raw_key, str) or len(raw_key) > 128:
                continue
            if value is None or isinstance(value, bool | int | float):
                safe[raw_key] = value
            elif isinstance(value, str):
                safe[raw_key] = value[:4096]
    except BaseException:  # pragma: no cover - adversarial mappings
        return safe or None
    return safe or None


def safe_exception_attrs(error: BaseException) -> dict[str, Any]:
    """Return bounded diagnostic fields without trusting exception formatting."""

    error_type_qualified = f"{type(error).__module__}.{type(error).__qualname__}"
    attrs: dict[str, Any] = {
        "error_type": type(error).__name__,
        "error_type_qualified": error_type_qualified,
        "error_message": _guarded_text(error, "str"),
        "error_repr": _guarded_text(error, "repr"),
    }
    details = _safe_structured_details(error)
    if details is not None:
        attrs["error_details"] = details
    return attrs


def safe_exception_message(error: BaseException) -> str:
    """Return the guarded human-readable form used by non-structured logs."""

    return str(safe_exception_attrs(error)["error_message"])
