"""Best-effort telemetry export attachment for failed attribution runs."""

from collections.abc import Mapping


_TELEMETRY_EXCEPTION_SUMMARY_ATTR = "circuit_tracer_telemetry_summary"
_TELEMETRY_EXCEPTION_EVENTS_ATTR = "circuit_tracer_telemetry_events"


def _attach_telemetry_export_to_exception(
    exc: BaseException | None,
    telemetry_export: Mapping[str, object],
) -> None:
    """Attach telemetry export to an exception for callers that persist failures.

    ``attribute`` normally returns telemetry through the compact result. If an
    exception interrupts the run before a compact result exists, the caller still
    needs the recorded telemetry events to diagnose the failure. Best-effort
    exception attributes avoid changing the public return type and survive common
    wrappers such as NNsight exceptions.
    """

    if exc is None:
        return
    try:
        setattr(
            exc,
            _TELEMETRY_EXCEPTION_SUMMARY_ATTR,
            telemetry_export.get("summary"),
        )
        setattr(
            exc,
            _TELEMETRY_EXCEPTION_EVENTS_ATTR,
            telemetry_export.get("events", []),
        )
    except Exception:  # pragma: no cover - defensive for unusual exception types
        return
