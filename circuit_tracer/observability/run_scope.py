"""Backend-neutral lifecycle and terminal evidence ownership."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from circuit_tracer.observability.events import TraceEvent
from circuit_tracer.execution_identity import ExecutionIdentityState


@dataclass(frozen=True)
class ObservabilityEvidence:
    """Terminal observability evidence returned beside the scientific output."""

    summary: dict[str, object]
    events: tuple[dict[str, object], ...] = ()


@dataclass
class TraceRunScope:
    """Own one canonical trace lifecycle for every attribution backend."""

    observer: Any
    logger: Any
    compact_output: bool
    profile: bool
    execution_identity: ExecutionIdentityState
    governor_admission_mode: str = "enforce"
    started_at: float = field(default_factory=time.perf_counter)
    _closed: bool = False

    def close(
        self,
        primary_error: BaseException | None,
        *,
        terminal_status: str | None = None,
    ) -> ObservabilityEvidence:
        if self._closed:
            raise RuntimeError("trace run scope is already closed")
        self._closed = True
        elapsed_ms = (time.perf_counter() - self.started_at) * 1000.0
        if terminal_status is None:
            terminal_status = "succeeded" if primary_error is None else "failed"
        if terminal_status not in {"succeeded", "probe_completed", "failed", "refused"}:
            raise ValueError(f"unsupported terminal status: {terminal_status!r}")
        if primary_error is not None and terminal_status != "failed":
            raise ValueError("a primary error requires terminal_status='failed'")
        name = {
            "succeeded": "attribute.done",
            "probe_completed": "attribute.probe_completed",
            "failed": "attribute.failed",
            "refused": "attribute.refused",
        }[terminal_status]
        attrs: dict[str, object] = {
            "compact_output": self.compact_output,
            "requested_execution_fingerprint": self.execution_identity.requested_fingerprint,
            "effective_execution_fingerprint": self.execution_identity.effective_fingerprint,
            "execution_fingerprint": self.execution_identity.execution_fingerprint,
            "governor_admission_mode": self.governor_admission_mode,
            "status": terminal_status,
        }
        if primary_error is not None:
            attrs.update(
                error_type=type(primary_error).__name__,
                error_message=str(primary_error),
            )

        terminal_failure: BaseException | None = None
        try:
            self.observer.observe(
                TraceEvent(
                    scope="run",
                    name=name,
                    elapsed_ms=elapsed_ms,
                    attrs=attrs,
                    wall_clock=True,
                )
            )
        except BaseException as exc:
            terminal_failure = exc

        try:
            telemetry_export = self.observer.close_export()
        except BaseException as exc:
            telemetry_export = {"summary": {}, "events": []}
            if terminal_failure is None:
                terminal_failure = exc
            elif primary_error is not None:
                primary_error.add_note(f"observability close also failed: {exc!r}")

        summary = telemetry_export.get("summary", {})
        terminal_summary = dict(summary) if isinstance(summary, dict) else {}
        terminal_summary.update(
            requested_execution_fingerprint=self.execution_identity.requested_fingerprint,
            effective_execution_fingerprint=self.execution_identity.effective_fingerprint,
            execution_fingerprint=self.execution_identity.execution_fingerprint,
            governor_admission_mode=self.governor_admission_mode,
        )
        telemetry_export["summary"] = terminal_summary

        if primary_error is not None:
            if terminal_failure is not None:
                primary_error.add_note(
                    f"terminal observability failed without masking the primary error: "
                    f"{terminal_failure!r}"
                )
            try:
                self.observer.attach_exception(primary_error, telemetry_export)
            except BaseException as exc:
                primary_error.add_note(f"observability exception attachment failed: {exc!r}")
        elif terminal_failure is not None:
            raise terminal_failure

        if self.profile:
            try:
                self.observer.render_human_summary(self.logger, telemetry_export)
            except BaseException as exc:
                if primary_error is not None:
                    primary_error.add_note(f"human telemetry rendering failed: {exc!r}")
                else:
                    raise

        events = telemetry_export.get("events", ())
        return ObservabilityEvidence(
            summary=terminal_summary,
            events=tuple(events) if isinstance(events, list) else (),
        )
