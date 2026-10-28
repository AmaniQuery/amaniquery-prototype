"""
Minimal OpenTelemetry bootstrap shared by AmaniQuery entrypoints.

Design goals:
- Additive: nothing here is required for the app to run.
- Fail-open: every step is wrapped so an unreachable collector or a
  missing package can never break the host service.
- Opt-in: telemetry is only configured when OTEL_EXPORTER_OTLP_ENDPOINT
  is set (the compose file sets it to http://otel-collector:4318).

Usage (from an entrypoint / app module)::

    from observability.otel_bootstrap import setup_telemetry, instrument_fastapi

    setup_telemetry()          # idempotent; safe to call from anywhere
    instrument_fastapi(app)    # after the FastAPI app object exists

Note: Module4_NiruAPI/agents/prefetch.py and tool_executor_v0.py also
configure OTel (gRPC exporters) when OTEL_EXPORTER_OTLP_ENDPOINT is set.
If this module sets the global providers first, their set_tracer_provider
call becomes a no-op (OpenTelemetry forbids overriding), and their spans
flow through this module's HTTP exporter instead - which is the desired
behavior since the collector's gRPC receiver is on :4317 but services are
wired to the HTTP endpoint :4318.
"""

import os

from loguru import logger

_INITIALIZED = False


def _endpoint_configured() -> bool:
    return bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())


def setup_telemetry() -> bool:
    """
    Configure the global TracerProvider and MeterProvider with OTLP/HTTP
    exporters. Idempotent and fail-open. Returns True if telemetry was
    configured, False if skipped or failed.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return True

    if not _endpoint_configured():
        logger.debug(
            "[OTEL] OTEL_EXPORTER_OTLP_ENDPOINT not set - telemetry disabled"
        )
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        service_name = (
            os.getenv("OTEL_SERVICE_NAME") or "amaniquery-unknown"
        )
        resource = Resource.create({"service.name": service_name})

        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter())  # reads OTEL_EXPORTER_OTLP_ENDPOINT
        )
        # First call wins; a later call from legacy code is a warned no-op.
        trace.set_tracer_provider(provider)

        # Optional: metrics (only if the metrics SDK/exporter are available)
        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )

            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(), export_interval_millis=60000
            )
            metrics.set_meter_provider(
                MeterProvider(resource=resource, metric_readers=[reader])
            )
        except Exception as e:
            logger.debug(f"[OTEL] Metrics provider not configured: {e}")

        _INITIALIZED = True
        logger.info(
            f"[OTEL] Telemetry configured for service '{service_name}' -> "
            f"{os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}"
        )
        return True

    except ImportError as e:
        logger.warning(
            f"[OTEL] opentelemetry packages not installed, telemetry disabled: {e}"
        )
        return False
    except Exception as e:
        logger.warning(f"[OTEL] Telemetry setup failed (continuing): {e}")
        return False


def instrument_fastapi(app) -> bool:
    """
    Attach OpenTelemetry instrumentation to a FastAPI app.
    Call AFTER the app object is created. Idempotent per-app and fail-open.
    """
    if not _endpoint_configured():
        return False
    try:
        from opentelemetry.instrumentation.fastapi import (  # type: ignore
            FastAPIInstrumentor,
        )

        FastAPIInstrumentor.instrument_app(app)
        logger.info("[OTEL] FastAPI instrumentation attached")
        return True
    except ImportError as e:
        logger.warning(f"[OTEL] FastAPI instrumentation unavailable: {e}")
        return False
    except Exception as e:
        logger.warning(f"[OTEL] FastAPI instrumentation failed (continuing): {e}")
        return False


def instrument_celery() -> bool:
    """
    Instrument Celery (task spans on workers). Call before tasks run;
    safe to call from the celery_app module. Fail-open.
    """
    if not _endpoint_configured():
        return False
    try:
        from opentelemetry.instrumentation.celery import (  # type: ignore
            CeleryInstrumentor,
        )

        CeleryInstrumentor().instrument()
        logger.info("[OTEL] Celery instrumentation attached")
        return True
    except ImportError as e:
        logger.warning(f"[OTEL] Celery instrumentation unavailable: {e}")
        return False
    except Exception as e:
        logger.warning(f"[OTEL] Celery instrumentation failed (continuing): {e}")
        return False
