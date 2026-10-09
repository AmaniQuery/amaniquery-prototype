"""
Metrics collection for voice agent (Prometheus-compatible)
"""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
from threading import Lock

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not available, metrics will be collected but not exported"
    )


@dataclass
class VoiceMetrics:
    """Voice agent metrics"""

    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Provider metrics
    stt_requests: int = 0
    stt_successes: int = 0
    stt_failures: int = 0
    tts_requests: int = 0
    tts_successes: int = 0
    tts_failures: int = 0
    rag_requests: int = 0
    rag_successes: int = 0
    rag_failures: int = 0

    # Latency metrics (in seconds)
    stt_latencies: List[float] = field(default_factory=list)
    tts_latencies: List[float] = field(default_factory=list)
    rag_latencies: List[float] = field(default_factory=list)
    total_latencies: List[float] = field(default_factory=list)

    # Session metrics
    active_sessions: int = 0
    total_sessions: int = 0
    expired_sessions: int = 0

    # Error metrics
    error_counts: Dict[str, int] = field(default_factory=dict)

    # Provider-specific metrics
    provider_metrics: Dict[str, Dict] = field(default_factory=dict)

    max_latency_samples: int = 1000  # Keep last N samples

    def add_latency(self, metric_type: str, latency: float):
        """Add latency sample"""
        if metric_type == "stt":
            self.stt_latencies.append(latency)
            if len(self.stt_latencies) > self.max_latency_samples:
                self.stt_latencies.pop(0)
        elif metric_type == "tts":
            self.tts_latencies.append(latency)
            if len(self.tts_latencies) > self.max_latency_samples:
                self.tts_latencies.pop(0)
        elif metric_type == "rag":
            self.rag_latencies.append(latency)
            if len(self.rag_latencies) > self.max_latency_samples:
                self.rag_latencies.pop(0)
        elif metric_type == "total":
            self.total_latencies.append(latency)
            if len(self.total_latencies) > self.max_latency_samples:
                self.total_latencies.pop(0)

    def get_average_latency(self, metric_type: str) -> float:
        """Get average latency for a metric type"""
        latencies = {
            "stt": self.stt_latencies,
            "tts": self.tts_latencies,
            "rag": self.rag_latencies,
            "total": self.total_latencies,
        }.get(metric_type, [])

        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)

    def get_percentile_latency(self, metric_type: str, percentile: float) -> float:
        """Get percentile latency (e.g., p50, p95, p99)"""
        latencies = {
            "stt": self.stt_latencies,
            "tts": self.tts_latencies,
            "rag": self.rag_latencies,
            "total": self.total_latencies,
        }.get(metric_type, [])

        if not latencies:
            return 0.0

        sorted_latencies = sorted(latencies)
        index = int(len(sorted_latencies) * percentile / 100.0)
        index = min(index, len(sorted_latencies) - 1)
        return sorted_latencies[index]

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            "requests": {
                "total": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": (
                    self.successful_requests / self.total_requests
                    if self.total_requests > 0
                    else 0.0
                ),
            },
            "stt": {
                "requests": self.stt_requests,
                "successes": self.stt_successes,
                "failures": self.stt_failures,
                "success_rate": (
                    self.stt_successes / self.stt_requests
                    if self.stt_requests > 0
                    else 0.0
                ),
                "avg_latency": self.get_average_latency("stt"),
                "p50_latency": self.get_percentile_latency("stt", 50),
                "p95_latency": self.get_percentile_latency("stt", 95),
                "p99_latency": self.get_percentile_latency("stt", 99),
            },
            "tts": {
                "requests": self.tts_requests,
                "successes": self.tts_successes,
                "failures": self.tts_failures,
                "success_rate": (
                    self.tts_successes / self.tts_requests
                    if self.tts_requests > 0
                    else 0.0
                ),
                "avg_latency": self.get_average_latency("tts"),
                "p50_latency": self.get_percentile_latency("tts", 50),
                "p95_latency": self.get_percentile_latency("tts", 95),
                "p99_latency": self.get_percentile_latency("tts", 99),
            },
            "rag": {
                "requests": self.rag_requests,
                "successes": self.rag_successes,
                "failures": self.rag_failures,
                "success_rate": (
                    self.rag_successes / self.rag_requests
                    if self.rag_requests > 0
                    else 0.0
                ),
                "avg_latency": self.get_average_latency("rag"),
                "p50_latency": self.get_percentile_latency("rag", 50),
                "p95_latency": self.get_percentile_latency("rag", 95),
                "p99_latency": self.get_percentile_latency("rag", 99),
            },
            "sessions": {
                "active": self.active_sessions,
                "total": self.total_sessions,
                "expired": self.expired_sessions,
            },
            "errors": dict(self.error_counts),
        }


class MetricsCollector:
    """
    Collects and exports metrics for the voice agent
    """

    def __init__(self, enable_prometheus: bool = False, prometheus_port: int = 9090):
        """
        Initialize metrics collector

        Args:
            enable_prometheus: Enable Prometheus metrics export
            prometheus_port: Port for Prometheus HTTP server
        """
        self.metrics = VoiceMetrics()
        self._lock = Lock()
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.prometheus_port = prometheus_port

        # Prometheus metrics
        if self.enable_prometheus:
            self._init_prometheus_metrics()
            try:
                start_http_server(self.prometheus_port)
                logger.info(
                    f"Prometheus metrics server started on port {self.prometheus_port}"
                )
            except Exception as e:
                logger.warning(f"Failed to start Prometheus server: {e}")
                self.enable_prometheus = False
        else:
            if enable_prometheus and not PROMETHEUS_AVAILABLE:
                logger.warning(
                    "Prometheus client not available, install with: pip install prometheus-client"
                )
            self._prometheus_counters = {}
            self._prometheus_histograms = {}
            self._prometheus_gauges = {}

        logger.info("Metrics collector initialized")

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        # Request counters
        self._prometheus_counters = {
            "total_requests": Counter("voice_requests_total", "Total voice requests"),
            "successful_requests": Counter(
                "voice_requests_successful", "Successful voice requests"
            ),
            "failed_requests": Counter(
                "voice_requests_failed", "Failed voice requests"
            ),
            "stt_requests": Counter("voice_stt_requests_total", "Total STT requests"),
            "stt_successes": Counter(
                "voice_stt_successes_total", "Successful STT requests"
            ),
            "stt_failures": Counter("voice_stt_failures_total", "Failed STT requests"),
            "tts_requests": Counter("voice_tts_requests_total", "Total TTS requests"),
            "tts_successes": Counter(
                "voice_tts_successes_total", "Successful TTS requests"
            ),
            "tts_failures": Counter("voice_tts_failures_total", "Failed TTS requests"),
            "rag_requests": Counter("voice_rag_requests_total", "Total RAG requests"),
            "rag_successes": Counter(
                "voice_rag_successes_total", "Successful RAG requests"
            ),
            "rag_failures": Counter("voice_rag_failures_total", "Failed RAG requests"),
        }

        # Latency histograms
        self._prometheus_histograms = {
            "stt_latency": Histogram(
                "voice_stt_latency_seconds",
                "STT latency",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            ),
            "tts_latency": Histogram(
                "voice_tts_latency_seconds",
                "TTS latency",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            ),
            "rag_latency": Histogram(
                "voice_rag_latency_seconds",
                "RAG latency",
                buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            ),
            "total_latency": Histogram(
                "voice_total_latency_seconds",
                "Total request latency",
                buckets=[1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            ),
        }

        # Gauges
        self._prometheus_gauges = {
            "active_sessions": Gauge("voice_active_sessions", "Active voice sessions"),
            "total_sessions": Gauge("voice_total_sessions", "Total voice sessions"),
        }

    def record_request(self, success: bool, latency: Optional[float] = None):
        """Record a voice request"""
        with self._lock:
            self.metrics.total_requests += 1
            if success:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1

            if latency is not None:
                self.metrics.add_latency("total", latency)

            if self.enable_prometheus:
                self._prometheus_counters["total_requests"].inc()
                if success:
                    self._prometheus_counters["successful_requests"].inc()
                else:
                    self._prometheus_counters["failed_requests"].inc()
                if latency is not None:
                    self._prometheus_histograms["total_latency"].observe(latency)

    def record_stt(
        self,
        success: bool,
        latency: Optional[float] = None,
        provider: Optional[str] = None,
    ):
        """Record STT operation"""
        with self._lock:
            self.metrics.stt_requests += 1
            if success:
                self.metrics.stt_successes += 1
            else:
                self.metrics.stt_failures += 1

            if latency is not None:
                self.metrics.add_latency("stt", latency)

            if self.enable_prometheus:
                self._prometheus_counters["stt_requests"].inc()
                if success:
                    self._prometheus_counters["stt_successes"].inc()
                else:
                    self._prometheus_counters["stt_failures"].inc()
                if latency is not None:
                    self._prometheus_histograms["stt_latency"].observe(latency)

    def record_tts(
        self,
        success: bool,
        latency: Optional[float] = None,
        provider: Optional[str] = None,
    ):
        """Record TTS operation"""
        with self._lock:
            self.metrics.tts_requests += 1
            if success:
                self.metrics.tts_successes += 1
            else:
                self.metrics.tts_failures += 1

            if latency is not None:
                self.metrics.add_latency("tts", latency)

            if self.enable_prometheus:
                self._prometheus_counters["tts_requests"].inc()
                if success:
                    self._prometheus_counters["tts_successes"].inc()
                else:
                    self._prometheus_counters["tts_failures"].inc()
                if latency is not None:
                    self._prometheus_histograms["tts_latency"].observe(latency)

    def record_rag(self, success: bool, latency: Optional[float] = None):
        """Record RAG operation"""
        with self._lock:
            self.metrics.rag_requests += 1
            if success:
                self.metrics.rag_successes += 1
            else:
                self.metrics.rag_failures += 1

            if latency is not None:
                self.metrics.add_latency("rag", latency)

            if self.enable_prometheus:
                self._prometheus_counters["rag_requests"].inc()
                if success:
                    self._prometheus_counters["rag_successes"].inc()
                else:
                    self._prometheus_counters["rag_failures"].inc()
                if latency is not None:
                    self._prometheus_histograms["rag_latency"].observe(latency)

    def record_error(self, error_type: str):
        """Record an error"""
        with self._lock:
            self.metrics.error_counts[error_type] = (
                self.metrics.error_counts.get(error_type, 0) + 1
            )

    def update_sessions(self, active: int, total: int, expired: int = 0):
        """Update session metrics"""
        with self._lock:
            self.metrics.active_sessions = active
            self.metrics.total_sessions = total
            self.metrics.expired_sessions += expired

            if self.enable_prometheus:
                self._prometheus_gauges["active_sessions"].set(active)
                self._prometheus_gauges["total_sessions"].set(total)

    def get_metrics(self) -> Dict:
        """Get current metrics"""
        with self._lock:
            return self.metrics.to_dict()

    def reset_metrics(self):
        """Reset all metrics (use with caution)"""
        with self._lock:
            self.metrics = VoiceMetrics()
            logger.info("Metrics reset")
