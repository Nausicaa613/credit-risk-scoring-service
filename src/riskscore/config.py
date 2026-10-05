"""Runtime configuration.

All settings come from environment variables so the same artifact can run in a
container, in CI, or on a laptop without code changes. Defaults are chosen so
that a fresh clone works with no configuration at all.

Environment variables
---------------------
``RISKSCORE_HOST``            bind address                (default 127.0.0.1)
``RISKSCORE_PORT``            TCP port                    (default 8080)
``RISKSCORE_DB_PATH``         SQLite file path            (default data/riskscore.db)
``RISKSCORE_MODEL_PATH``      model artifact path         (default models/model.json)
``RISKSCORE_DECISION_THRESHOLD``  PD above which we decline (default 0.35)
``RISKSCORE_MAX_BODY_BYTES``  request body cap            (default 65536)
``RISKSCORE_LOG_LEVEL``       DEBUG/INFO/WARNING/ERROR    (default INFO)
``RISKSCORE_METRICS_ENABLED`` expose /metrics             (default 1)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Union

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_DB_PATH = "data/riskscore.db"
DEFAULT_MODEL_PATH = "models/model.json"
DEFAULT_DECISION_THRESHOLD = 0.35
DEFAULT_MAX_BODY_BYTES = 64 * 1024
DEFAULT_LOG_LEVEL = "INFO"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _parse_bool(raw: str, *, name: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be one of {sorted(_TRUE | _FALSE)}, got {raw!r}")


def _parse_int(raw: str, *, name: str, minimum: int = 1) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _parse_float(raw: str, *, name: str, low: float, high: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not low <= value <= high:
        raise ValueError(f"{name} must be within [{low}, {high}], got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable service settings."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    db_path: str = DEFAULT_DB_PATH
    model_path: str = DEFAULT_MODEL_PATH
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    log_level: str = DEFAULT_LOG_LEVEL
    metrics_enabled: bool = True

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        """Build settings from a mapping, defaulting to ``os.environ``.

        Unknown keys are ignored, so callers can pass a whole environment.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        settings = cls()

        def get(name: str) -> Optional[str]:
            raw = source.get(name)
            if raw is None or raw == "":
                return None
            return raw

        overrides: Dict[str, Any] = {}
        if (raw := get("RISKSCORE_HOST")) is not None:
            overrides["host"] = raw
        if (raw := get("RISKSCORE_PORT")) is not None:
            overrides["port"] = _parse_int(raw, name="RISKSCORE_PORT")
        if (raw := get("RISKSCORE_DB_PATH")) is not None:
            overrides["db_path"] = raw
        if (raw := get("RISKSCORE_MODEL_PATH")) is not None:
            overrides["model_path"] = raw
        if (raw := get("RISKSCORE_DECISION_THRESHOLD")) is not None:
            overrides["decision_threshold"] = _parse_float(
                raw, name="RISKSCORE_DECISION_THRESHOLD", low=0.0, high=1.0
            )
        if (raw := get("RISKSCORE_MAX_BODY_BYTES")) is not None:
            overrides["max_body_bytes"] = _parse_int(raw, name="RISKSCORE_MAX_BODY_BYTES")
        if (raw := get("RISKSCORE_LOG_LEVEL")) is not None:
            overrides["log_level"] = raw.strip().upper()
        if (raw := get("RISKSCORE_METRICS_ENABLED")) is not None:
            overrides["metrics_enabled"] = _parse_bool(raw, name="RISKSCORE_METRICS_ENABLED")

        return replace(settings, **overrides)

    def as_dict(self) -> Dict[str, Union[str, int, float, bool]]:
        """Serialise for logging and for the ``/healthz`` payload."""
        return {
            "host": self.host,
            "port": self.port,
            "db_path": self.db_path,
            "model_path": self.model_path,
            "decision_threshold": self.decision_threshold,
            "max_body_bytes": self.max_body_bytes,
            "log_level": self.log_level,
            "metrics_enabled": self.metrics_enabled,
        }


__all__ = ["Settings", "DEFAULT_DECISION_THRESHOLD"]
