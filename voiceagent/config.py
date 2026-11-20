"""Settings from config.yaml, overridable with VOICE_<KEY> environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class Settings:
    stt_model: str = "whisper-1"
    stt_language: str = "en"

    chat_model: str = "gpt-4.1-mini"
    temperature: float | None = 0.6
    max_reply_tokens: int = 300
    history_turns: int = 6
    assistant_name: str = "Nova"
    system_prompt: str = "You are {name}, a friendly voice assistant. The time is {now}."

    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    tts_format: str = "mp3"
    tts_speed: float = 1.0
    tts_instructions: str = ""

    log_dir: str = "logs"
    save_audio: bool = True

    max_audio_mb: float = 25
    min_audio_bytes: int = 2000
    request_timeout_s: float = 60

    log_level: str = "INFO"

    @property
    def log_path(self) -> Path:
        path = Path(self.log_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _coerce(raw: str, current: Any) -> Any:
    """Convert an environment-variable string to the type of the default value."""
    if raw.lower() in ("null", "none", ""):
        return None
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


class _IgnoreClientDisconnects(logging.Filter):
    """Drop asyncio's 'connection forcibly closed' tracebacks on Windows.

    When a browser tab refreshes or closes, Windows' Proactor event loop logs a
    ConnectionResetError (WinError 10054) while closing the socket. It is
    harmless: the server keeps running, so the traceback is just noise.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not isinstance(exc, (ConnectionResetError, ConnectionAbortedError))


def _quiet_disconnect_errors() -> None:
    asyncio_log = logging.getLogger("asyncio")
    if not any(isinstance(f, _IgnoreClientDisconnects) for f in asyncio_log.filters):
        asyncio_log.addFilter(_IgnoreClientDisconnects())


def load_settings(path: Path | None = None) -> Settings:
    _quiet_disconnect_errors()
    load_dotenv(PROJECT_ROOT / ".env")
    data: dict[str, Any] = {}
    config_file = path or CONFIG_PATH
    if config_file.exists():
        data = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}

    settings = Settings()
    known = {f.name for f in fields(Settings)}
    for key, value in data.items():
        if key in known:
            setattr(settings, key, value)
    for name in known:
        raw = os.environ.get(f"VOICE_{name.upper()}")
        if raw is not None:
            setattr(settings, name, _coerce(raw, getattr(settings, name)))

    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return settings


def has_api_key() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key) and "your-key-here" not in key
