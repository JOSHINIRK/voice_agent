"""Speech-to-text with Whisper and text-to-speech with the OpenAI audio API."""

from __future__ import annotations

import logging
from pathlib import Path

from voiceagent.config import Settings

log = logging.getLogger(__name__)

# Formats the transcription endpoint accepts.
SUPPORTED_AUDIO = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"}


class AudioError(ValueError):
    """The recording can't be transcribed (too large or wrong format)."""


def validate_audio(audio: bytes, filename: str, settings: Settings) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO:
        allowed = ", ".join(sorted(SUPPORTED_AUDIO))
        raise AudioError(f"Unsupported audio format '{suffix or filename}'. Use one of: {allowed}.")
    if len(audio) > settings.max_audio_mb * 1024 * 1024:
        raise AudioError(f"The recording is larger than {settings.max_audio_mb} MB.")


def transcribe(client, audio: bytes, filename: str, settings: Settings) -> str:
    """Return the text spoken in `audio` (Whisper)."""
    validate_audio(audio, filename, settings)
    kwargs = {"model": settings.stt_model, "file": (Path(filename).name, audio)}
    if settings.stt_language:
        kwargs["language"] = settings.stt_language
    result = client.audio.transcriptions.create(**kwargs)
    text = result if isinstance(result, str) else getattr(result, "text", "")
    log.info("Transcribed %d bytes -> %r", len(audio), (text or "")[:80])
    return (text or "").strip()


def synthesize(client, text: str, settings: Settings, voice: str | None = None) -> bytes:
    """Return spoken audio (settings.tts_format) for `text`."""
    kwargs = {
        "model": settings.tts_model,
        "voice": voice or settings.tts_voice,
        "input": text[:4000],  # API limit is 4096 characters
        "response_format": settings.tts_format,
        "speed": settings.tts_speed,
    }
    # Only the gpt-4o TTS models accept delivery instructions.
    if settings.tts_instructions and settings.tts_model.startswith("gpt-4o"):
        kwargs["instructions"] = settings.tts_instructions
    response = client.audio.speech.create(**kwargs)
    audio = response.content if hasattr(response, "content") else response.read()
    log.info("Synthesized %d chars -> %d bytes", len(text), len(audio))
    return audio
