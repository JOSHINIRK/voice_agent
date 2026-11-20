"""Offline fakes for the OpenAI client, so the test suite needs no key or network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from voiceagent.config import Settings

FAKE_MP3 = b"ID3fake-mp3-bytes"
FAKE_WAV = b"RIFF" + b"\x00" * 4000


class FakeClient:
    def __init__(self, transcript: str = "What is the capital of France?", reply: str = "Paris is the capital."):
        self.transcript = transcript
        self.reply = reply
        self.calls: dict[str, list[dict]] = {"stt": [], "chat": [], "tts": []}
        self.tts_fails = False

        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._transcribe),
            speech=SimpleNamespace(create=self._speak),
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._complete))

    def _transcribe(self, **kwargs):
        self.calls["stt"].append(kwargs)
        return SimpleNamespace(text=self.transcript)

    def _complete(self, **kwargs):
        self.calls["chat"].append(kwargs)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(total_tokens=42))

    def _speak(self, **kwargs):
        self.calls["tts"].append(kwargs)
        if self.tts_fails:
            raise RuntimeError("tts down")
        return SimpleNamespace(content=FAKE_MP3)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(log_dir=str(tmp_path / "logs"), system_prompt="You are {name}. Now: {now}.")


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()
