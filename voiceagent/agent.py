"""End-to-end voice turn: audio -> Whisper -> chatbot/GPT -> TTS audio, with timings and logging."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from voiceagent.chat import DIDNT_CATCH, ChatBrain, Reply
from voiceagent.config import Settings
from voiceagent.logger import ConversationLogger
from voiceagent.speech import synthesize, transcribe

log = logging.getLogger(__name__)


def make_client(settings: Settings):
    from openai import OpenAI

    return OpenAI(timeout=settings.request_timeout_s, max_retries=2)


@dataclass
class Turn:
    index: int
    user_text: str
    reply_text: str
    source: str  # "gpt" | "command:<name>" | "empty"
    input_mode: str  # "voice" | "text"
    reply_audio: bytes | None = None
    user_audio: bytes | None = None
    user_audio_name: str | None = None
    ended: bool = False
    tokens: int = 0
    timings: dict[str, float] = field(default_factory=dict)  # seconds per stage: stt, llm, tts
    error: str | None = None

    @property
    def total_s(self) -> float:
        return round(sum(self.timings.values()), 2)


class VoiceAgent:
    def __init__(self, settings: Settings, client=None, logger: ConversationLogger | None = None):
        self.settings = settings
        self.client = client or make_client(settings)
        self.brain = ChatBrain(self.client, settings)
        self.logger = logger or ConversationLogger(settings)
        self.turns: list[Turn] = []

    # -- public API ---------------------------------------------------------
    def handle_audio(
        self, audio: bytes, filename: str = "query.wav", voice: str | None = None, speak: bool = True
    ) -> Turn:
        """Full voice turn: transcribe the recording, reply, and speak the reply."""
        timings: dict[str, float] = {}
        if len(audio) < self.settings.min_audio_bytes:
            return self._finish("", Reply(DIDNT_CATCH, "empty"), "voice", timings, voice, speak, audio, filename)
        start = time.perf_counter()
        user_text = transcribe(self.client, audio, filename, self.settings)
        timings["stt"] = time.perf_counter() - start
        return self._respond(user_text, "voice", timings, voice, speak, audio, filename)

    def handle_text(self, text: str, voice: str | None = None, speak: bool = True) -> Turn:
        """Typed fallback: same chatbot and TTS, no transcription."""
        return self._respond(text, "text", {}, voice, speak, None, None)

    def reset(self) -> None:
        self.brain.reset()
        self.brain.last_reply = None
        self.turns.clear()
        self.logger.new_session()

    # -- internals ----------------------------------------------------------
    def _respond(self, user_text, mode, timings, voice, speak, audio, filename) -> Turn:
        start = time.perf_counter()
        reply = self.brain.respond(user_text)
        timings["llm"] = time.perf_counter() - start
        return self._finish(user_text, reply, mode, timings, voice, speak, audio, filename)

    def _finish(self, user_text, reply: Reply, mode, timings, voice, speak, audio, filename) -> Turn:
        turn = Turn(
            index=len(self.turns) + 1,
            user_text=user_text,
            reply_text=reply.text,
            source=reply.source,
            input_mode=mode,
            user_audio=audio,
            user_audio_name=filename,
            ended=reply.ended,
            tokens=reply.tokens,
        )
        if speak:
            start = time.perf_counter()
            try:
                turn.reply_audio = synthesize(self.client, reply.text, self.settings, voice)
            except Exception as exc:  # the text reply is still useful without audio
                log.exception("Text-to-speech failed")
                turn.error = f"Text-to-speech failed: {exc}"
            timings["tts"] = time.perf_counter() - start
        turn.timings = {k: round(v, 2) for k, v in timings.items()}
        self.turns.append(turn)
        self.logger.log_turn(turn, voice or self.settings.tts_voice)
        return turn
