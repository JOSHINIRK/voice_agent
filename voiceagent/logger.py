"""Conversation log: one JSON line per turn, the audio of each turn, and a Markdown export."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from voiceagent.config import Settings

if TYPE_CHECKING:
    from voiceagent.agent import Turn


class ConversationLogger:
    def __init__(self, settings: Settings, enabled: bool = True):
        self.settings = settings
        self.enabled = enabled
        self.new_session()

    def new_session(self) -> None:
        self.session_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"

    @property
    def jsonl_path(self) -> Path:
        return self.settings.log_path / "conversations" / f"{self.session_id}.jsonl"

    @property
    def audio_dir(self) -> Path:
        return self.settings.log_path / "audio" / self.session_id

    def log_turn(self, turn: Turn, voice: str) -> dict:
        record = {
            "session": self.session_id,
            "turn": turn.index,
            "time": datetime.now().isoformat(timespec="seconds"),
            "input_mode": turn.input_mode,
            "user": turn.user_text,
            "assistant": turn.reply_text,
            "source": turn.source,
            "ended": turn.ended,
            "timings_s": turn.timings,
            "total_s": turn.total_s,
            "tokens": turn.tokens,
            "models": {
                "stt": self.settings.stt_model if "stt" in turn.timings else None,
                "chat": self.settings.chat_model if turn.source == "gpt" else None,
                "tts": self.settings.tts_model if turn.reply_audio else None,
                "voice": voice if turn.reply_audio else None,
            },
            "error": turn.error,
        }
        if not self.enabled:
            return record

        if self.settings.save_audio:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            if turn.user_audio:
                suffix = Path(turn.user_audio_name or "query.wav").suffix or ".wav"
                path = self.audio_dir / f"turn{turn.index:02d}_user{suffix}"
                path.write_bytes(turn.user_audio)
                record["user_audio"] = path.relative_to(self.settings.log_path).as_posix()
            if turn.reply_audio:
                path = self.audio_dir / f"turn{turn.index:02d}_assistant.{self.settings.tts_format}"
                path.write_bytes(turn.reply_audio)
                record["assistant_audio"] = path.relative_to(self.settings.log_path).as_posix()

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def read_session(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_sessions(settings: Settings) -> list[Path]:
    folder = settings.log_path / "conversations"
    return sorted(folder.glob("*.jsonl"), reverse=True) if folder.exists() else []


def to_markdown(records: list[dict], title: str | None = None) -> str:
    if not records:
        return "# Conversation\n\n(empty)\n"
    lines = [f"# {title or 'Conversation ' + records[0]['session']}", ""]
    voice = next((r["models"].get("voice") for r in records if r["models"].get("voice")), "-")
    lines += [f"Started {records[0]['time']} · TTS voice: {voice}", ""]
    for r in records:
        who = "User (spoken)" if r["input_mode"] == "voice" else "User (typed)"
        stages = " · ".join(f"{k.upper()} {v:.2f}s" for k, v in r.get("timings_s", {}).items())
        lines += [
            f"### Turn {r['turn']}",
            f"*{stages} · total {r['total_s']:.2f}s · handled by: {r['source']}*",
            "",
            f"**{who}:** {r['user'] or '(silence)'}",
            "",
            f"**Assistant:** {r['assistant']}",
            "",
        ]
    return "\n".join(lines)
