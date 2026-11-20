"""Chatbot logic: conversation memory, built-in voice commands and the GPT call."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from voiceagent.config import Settings

DIDNT_CATCH = "Sorry, I didn't catch that. Could you say it again?"
NOTHING_TO_REPEAT = "I haven't said anything yet. What would you like to know?"
RESET_REPLY = "Okay, let's start fresh. What would you like to talk about?"
GOODBYE_REPLY = "Goodbye! It was nice talking with you."

# A command must be the whole utterance (optionally with polite filler), so that
# "can you repeat the history of Rome" still goes to GPT.
_FILLER = r"(?:(?:please|hey|ok|okay|so|um|uh|could you|can you|would you|nova)\s+)*"
_END = r"(?:\s+please)?"
COMMANDS = {
    "repeat": re.compile(
        _FILLER
        + r"(?:repeat(?: that| it| yourself)?|say (?:that|it) again|come again|pardon(?: me)?|what did you say)"
        + r"(?: again)?"
        + _END
    ),
    "reset": re.compile(
        _FILLER
        + r"(?:start over|reset(?: the)?(?: conversation| chat)?|clear(?: the)? (?:conversation|chat|history)"
        + r"|new (?:conversation|chat)|forget everything)"
        + _END
    ),
    "goodbye": re.compile(
        _FILLER
        + r"(?:good ?bye|bye(?: bye)?|see you(?: later)?|exit|quit|stop(?: listening)?|that'?s all|that is all)"
        + r"(?: (?:for now|thanks|thank you))?"
        + _END
    ),
}


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_command(text: str) -> str | None:
    norm = normalize(text)
    for name, pattern in COMMANDS.items():
        if pattern.fullmatch(norm):
            return name
    return None


def clean_for_speech(text: str) -> str:
    """Remove markdown and other things that sound bad when read aloud."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # code blocks
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"https?://\S+", "the link", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)  # headings
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", text, flags=re.M)  # list markers
    text = re.sub(r"(\*\*|__|\*|`)", "", text)  # emphasis / inline code
    text = re.sub(r"\s*\n+\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


@dataclass
class Reply:
    text: str
    source: str  # "gpt" | "command:<name>" | "empty"
    ended: bool = False
    tokens: int = 0


class ChatBrain:
    def __init__(self, client, settings: Settings):
        self.client = client
        self.settings = settings
        self.history: list[dict[str, str]] = []
        self.last_reply: str | None = None

    def system_message(self) -> dict[str, str]:
        now = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
        prompt = self.settings.system_prompt.format(name=self.settings.assistant_name, now=now)
        return {"role": "system", "content": prompt}

    def reset(self) -> None:
        self.history.clear()

    def respond(self, user_text: str) -> Reply:
        user_text = (user_text or "").strip()
        if len(normalize(user_text)) < 2:
            return Reply(DIDNT_CATCH, "empty")

        command = detect_command(user_text)
        if command == "repeat":
            return Reply(self.last_reply or NOTHING_TO_REPEAT, "command:repeat")
        if command == "reset":
            self.reset()
            return self._remember(Reply(RESET_REPLY, "command:reset"))
        if command == "goodbye":
            self.reset()
            return self._remember(Reply(GOODBYE_REPLY, "command:goodbye", ended=True))

        return self._remember(self._ask_gpt(user_text))

    def _ask_gpt(self, user_text: str) -> Reply:
        keep = self.settings.history_turns * 2
        recent = self.history[-keep:] if keep > 0 else []
        messages = [self.system_message(), *recent, {"role": "user", "content": user_text}]
        kwargs = {
            "model": self.settings.chat_model,
            "messages": messages,
            "max_completion_tokens": self.settings.max_reply_tokens,
        }
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        completion = self.client.chat.completions.create(**kwargs)
        text = clean_for_speech(completion.choices[0].message.content or "") or DIDNT_CATCH
        usage = getattr(completion, "usage", None)
        tokens = getattr(usage, "total_tokens", 0) or 0

        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": text}]
        return Reply(text, "gpt", tokens=tokens)

    def _remember(self, reply: Reply) -> Reply:
        self.last_reply = reply.text
        return reply
