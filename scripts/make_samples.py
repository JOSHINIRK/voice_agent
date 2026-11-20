"""Generate the sample conversations in samples/ with real API calls.

For each scripted conversation, every user question is first spoken with a TTS
"user" voice, then sent through the full pipeline exactly like a microphone
recording: Whisper transcription -> chatbot / GPT -> TTS reply. The logged turns
are exported to samples/conversations/<name>.md and .jsonl, and the audio to
samples/audio/<name>/.

    python scripts/make_samples.py              # all conversations
    python scripts/make_samples.py study_help   # just one
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voiceagent.agent import VoiceAgent  # noqa: E402
from voiceagent.config import has_api_key, load_settings  # noqa: E402
from voiceagent.logger import read_session, to_markdown  # noqa: E402
from voiceagent.speech import synthesize  # noqa: E402

USER_VOICE = "echo"  # voice that plays the user, different from the assistant's

CONVERSATIONS: dict[str, dict] = {
    "general_knowledge": {
        "title": "Sample 1: General knowledge with follow-up questions",
        "turns": [
            "Hi! What can you help me with?",
            "Why is the sky blue?",
            "Can you explain that more simply, like I'm ten years old?",
            "Thanks. Goodbye!",
        ],
    },
    "study_help": {
        "title": "Sample 2: Study help, memory and the repeat command",
        "turns": [
            "I have an exam on machine learning next week. Give me a quick study plan.",
            "What's the difference between supervised and unsupervised learning?",
            "Repeat that, please.",
            "Which of the two does Whisper use?",
            "Start over.",
            "What did I say my exam was about?",
        ],
    },
    "daily_assistant": {
        "title": "Sample 3: Everyday assistant questions",
        "turns": [
            "What day is it today?",
            "Suggest a quick healthy breakfast I can make in five minutes.",
            "Convert 30 degrees Celsius to Fahrenheit.",
            "What's the weather like in Chennai right now?",
            "That's all, thank you.",
        ],
    },
}


def run(name: str, spec: dict, settings) -> None:
    agent = VoiceAgent(settings)
    print(f"\n=== {spec['title']} ===")
    for question in spec["turns"]:
        spoken = synthesize(agent.client, question, settings, voice=USER_VOICE)  # simulated microphone input
        turn = agent.handle_audio(spoken, f"question.{settings.tts_format}")
        print(f"  You (spoken):     {question}")
        print(f"  Whisper heard:    {turn.user_text}")
        print(f"  Assistant:        {turn.reply_text}")
        print(f"  timings: {turn.timings} total={turn.total_s}s via {turn.source}\n")

    records = read_session(agent.logger.jsonl_path)
    out_dir = ROOT / "samples" / "conversations"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out = ROOT / "samples" / "audio" / name
    if audio_out.exists():
        shutil.rmtree(audio_out)
    shutil.copytree(agent.logger.audio_dir, audio_out)
    for r in records:  # point audio paths at the copies under samples/
        for key in ("user_audio", "assistant_audio"):
            if r.get(key):
                r[key] = f"audio/{name}/{Path(r[key]).name}"
    (out_dir / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    (out_dir / f"{name}.md").write_text(to_markdown(records, spec["title"]), encoding="utf-8")
    print(f"  -> samples/conversations/{name}.md")


def main() -> None:
    settings = load_settings()
    if not has_api_key():
        sys.exit("OPENAI_API_KEY is missing: copy .env.example to .env and add your key.")
    names = sys.argv[1:] or list(CONVERSATIONS)
    for name in names:
        run(name, CONVERSATIONS[name], settings)


if __name__ == "__main__":
    main()
