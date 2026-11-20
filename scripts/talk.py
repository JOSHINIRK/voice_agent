"""Terminal voice loop: record from the microphone (or pass audio files), hear the reply.

    python scripts/talk.py                  # push-to-talk with the microphone (needs: pip install sounddevice)
    python scripts/talk.py --file q.wav     # send an existing recording
    python scripts/talk.py --text           # type instead of speaking; replies are still spoken

Replies are played with the built-in `winsound` module on Windows, so no extra player is needed.
"""

from __future__ import annotations

import argparse
import io
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voiceagent.agent import VoiceAgent  # noqa: E402
from voiceagent.config import has_api_key, load_settings  # noqa: E402

SAMPLE_RATE = 16_000


def record_push_to_talk() -> bytes:
    """Record from the default microphone between two Enter presses; return WAV bytes."""
    import numpy as np
    import sounddevice as sd

    input("\n[Enter] to start speaking...")
    chunks: list[np.ndarray] = []
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=lambda d, *_: chunks.append(d.copy()))
    with stream:
        input("Recording... [Enter] to stop.")
    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype="int16")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio.tobytes())
    return buf.getvalue()


def play(audio: bytes | None) -> None:
    if not audio:
        return
    if sys.platform == "win32":
        import winsound

        winsound.PlaySound(audio, winsound.SND_MEMORY)
    else:
        out = Path("last_reply.wav")
        out.write_bytes(audio)
        print(f"(reply audio saved to {out})")


def show(turn) -> None:
    print(f"You:  {turn.user_text or '(silence)'}")
    print(f"Bot:  {turn.reply_text}")
    stages = "  ".join(f"{k}={v:.2f}s" for k, v in turn.timings.items())
    print(f"      [{stages}  total={turn.total_s:.2f}s  {turn.source}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", nargs="+", help="audio file(s) to send instead of using the microphone")
    parser.add_argument("--text", action="store_true", help="type questions instead of speaking")
    parser.add_argument("--voice", help="TTS voice, e.g. nova")
    parser.add_argument("--mute", action="store_true", help="don't play the replies")
    args = parser.parse_args()

    settings = load_settings()
    settings.tts_format = "wav"  # winsound plays WAV only
    if not has_api_key():
        sys.exit("OPENAI_API_KEY is missing: copy .env.example to .env and add your key.")
    agent = VoiceAgent(settings)
    print(f"Logging to {agent.logger.jsonl_path}")

    if args.file:
        for name in args.file:
            path = Path(name)
            turn = agent.handle_audio(path.read_bytes(), path.name, voice=args.voice)
            show(turn)
            if not args.mute:
                play(turn.reply_audio)
        return

    if not args.text:
        try:
            import sounddevice  # noqa: F401
        except Exception:
            sys.exit("Microphone mode needs 'pip install sounddevice'. Or use --text / --file, or the web app.")

    print("Say 'goodbye' to finish. Ctrl+C to quit.")
    while True:
        try:
            if args.text:
                typed = input("\nYou (type): ").strip()
                turn = agent.handle_text(typed, voice=args.voice)
            else:
                turn = agent.handle_audio(record_push_to_talk(), "query.wav", voice=args.voice)
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            return
        show(turn)
        if not args.mute:
            play(turn.reply_audio)
        if turn.ended:
            return


if __name__ == "__main__":
    main()
