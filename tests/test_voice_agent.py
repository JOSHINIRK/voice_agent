from __future__ import annotations

import pytest

from tests.conftest import FAKE_MP3, FAKE_WAV
from voiceagent.agent import VoiceAgent
from voiceagent.chat import DIDNT_CATCH, GOODBYE_REPLY, RESET_REPLY, clean_for_speech, detect_command
from voiceagent.config import Settings, load_settings
from voiceagent.logger import read_session, to_markdown
from voiceagent.speech import AudioError, synthesize, transcribe


# ------------------------------------------------------------------ pipeline
def test_voice_turn_runs_stt_llm_tts_and_logs(settings, client):
    agent = VoiceAgent(settings, client=client)
    turn = agent.handle_audio(FAKE_WAV, "query.wav")

    assert turn.user_text == "What is the capital of France?"
    assert turn.reply_text == "Paris is the capital."
    assert turn.reply_audio == FAKE_MP3
    assert set(turn.timings) == {"stt", "llm", "tts"}
    assert client.calls["stt"][0]["model"] == "whisper-1"
    assert client.calls["stt"][0]["file"] == ("query.wav", FAKE_WAV)

    [record] = read_session(agent.logger.jsonl_path)
    assert record["user"] == turn.user_text and record["assistant"] == turn.reply_text
    assert record["models"]["stt"] == "whisper-1"
    assert (settings.log_path / record["user_audio"]).read_bytes() == FAKE_WAV
    assert (settings.log_path / record["assistant_audio"]).read_bytes() == FAKE_MP3


def test_history_is_sent_with_follow_ups(settings, client):
    agent = VoiceAgent(settings, client=client)
    agent.handle_text("first question")
    agent.handle_text("second question")
    messages = client.calls["chat"][1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "first question"


def test_history_is_capped(client, tmp_path):
    settings = Settings(log_dir=str(tmp_path), history_turns=1)
    agent = VoiceAgent(settings, client=client)
    for i in range(4):
        agent.handle_text(f"question {i}")
    messages = client.calls["chat"][-1]["messages"]
    assert len(messages) == 4  # system + 1 exchange + new question
    assert messages[1]["content"] == "question 2"


def test_tiny_recording_is_not_sent_to_whisper(settings, client):
    turn = VoiceAgent(settings, client=client).handle_audio(b"RIFF", "query.wav")
    assert turn.reply_text == DIDNT_CATCH and turn.source == "empty"
    assert client.calls["stt"] == [] and client.calls["chat"] == []


def test_empty_transcript_asks_to_repeat(settings, client):
    client.transcript = "  "
    turn = VoiceAgent(settings, client=client).handle_audio(FAKE_WAV, "query.wav")
    assert turn.reply_text == DIDNT_CATCH and client.calls["chat"] == []


def test_tts_failure_keeps_text_reply(settings, client):
    client.tts_fails = True
    turn = VoiceAgent(settings, client=client).handle_text("hello there")
    assert turn.reply_text == "Paris is the capital."
    assert turn.reply_audio is None and "tts down" in turn.error


def test_speak_false_skips_tts(settings, client):
    turn = VoiceAgent(settings, client=client).handle_text("hello there", speak=False)
    assert turn.reply_audio is None and client.calls["tts"] == []


# ------------------------------------------------------------------ commands
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Repeat that, please.", "repeat"),
        ("Can you say that again?", "repeat"),
        ("Start over.", "reset"),
        ("Clear the conversation", "reset"),
        ("Goodbye!", "goodbye"),
        ("That's all, thank you.", "goodbye"),
        ("Can you repeat the history of Rome?", None),
        ("How do I stop procrastinating?", None),
        ("Say goodbye in French", None),
    ],
)
def test_detect_command(text, expected):
    assert detect_command(text) == expected


def test_repeat_replays_last_answer_without_gpt(settings, client):
    agent = VoiceAgent(settings, client=client)
    agent.handle_text("question")
    turn = agent.handle_text("repeat that")
    assert turn.reply_text == "Paris is the capital." and turn.source == "command:repeat"
    assert len(client.calls["chat"]) == 1


def test_reset_clears_memory(settings, client):
    agent = VoiceAgent(settings, client=client)
    agent.handle_text("my name is Sam")
    assert agent.handle_text("start over").reply_text == RESET_REPLY
    agent.handle_text("what's my name?")
    assert len(client.calls["chat"][-1]["messages"]) == 2  # system + question only


def test_goodbye_ends_conversation(settings, client):
    turn = VoiceAgent(settings, client=client).handle_text("bye")
    assert turn.ended and turn.reply_text == GOODBYE_REPLY


# ------------------------------------------------------------------ speech helpers
def test_clean_for_speech_strips_markdown():
    text = "## Plan\n- **Read** chapter 1\n- See [docs](https://x.y)\n1. Practice `code`"
    assert clean_for_speech(text) == "Plan Read chapter 1 See docs Practice code"


def test_transcribe_rejects_unsupported_format(settings, client):
    with pytest.raises(AudioError):
        transcribe(client, FAKE_WAV, "notes.txt", settings)


def test_transcribe_rejects_oversized_audio(client, tmp_path):
    settings = Settings(log_dir=str(tmp_path), max_audio_mb=0.001)
    with pytest.raises(AudioError):
        transcribe(client, b"x" * 5000, "q.wav", settings)


def test_tts_instructions_only_for_gpt4o_models(settings, client):
    settings.tts_instructions = "be cheerful"
    synthesize(client, "hi", settings)
    assert client.calls["tts"][-1]["instructions"] == "be cheerful"
    settings.tts_model = "tts-1"
    synthesize(client, "hi", settings, voice="nova")
    assert "instructions" not in client.calls["tts"][-1] and client.calls["tts"][-1]["voice"] == "nova"


# ------------------------------------------------------------------ config & logs
def test_env_overrides_config(monkeypatch):
    monkeypatch.setenv("VOICE_TTS_VOICE", "nova")
    monkeypatch.setenv("VOICE_HISTORY_TURNS", "2")
    monkeypatch.setenv("VOICE_TEMPERATURE", "null")
    s = load_settings()
    assert (s.tts_voice, s.history_turns, s.temperature) == ("nova", 2, None)


def test_temperature_none_is_not_sent(settings, client):
    settings.temperature = None
    VoiceAgent(settings, client=client).handle_text("hi there")
    assert "temperature" not in client.calls["chat"][0]


def test_markdown_export(settings, client):
    agent = VoiceAgent(settings, client=client)
    agent.handle_audio(FAKE_WAV, "query.wav")
    agent.handle_text("bye")
    md = to_markdown(read_session(agent.logger.jsonl_path), "Demo")
    assert md.startswith("# Demo")
    assert "**User (spoken):** What is the capital of France?" in md
    assert "**User (typed):** bye" in md and GOODBYE_REPLY in md


def test_new_conversation_starts_new_log(settings, client):
    agent = VoiceAgent(settings, client=client)
    agent.handle_text("hello there")
    first = agent.logger.jsonl_path
    agent.reset()
    agent.handle_text("hello again")
    assert agent.logger.jsonl_path != first and agent.turns[0].index == 1
