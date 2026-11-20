"""Streamlit interface: record a question, hear the answer.

Run with:  .\\.venv\\Scripts\\python.exe -m streamlit run app.py
"""

from __future__ import annotations

import json

import streamlit as st

from voiceagent.agent import Turn, VoiceAgent
from voiceagent.config import has_api_key, load_settings
from voiceagent.logger import list_sessions, read_session, to_markdown
from voiceagent.speech import SUPPORTED_AUDIO, AudioError

VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]
AUDIO_MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac"}
SAMPLE_PROMPTS = [
    "What can you help me with?",
    "Give me three tips to stay focused while studying.",
    "Explain what Whisper does in one sentence.",
]

st.set_page_config(page_title="Mini Voice Agent", page_icon="🎙️", layout="centered")
settings = load_settings()

if not has_api_key():
    st.title("🎙️ Mini Voice Agent")
    st.error("No OpenAI API key found.")
    st.markdown(
        "1. Copy `.env.example` to `.env` in the project folder.\n"
        "2. Paste your key after `OPENAI_API_KEY=`.\n"
        "3. Restart the app."
    )
    st.stop()

state = st.session_state
if "agent" not in state:
    state.agent = VoiceAgent(settings)
    state.mic_key = 0  # bumped after each recording so the recorder clears itself
    state.upload_key = 0
    state.autoplay_turn = None
agent: VoiceAgent = state.agent

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    voice = st.selectbox("Assistant voice", VOICES, index=VOICES.index(settings.tts_voice))
    speak = st.toggle("Speak replies (TTS)", value=True)
    autoplay = st.toggle("Auto-play replies", value=True, disabled=not speak)
    st.caption(
        f"STT: `{settings.stt_model}` · Chat: `{settings.chat_model}` · TTS: `{settings.tts_model}`\n\n"
        f"Memory: last {settings.history_turns} exchanges"
    )
    if st.button("🔄 New conversation", use_container_width=True):
        agent.reset()
        state.autoplay_turn = None
        st.rerun()

    st.divider()
    st.subheader("Voice commands")
    st.markdown(
        "- **“Repeat that”**: replays the last answer\n"
        "- **“Start over”**: clears the conversation memory\n"
        "- **“Goodbye”**: ends the conversation"
    )

    st.divider()
    st.subheader("This session's log")
    st.caption(f"`logs/conversations/{agent.logger.session_id}.jsonl`")
    if agent.logger.jsonl_path.exists():
        records = read_session(agent.logger.jsonl_path)
        st.download_button(
            "Download as Markdown",
            to_markdown(records),
            file_name=f"conversation-{agent.logger.session_id}.md",
            use_container_width=True,
        )
        st.download_button(
            "Download as JSON Lines",
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
            file_name=f"conversation-{agent.logger.session_id}.jsonl",
            use_container_width=True,
        )


# ---------------------------------------------------------------- helpers
def run_turn(fn, *args) -> None:
    """Run one turn with a status line, then rerun so the transcript re-renders."""
    try:
        with st.status("Working...", expanded=False) as status:
            status.update(label="Listening to your recording..." if fn == agent.handle_audio else "Thinking...")
            turn: Turn = fn(*args, voice=voice, speak=speak)
            status.update(label=f"Done in {turn.total_s:.1f}s", state="complete")
        state.autoplay_turn = turn.index if speak and autoplay else None
    except AudioError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # network, quota, bad key...
        st.error(f"Something went wrong: {exc}")
        return
    st.rerun()


def render_turn(turn: Turn) -> None:
    with st.chat_message("user", avatar="🎙️" if turn.input_mode == "voice" else "⌨️"):
        st.markdown(turn.user_text or "_(no speech detected)_")
        if turn.user_audio:
            st.audio(turn.user_audio, format="audio/wav")
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(turn.reply_text)
        if turn.reply_audio:
            st.audio(
                turn.reply_audio,
                format=AUDIO_MIME.get(settings.tts_format, "audio/mpeg"),
                autoplay=turn.index == state.autoplay_turn,
            )
        stages = " · ".join(f"{k.upper()} {v:.2f}s" for k, v in turn.timings.items())
        label = "built-in command" if turn.source.startswith("command") else turn.source
        st.caption(f"{stages} · total {turn.total_s:.2f}s · {label}")
        if turn.error:
            st.warning(turn.error)


# ---------------------------------------------------------------- main
st.title("🎙️ Mini Voice Agent")
st.caption("Speak a question → Whisper transcribes it → GPT answers → the answer is spoken back.")

talk_tab, logs_tab = st.tabs(["Talk", "Conversation logs"])

with talk_tab:
    if not agent.turns:
        st.info("Click the microphone below, ask a question, then click stop. Or type, if you prefer.")
        cols = st.columns(len(SAMPLE_PROMPTS))
        for col, prompt in zip(cols, SAMPLE_PROMPTS):
            if col.button(prompt, use_container_width=True):
                run_turn(agent.handle_text, prompt)

    for turn in agent.turns:
        render_turn(turn)

    if agent.turns and agent.turns[-1].ended:
        st.success("Conversation ended. Start speaking again to begin a new one, or click **New conversation**.")

    recording = st.audio_input("Record your question", key=f"mic_{state.mic_key}")
    if recording is not None:
        state.mic_key += 1
        run_turn(agent.handle_audio, recording.getvalue(), recording.name or "query.wav")

    with st.expander("Or upload an audio file"):
        upload = st.file_uploader(
            "Audio file", type=sorted(s.lstrip(".") for s in SUPPORTED_AUDIO), key=f"upload_{state.upload_key}"
        )
        if upload is not None and st.button("Send audio file"):
            state.upload_key += 1
            run_turn(agent.handle_audio, upload.getvalue(), upload.name)

    typed = st.chat_input("...or type a message")
    if typed:
        run_turn(agent.handle_text, typed)

with logs_tab:
    sessions = list_sessions(settings)
    if not sessions:
        st.info("No conversations logged yet.")
    else:
        chosen = st.selectbox("Session", sessions, format_func=lambda p: p.stem)
        records = read_session(chosen)
        total = sum(r["total_s"] for r in records)
        c1, c2, c3 = st.columns(3)
        c1.metric("Turns", len(records))
        c2.metric("Avg. response time", f"{total / len(records):.1f}s")
        c3.metric("GPT tokens", sum(r.get("tokens", 0) for r in records))
        st.markdown(to_markdown(records))
        for r in records:
            if r.get("assistant_audio"):
                path = settings.log_path / r["assistant_audio"]
                if path.exists():
                    st.audio(path.read_bytes(), format="audio/mpeg")
                    st.caption(f"Turn {r['turn']} reply")
