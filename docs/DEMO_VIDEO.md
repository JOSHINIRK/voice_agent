# Recording the demo video

A 2–3 minute screen recording with system audio, so viewers hear both your question and the spoken reply.

## Setup

1. Start the app: `.\.venv\Scripts\python.exe -m streamlit run app.py` and open http://localhost:8501.
2. Click the mic once and allow microphone access, so the permission pop-up doesn't appear on camera.
3. Click **New conversation** in the sidebar.
4. Use headphones or keep speaker volume moderate, so the reply isn't picked up by the mic.

## Recording tool (built into Windows 11)

- **Xbox Game Bar:** `Win + Alt + R` starts and stops recording of the browser window. Turn on "Record mic" in Game Bar settings (`Win + G` → Settings → Capturing) so your voice is captured along with the reply audio. Videos land in `Videos\Captures`.
- **Or Snipping Tool:** `Win + Shift + R` → select the area → Record (turn on the mic and system audio toggles).
- Trim it in **Clipchamp** (preinstalled) if needed.

## Script (about 2.5 minutes)

| # | Show / say | What it demonstrates |
|---|---|---|
| 1 | 10 s intro over the empty app: "This is a voice assistant: Whisper for speech-to-text, GPT for the answer, OpenAI TTS for the spoken reply." | Overview |
| 2 | Mic → *"Hi, what can you help me with?"* | Full voice turn, auto-played reply |
| 3 | Mic → *"Why is the sky blue?"* | General question |
| 4 | Mic → *"Explain that more simply, like I'm ten."* | Conversation memory (follow-up) |
| 5 | Point at the caption under a reply: STT / LLM / TTS seconds | Latency per stage |
| 6 | Mic → *"Repeat that."* | Built-in command, no GPT call |
| 7 | Sidebar → change voice to **nova**, mic → *"Give me a tip to stay focused."* | Voice selection |
| 8 | Type *"Convert 30 Celsius to Fahrenheit"* in the chat box | Typed fallback, still spoken |
| 9 | Mic → *"Goodbye."* | Conversation end |
| 10 | **Conversation logs** tab: metrics, transcript, replay audio; sidebar **Download as Markdown** | Logging |
| 11 | Optional: terminal `python -m pytest` showing 27 passed | Testing |

Save the final video as `docs/demo.mp4` (or upload it and link it in the README).
