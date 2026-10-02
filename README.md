# Yumi — Phase 4: 3D avatar (VRM + Three.js)

Same brain as before (Gemini for chat, FAISS for long-term memory, ElevenLabs
for voice) — now served as a local web app instead of a terminal loop, with
your VRM model rendered live and reacting to what she says.

## What changed from the Phase 1 build

- `main.py` → `app.py`: same personality/memory/voice logic, but exposed as
  a FastAPI server (`POST /chat`) instead of a `while True` input loop.
- Gemini now returns structured JSON each turn — `{reply, emotion, gesture}`
  — instead of plain text, so the frontend knows what face to make and
  whether to play a gesture.
- Voice: `speak()`/`playsound` is gone. ElevenLabs audio is generated
  server-side, sent to the browser as base64, and played there — because
  that's also what drives lip sync (an `AnalyserNode` reads the audio's
  amplitude in real time and puppets the mouth-open blendshape).
- New `static/index.html`: Three.js + `@pixiv/three-vrm` scene that loads
  your uploaded model (`static/yumi.vrm`), a floating chat dock, and a
  message log.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set your API keys (PowerShell):
   ```
   $env:GEMINI_API_KEY="your-key-here"
   $env:ELEVENLABS_API_KEY="your-elevenlabs-key-here"     # optional
   $env:ELEVENLABS_VOICE_ID="21m00Tcm4TlvDq8ikWAM"          # optional
   ```
   Voice is optional — without an ElevenLabs key, Yumi still talks (text +
   expressions + gestures), just silently.

3. Run the server:
   ```
   uvicorn app:app --reload
   ```

4. Open **http://localhost:8000** in your browser. She should load in a few
   seconds (17MB model), then you can type to her in the dock at the bottom.

## How emotions and gestures work

Every reply from Gemini comes back as:
```json
{"reply": "...", "emotion": "happy", "gesture": "wave"}
```
- **Emotion** → cross-fades between VRM's built-in expression presets
  (`happy`, `sad`, `angry`, `surprised`, `relaxed`, `neutral`) using
  `vrm.expressionManager`.
- **Gesture** → a short procedural bone animation (`wave`, `nod`, `think`,
  `laugh`, `shrug`, or `none`), driven directly on the VRM's humanoid bones —
  no external animation files needed. It plays once, then she returns to idle.
- **Idle** → a constant subtle sway, head turn, and blink loop so she's never
  a statue between replies.
- **Lip sync** → amplitude-based: the browser analyzes the ElevenLabs audio
  as it plays and drives the `aa` mouth-open blendshape from volume. This is
  the simple version — good enough for a first pass. A future upgrade is
  phoneme-accurate lip sync via a tool like Rhubarb Lip Sync run on the
  generated audio, which gives much more natural mouth shapes than pure
  amplitude.

## Known limitations / next steps

- Gestures are hand-coded bone rotations, not motion-captured animation clips
  — they're simple and a bit stiff by design. If you want richer movement,
  the next step is sourcing animation clips (e.g. Mixamo, retargeted to VRM
  humanoid bones) and crossfading into them instead of the procedural code
  in `playGesture()`/`applyGesture()`.
- Single-user, single-session server (one `_history` in memory) — fine for
  one person talking to Yumi locally, not built for multiple simultaneous
  users.
- No streaming: each reply waits for the full Gemini + ElevenLabs round trip
  before anything appears. If that feels slow, the natural next step is
  streaming text first and generating audio slightly behind it.

## Earlier phases

- ~~Phase 1: text chat~~ ✅
- ~~Phase 2: voice (ElevenLabs)~~ ✅
- ~~Phase 3: long-term memory (FAISS)~~ ✅
- ~~Phase 4: VRM + Three.js avatar~~ ✅ (this build)
- Phase 5 ideas: mixamo-driven gestures, phoneme lip sync, streaming replies
