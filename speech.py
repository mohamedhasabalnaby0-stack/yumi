"""Audio-clock lip cues: optional Rhubarb recognition, aligned-text fallback."""
import base64
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave


MOUTHS = {
    "A": {}, "B": {"ih": 0.25}, "C": {"aa": 0.5, "ee": 0.15},
    "D": {"aa": 0.85}, "E": {"oh": 0.7}, "F": {"ou": 0.7},
    "G": {"ee": 0.2}, "H": {"ih": 0.45}, "X": {},
}


def field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def pcm_wave(pcm):
    if not pcm or len(pcm) % 2:
        raise ValueError("Invalid 16-bit PCM audio")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    return output.getvalue()


def aligned_cues(alignment, duration):
    """Character timings are NOT phoneme recognition; expose that distinction to UI."""
    chars = field(alignment, "characters", []) or []
    starts = field(alignment, "character_start_times_seconds", []) or []
    ends = field(alignment, "character_end_times_seconds", []) or []
    if not (len(chars) == len(starts) == len(ends)):
        return []
    cues = []
    previous = 0.0
    for char, start, end in zip(chars, starts, ends):
        if not isinstance(char, str):
            continue
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        start, end = max(0.0, previous, start), min(duration, end)
        if end <= start:
            continue
        letter = char.casefold()
        shape = {}
        if letter in ("a",):
            shape = {"aa": 0.75}
        elif letter in ("e",):
            shape = {"ee": 0.6}
        elif letter in ("i", "y"):
            shape = {"ih": 0.55}
        elif letter in ("o",):
            shape = {"oh": 0.7}
        elif letter in ("u", "w", "q"):
            shape = {"ou": 0.65}
        elif letter.isalpha() and letter not in ("m", "b", "p"):
            shape = {"ih": 0.22}
        cues.append({"start": start, "end": end, "weights": shape})
        previous = end
    return cues


def rhubarb_cues(wav, text, duration):
    executable = shutil.which(os.environ.get("RHUBARB_PATH", "rhubarb"))
    if not executable:
        return []
    with tempfile.TemporaryDirectory(prefix="yumi-speech-") as directory:
        audio = Path(directory) / "speech.wav"
        dialog = Path(directory) / "dialog.txt"
        audio.write_bytes(wav)
        dialog.write_text(text, encoding="utf-8")
        # No shell, fixed argument names, temporary local files, bounded runtime.
        process = subprocess.run(
            [executable, "-f", "json", "-r", "phonetic", "--dialogFile", str(dialog), str(audio)],
            capture_output=True, text=True, timeout=20, check=True,
        )
        cues = []
        previous = 0.0
        for cue in json.loads(process.stdout).get("mouthCues", []):
            start, end = float(cue["start"]), float(cue["end"])
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            start, end = max(previous, 0.0, start), min(duration, end)
            if end <= start or cue.get("value") not in MOUTHS:
                continue
            cues.append({"start": start, "end": end, "weights": MOUTHS[cue["value"]]})
            previous = end
        return cues


def synthesize(client, voice_id, text):
    empty = {"audio": None, "mouth_cues": [], "lip_sync_mode": "silent", "voice_error": None}
    if client is None:
        return empty
    try:
        response = client.text_to_speech.convert_with_timestamps(
            voice_id=voice_id, text=text,
            model_id=os.environ.get("ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5"),
            output_format="pcm_16000",
        )
        pcm = base64.b64decode(field(response, "audio_base_64"), validate=True)
        wav = pcm_wave(pcm)
        duration = len(pcm) / 32000
        cues = []
        try:
            cues = rhubarb_cues(wav, text, duration)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
            print(f"[lip recognition unavailable] {type(error).__name__}")
        mode = "rhubarb" if cues else "aligned-text"
        if not cues:
            cues = aligned_cues(field(response, "normalized_alignment") or field(response, "alignment"), duration)
        return {
            "audio": "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii"),
            "mouth_cues": cues,
            "lip_sync_mode": mode if cues else "audio-energy",
            "voice_error": None,
        }
    except Exception as error:
        # Preserve text chat, without leaking provider responses or API keys.
        print(f"[voice unavailable] {type(error).__name__}")
        return {**empty, "voice_error": "Voice generation failed. Check the server's ElevenLabs configuration or quota."}
