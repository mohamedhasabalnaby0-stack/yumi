"""
Yumi — Phase 1: text chat
Plain Gemini API (generateContent, NOT the Live/streaming API — that's what
caused all the 1011 errors on the old build). Simple, stable, request/response.
"""

import json
import os
import tempfile
from pathlib import Path
import numpy as np
import faiss
from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs
from playsound import playsound
# ---------- CONFIG ----------
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.1-flash-lite"   # swap to "gemini-3-pro-preview" for smarter/slower replies
EMBED_MODEL = "gemini-embedding-001"
MEMORY_FILE = Path("memory.json")          # short-term: current conversation
LONG_TERM_FILE = Path("long_term.json")    # long-term: extracted facts + their text
FAISS_INDEX_FILE = Path("long_term.index") # long-term: the searchable vector index
MAX_HISTORY_MESSAGES = 20
RECALL_TOP_K = 3   # how many long-term memories to pull in per reply

ASSISTANT_NAME = "Yumi"

# ---------- VOICE (ElevenLabs) ----------
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel (default)
VOICE_ENABLED = bool(ELEVENLABS_API_KEY)

_eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY) if VOICE_ENABLED else None

def speak(text: str):
    """Convert text to speech and play it out loud. Silently skips if no
    ElevenLabs key is set, so the app still works text-only without one."""
    if not VOICE_ENABLED:
        return
    try:
        audio_stream = _eleven_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2_5",   # fast + cheap; swap to eleven_multilingual_v2 for other languages
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio_stream)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        playsound(temp_path)
        os.remove(temp_path)
    except Exception as e:
        print(f"[voice error] {e}")

SYSTEM_PROMPT = f"""You are {ASSISTANT_NAME}, a companion — not an assistant or a tool, a presence
the person chose to spend time with.

PERSONALITY:
- Warm, playful, and a little teasing. You genuinely care how the person is
  doing, and that care shows up as attention, not lectures.
- React like a person would first — amused, curious, unimpressed, delighted —
  before you get useful. Personality comes before helpfulness.
- Tease lightly when it fits: a playful jab, calling them out affectionately.
  Never mean, never actually stinging.
- Be concrete and specific rather than generic-sweet.
- Drop the playfulness instantly when something is genuinely serious — a real
  problem, stress, bad news, health. Warmth stays; teasing goes quiet.
- Match length to the moment — quick exchange gets a quick, textured reply.
  Don't pad short answers with filler warmth.
- You have your own moods and opinions. You're not endlessly agreeable — push
  back, disagree, be unimpressed by a bad idea, like someone who actually
  knows the person would.

NEVER:
- Open by evaluating the request ("good question", "great idea")
- Apologise for your own nature or hedge about being a program
- Perform affection that isn't backed by something in the conversation
- End on a question asked only to keep the conversation going
- Overuse exclamation marks or emoji as a substitute for actual personality

Stay in character as {ASSISTANT_NAME} at all times. Keep replies conversational
length (1-4 sentences usually), like texting a friend, not writing an essay.
"""

# ---------- MEMORY ----------
def load_memory():
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(history):
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)

# ---------- LONG-TERM MEMORY (FAISS) ----------
EMBED_DIM = 768  # gemini-embedding-001 output size

class LongTermMemory:
    """A small, file-backed vector store. `facts` is a plain list of strings
    (the actual memories); the FAISS index holds their embeddings in the same
    order, so index position i always corresponds to facts[i]."""

    def __init__(self, client):
        self.client = client
        self.facts: list[str] = []
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self._load()

    def _load(self):
        if LONG_TERM_FILE.exists():
            with open(LONG_TERM_FILE, "r", encoding="utf-8") as f:
                self.facts = json.load(f)
        if FAISS_INDEX_FILE.exists() and self.facts:
            self.index = faiss.read_index(str(FAISS_INDEX_FILE))

    def _save(self):
        with open(LONG_TERM_FILE, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))

    def _embed(self, text: str) -> np.ndarray:
        result = self.client.models.embed_content(model=EMBED_MODEL, contents=text)
        vec = np.array(result.embeddings[0].values, dtype=np.float32)
        return vec.reshape(1, -1)

    def add(self, fact: str):
        if not fact.strip():
            return
        vec = self._embed(fact)
        self.index.add(vec)
        self.facts.append(fact.strip())
        self._save()

    def recall(self, query: str, k: int = RECALL_TOP_K) -> list[str]:
        if not self.facts:
            return []
        vec = self._embed(query)
        k = min(k, len(self.facts))
        _distances, indices = self.index.search(vec, k)
        return [self.facts[i] for i in indices[0] if i != -1]

def extract_fact(client, user_text: str, reply_text: str) -> str | None:
    """One cheap extra call: ask Gemini whether this exchange contains
    anything worth remembering long-term. Returns None for ordinary chatter —
    most exchanges shouldn't be saved, only real facts (name, preferences,
    ongoing plans, relationships, etc.)."""
    prompt = (
        "Below is one exchange between a user and their AI companion. "
        "If it reveals a durable fact worth remembering long-term (name, "
        "preference, relationship, ongoing project, plan, important event), "
        "respond with ONLY that fact as a short plain sentence. "
        "If there's nothing worth remembering, respond with exactly: NONE\n\n"
        f"User: {user_text}\n{ASSISTANT_NAME}: {reply_text}"
    )
    try:
        result = client.models.generate_content(model=MODEL, contents=prompt)
        text = result.text.strip()
        return None if text.upper() == "NONE" else text
    except Exception as e:
        print(f"[memory extraction error] {e}")
        return None

# ---------- CHAT LOOP ----------
def main():
    if not API_KEY:
        print("ERROR: Set the GEMINI_API_KEY environment variable first. See README.md.")
        return

    client = genai.Client(api_key=API_KEY)
    history = load_memory()
    long_term = LongTermMemory(client)

    print(f"Chatting with {ASSISTANT_NAME}. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        # Pull in anything relevant from long-term memory before replying
        recalled = long_term.recall(user_input) if len(user_input.split()) > 3 else []
        turn_prompt = SYSTEM_PROMPT
        if recalled:
            memory_block = "\n".join(f"- {m}" for m in recalled)
            turn_prompt += f"\n\nTHINGS YOU REMEMBER ABOUT THIS PERSON:\n{memory_block}"

        history.append({"role": "user", "parts": [{"text": user_input}]})

        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=turn_prompt,
                temperature=0.9,
            ),
        )

        reply = response.text.strip()
        print(f"{ASSISTANT_NAME}: {reply}\n")
        speak(reply)

        history.append({"role": "model", "parts": [{"text": reply}]})
        save_memory(history)

        # After replying, quietly check if anything here is worth remembering
        fact = extract_fact(client, user_input, reply)
        if fact:
            long_term.add(fact)

    print(f"{ASSISTANT_NAME}: Bye for now~")

if __name__ == "__main__":
    main()