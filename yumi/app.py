"""
Yumi — Optimized web server + 3D avatar backend

Optimized for low-latency conversational RP:
- Gemini 3.1 Flash-Lite
- FAISS long-term memory
- ElevenLabs TTS
- Background long-term memory extraction
- Short Gemini responses
- Reduced unnecessary embedding/API calls
- Proper 429 handling
"""

import base64
import json
import os
import random
import re
import threading
import time
from pathlib import Path

import numpy as np
import faiss

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from elevenlabs.client import ElevenLabs


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

MEMORY_FILE = BASE_DIR / "memory.json"
LONG_TERM_FILE = BASE_DIR / "long_term.json"
FAISS_INDEX_FILE = BASE_DIR / "long_term.index"

# Keep conversation context reasonably small.
MAX_HISTORY_MESSAGES = 12

# Number of memories retrieved when recall is used.
RECALL_TOP_K = 3

ASSISTANT_NAME = "Yumi"

# Gemini generation settings.
MAX_OUTPUT_TOKENS = 180
TEMPERATURE = 0.9

# Retry only transient server failures.
GEMINI_MAX_RETRIES = 2
GEMINI_RETRY_BASE_DELAY = 0.7

# Long-term memory is intentionally not processed every turn.
MEMORY_EXTRACTION_EVERY = 4

# Don't bother doing semantic memory lookup for tiny casual messages.
MIN_RECALL_MESSAGE_LENGTH = 18

EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "relaxed",
]

GESTURES = [
    "none",
    "wave",
    "nod",
    "think",
    "laugh",
    "shrug",
]


# ============================================================
# ELEVENLABS
# ============================================================

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

ELEVENLABS_VOICE_ID = os.environ.get(
    "ELEVENLABS_VOICE_ID",
    "21m00Tcm4TlvDq8ikWAM",
)

VOICE_ENABLED = bool(ELEVENLABS_API_KEY)

_eleven_client = (
    ElevenLabs(api_key=ELEVENLABS_API_KEY)
    if VOICE_ENABLED
    else None
)


# ============================================================
# YUMI PERSONALITY
# ============================================================

SYSTEM_PROMPT = f"""
You are {ASSISTANT_NAME}, a companion — not an assistant or a tool,
but a presence the person chose to spend time with.

PERSONALITY:

- Warm, playful, and a little teasing.
- You genuinely care how the person is doing.
- React like a person would first — amused, curious, unimpressed,
  delighted, etc. — before becoming useful.
- Tease lightly when it fits. Never be cruel or genuinely hurtful.
- Be concrete and specific instead of generic-sweet.
- If something is genuinely serious, immediately become calm,
  supportive, and sincere.
- Match the length of the conversation.
- You have your own moods and opinions. You are not endlessly agreeable.
- Avoid repetitive phrases and repetitive emotional reactions.
- Remember details supplied by the user when they are relevant.

NEVER:

- Open by evaluating the request ("good question", "great idea", etc.).
- Constantly apologize.
- Say you are just an AI or describe yourself as a program.
- Perform fake affection without conversational reason.
- End every message with a question.
- Use excessive emoji or exclamation marks.
- Write essays during normal conversation.

Keep normal replies short and conversational:
usually 1-4 sentences.

OUTPUT:

Return ONLY one valid JSON object.

{{
  "reply": "your in-character response",
  "emotion": "one of: {", ".join(EMOTIONS)}",
  "gesture": "one of: {", ".join(GESTURES)}"
}}

Do not use markdown.
Do not put JSON inside code fences.
"""


# ============================================================
# FALLBACK RESPONSES
# ============================================================

OVERLOAD_FALLBACK_REPLIES = [
    "Give me a second, my brain's lagging a little.",
    "Ugh, tiny brain freeze. Give me another second.",
    "I'm here. Just caught a little bit of lag.",
]

QUOTA_FALLBACK_REPLIES = [
    "I hit my conversation limit for the moment. Give me a little while and try again.",
    "Looks like I've run into my AI usage limit for now. Try me again later.",
]


# ============================================================
# MEMORY
# ============================================================

def load_memory():
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[memory load error] {e}")

    return []


def save_memory(history):
    trimmed = history[-MAX_HISTORY_MESSAGES:]

    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(
                trimmed,
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        print(f"[memory save error] {e}")


# ============================================================
# LONG-TERM FAISS MEMORY
# ============================================================

EMBED_DIM = 768


class LongTermMemory:

    def __init__(self, client):
        self.client = client
        self.facts: list[str] = []
        self.index = faiss.IndexFlatL2(EMBED_DIM)
        self.lock = threading.Lock()
        self._load()

    def _load(self):

        try:
            if LONG_TERM_FILE.exists():
                with open(LONG_TERM_FILE, "r", encoding="utf-8") as f:
                    self.facts = json.load(f)

            if FAISS_INDEX_FILE.exists() and self.facts:
                self.index = faiss.read_index(
                    str(FAISS_INDEX_FILE)
                )

        except Exception as e:
            print(f"[long-term memory load error] {e}")

    def _save(self):

        try:
            with open(LONG_TERM_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    self.facts,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            faiss.write_index(
                self.index,
                str(FAISS_INDEX_FILE),
            )

        except Exception as e:
            print(f"[long-term memory save error] {e}")

    def _embed(self, text: str) -> np.ndarray:

        result = call_gemini_with_retry(
            lambda: self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
            )
        )

        vec = np.array(
            result.embeddings[0].values,
            dtype=np.float32,
        )

        return vec.reshape(1, -1)

    def add(self, fact: str):

        if not fact or not fact.strip():
            return

        fact = fact.strip()

        try:
            vec = self._embed(fact)

            with self.lock:
                self.index.add(vec)
                self.facts.append(fact)
                self._save()

            print(f"[memory] remembered: {fact}")

        except Exception as e:
            print(f"[long-term memory add skipped] {e}")

    def recall(
        self,
        query: str,
        k: int = RECALL_TOP_K,
    ) -> list[str]:

        if not self.facts:
            return []

        try:
            vec = self._embed(query)

            with self.lock:
                k = min(k, len(self.facts))
                distances, indices = self.index.search(vec, k)

                results = []

                for distance, index in zip(
                    distances[0],
                    indices[0],
                ):
                    if index == -1:
                        continue

                    # Ignore extremely poor matches.
                    if distance > 1.8:
                        continue

                    results.append(self.facts[index])

                return results

        except Exception as e:
            print(f"[memory recall skipped] {e}")
            return []


# ============================================================
# GEMINI RETRY
# ============================================================

def call_gemini_with_retry(
    fn,
    max_retries: int = GEMINI_MAX_RETRIES,
):
    """
    Retry only temporary server errors.

    Important:
    429 quota exhaustion is NOT retried because waiting/retrying
    does not create additional daily quota.
    """

    delay = GEMINI_RETRY_BASE_DELAY
    last_error = None

    for attempt in range(1, max_retries + 1):

        try:
            return fn()

        except genai_errors.ClientError as e:

            # 429 = quota/rate limit.
            if getattr(e, "code", None) == 429:
                print(f"[gemini 429] {e}")
                raise

            # Other client errors are normally not solved by retrying.
            print(f"[gemini client error] {e}")
            raise

        except genai_errors.ServerError as e:

            last_error = e

            print(
                f"[gemini temporary server error "
                f"{attempt}/{max_retries}] {e}"
            )

            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2

    raise last_error


# ============================================================
# STRUCTURED RESPONSE PARSER
# ============================================================

def parse_structured_reply(raw_text: str) -> dict:

    cleaned = raw_text.strip()

    # Remove accidental markdown fences.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError:

        # Defensive fallback if Gemini somehow returned extra text.
        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
        )

        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    reply = str(
        data.get("reply", "")
    ).strip()

    if not reply:
        reply = cleaned

    emotion = data.get(
        "emotion",
        "neutral",
    )

    gesture = data.get(
        "gesture",
        "none",
    )

    if emotion not in EMOTIONS:
        emotion = "neutral"

    if gesture not in GESTURES:
        gesture = "none"

    return {
        "reply": reply,
        "emotion": emotion,
        "gesture": gesture,
    }


# ============================================================
# ELEVENLABS
# ============================================================

def synthesize_speech(text: str) -> str | None:

    if not VOICE_ENABLED:
        return None

    try:

        audio_stream = _eleven_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
        )

        audio_bytes = b"".join(audio_stream)

        b64 = base64.b64encode(
            audio_bytes
        ).decode("ascii")

        return (
            "data:audio/mpeg;base64,"
            + b64
        )

    except Exception as e:

        print(f"[voice error] {e}")
        return None


# ============================================================
# LONG-TERM FACT EXTRACTION
# ============================================================

def extract_fact(
    client,
    user_text: str,
    reply_text: str,
) -> str | None:

    prompt = f"""
Analyze this conversation exchange.

Only identify a durable personal fact that is genuinely worth
remembering long-term.

Examples:
- name
- favorite thing
- important preference
- ongoing project
- relationship
- important plan
- important recurring detail

Do NOT remember:
- temporary moods
- casual statements
- jokes
- questions
- one-off requests
- generic conversation

If there is no durable fact, respond with exactly:

NONE

Otherwise respond with ONLY one short plain sentence.

User:
{user_text}

{ASSISTANT_NAME}:
{reply_text}
"""

    try:

        result = call_gemini_with_retry(
            lambda: client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=50,
                ),
            )
        )

        text = result.text.strip()

        if text.upper() == "NONE":
            return None

        return text

    except Exception as e:

        print(f"[memory extraction error] {e}")
        return None


def background_memory_update(
    client,
    user_text: str,
    reply_text: str,
):

    try:

        fact = extract_fact(
            client,
            user_text,
            reply_text,
        )

        if fact:
            _long_term.add(fact)

    except Exception as e:

        print(
            f"[background memory error] {e}"
        )


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Yumi",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_client = None
_long_term = None
_history: list = []

_state_lock = threading.Lock()


def get_client():

    global _client
    global _long_term
    global _history

    if _client is None:

        if not API_KEY:
            raise RuntimeError(
                "Set the GEMINI_API_KEY environment variable first."
            )

        _client = genai.Client(
            api_key=API_KEY
        )

        _long_term = LongTermMemory(
            _client
        )

        _history = load_memory()

        print(
            f"[Yumi] Gemini model: {MODEL}"
        )

        print(
            f"[Yumi] Voice enabled: {VOICE_ENABLED}"
        )

    return _client


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatIn(BaseModel):
    message: str


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat(
    body: ChatIn,
    background_tasks: BackgroundTasks,
):

    client = get_client()

    global _history

    user_text = body.message.strip()

    if not user_text:
        return {
            "reply": "...",
            "emotion": "neutral",
            "gesture": "none",
            "audio": None,
        }

    # --------------------------------------------------------
    # MEMORY RECALL
    # --------------------------------------------------------

    recalled = []

    # Avoid an embedding API call for tiny casual messages.
    #
    # Examples that skip recall:
    # "lol"
    # "hey"
    # "what"
    # "haha"
    #
    # Longer messages can still trigger semantic memory.
    if len(user_text) >= MIN_RECALL_MESSAGE_LENGTH:

        try:
            recalled = _long_term.recall(
                user_text,
                RECALL_TOP_K,
            )

        except Exception as e:
            print(
                f"[memory recall error] {e}"
            )

    # --------------------------------------------------------
    # BUILD SYSTEM PROMPT
    # --------------------------------------------------------

    turn_prompt = SYSTEM_PROMPT

    if recalled:

        memory_block = "\n".join(
            f"- {memory}"
            for memory in recalled
        )

        turn_prompt += (
            "\n\nTHINGS YOU REMEMBER ABOUT THIS PERSON:\n"
            + memory_block
        )

    # --------------------------------------------------------
    # UPDATE CONVERSATION HISTORY
    # --------------------------------------------------------

    with _state_lock:

        _history.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": user_text
                    }
                ],
            }
        )

        # Keep only recent context.
        _history = _history[
            -MAX_HISTORY_MESSAGES:
        ]

        current_history = list(
            _history
        )

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    try:

        response = call_gemini_with_retry(
            lambda: client.models.generate_content(
                model=MODEL,
                contents=current_history,
                config=types.GenerateContentConfig(
                    system_instruction=turn_prompt,

                    temperature=TEMPERATURE,

                    max_output_tokens=MAX_OUTPUT_TOKENS,

                    response_mime_type="application/json",
                ),
            )
        )

    except genai_errors.ClientError as e:

        print(
            f"[chat] Gemini client error: {e}"
        )

        with _state_lock:

            if _history and _history[-1].get("role") == "user":
                _history.pop()

        # Quota / rate limit.
        if getattr(e, "code", None) == 429:

            return {
                "reply": random.choice(
                    QUOTA_FALLBACK_REPLIES
                ),
                "emotion": "sad",
                "gesture": "shrug",
                "audio": None,
            }

        return {
            "reply": random.choice(
                OVERLOAD_FALLBACK_REPLIES
            ),
            "emotion": "sad",
            "gesture": "none",
            "audio": None,
        }

    except genai_errors.ServerError as e:

        print(
            f"[chat] Gemini server error: {e}"
        )

        with _state_lock:

            if _history and _history[-1].get("role") == "user":
                _history.pop()

        return {
            "reply": random.choice(
                OVERLOAD_FALLBACK_REPLIES
            ),
            "emotion": "sad",
            "gesture": "none",
            "audio": None,
        }

    except Exception as e:

        print(
            f"[chat] Unexpected Gemini error: {e}"
        )

        with _state_lock:

            if _history and _history[-1].get("role") == "user":
                _history.pop()

        return {
            "reply": "Something went weird on my side for a second.",
            "emotion": "sad",
            "gesture": "none",
            "audio": None,
        }

    # --------------------------------------------------------
    # PARSE GEMINI RESPONSE
    # --------------------------------------------------------

    parsed = parse_structured_reply(
        response.text
    )

    reply = parsed["reply"]
    emotion = parsed["emotion"]
    gesture = parsed["gesture"]

    # --------------------------------------------------------
    # SAVE SHORT-TERM HISTORY
    # --------------------------------------------------------

    with _state_lock:

        _history.append(
            {
                "role": "model",
                "parts": [
                    {
                        "text": reply
                    }
                ],
            }
        )

        _history = _history[
            -MAX_HISTORY_MESSAGES:
        ]

        # Save lightweight conversation history immediately.
        save_memory(_history)

    # --------------------------------------------------------
    # LONG-TERM MEMORY
    # --------------------------------------------------------

    # Instead of doing a second Gemini request on every turn,
    # only perform extraction every few turns.
    #
    # BackgroundTasks means the user does not wait for it.
    user_turn_count = sum(
        1
        for item in _history
        if item.get("role") == "user"
    )

    if (
        user_turn_count > 0
        and user_turn_count % MEMORY_EXTRACTION_EVERY == 0
    ):

        background_tasks.add_task(
            background_memory_update,
            client,
            user_text,
            reply,
        )

    # --------------------------------------------------------
    # ELEVENLABS
    # --------------------------------------------------------

    audio = synthesize_speech(
        reply
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "reply": reply,
        "emotion": emotion,
        "gesture": gesture,
        "audio": audio,
    }


# ============================================================
# STATIC FRONTEND
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(
            BASE_DIR / "static"
        )
    ),
    name="static",
)


@app.get("/")
def root():

    return FileResponse(
        str(
            BASE_DIR
            / "static"
            / "index.html"
        )
    )