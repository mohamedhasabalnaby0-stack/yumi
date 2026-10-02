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

import json
import os
import random
import re
import threading
import time
from pathlib import Path

from memory_store import MemoryStore, atomic_json, read_history
from speech import synthesize

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

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

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")

DATA_DIR = Path(os.environ.get("YUMI_DATA_DIR", str(BASE_DIR / ".data")))
MEMORY_FILE = DATA_DIR / "memory.json"
LONG_TERM_FILE = DATA_DIR / "long_term.json"
FAISS_INDEX_FILE = DATA_DIR / "long_term.index"

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
    ElevenLabs(api_key=ELEVENLABS_API_KEY, timeout=45)
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
    source = MEMORY_FILE if MEMORY_FILE.exists() else BASE_DIR / "memory.json"
    return read_history(source, MAX_HISTORY_MESSAGES)


def save_memory(history):
    try:
        atomic_json(MEMORY_FILE, history[-MAX_HISTORY_MESSAGES:])
    except OSError as e:
        print(f"[memory save error] {e}")


# ============================================================
# LONG-TERM FAISS MEMORY
# ============================================================

EMBED_DIM = 768


class LongTermMemory(MemoryStore):
    def __init__(self, client):
        def embed(text):
            result = call_gemini_with_retry(
                lambda: client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
                )
            )
            return result.embeddings[0].values

        super().__init__(DATA_DIR, embed, EMBED_DIM, legacy_dir=BASE_DIR)


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

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("The model returned an empty response")
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

    if not isinstance(data, dict):
        raise ValueError("The model response must be an object")
    reply = data.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("The model response has no reply")
    reply = reply.strip()

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

def synthesize_speech(text: str) -> dict:
    return synthesize(_eleven_client, ELEVENLABS_VOICE_ID, text)


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


# Same-origin local companion: do not expose private chat to arbitrary websites.
_client = None
_long_term = None
_history: list = []

_state_lock = threading.Lock()
_chat_lock = threading.Lock()
_completed_turns = 0


def get_client():

    global _client
    global _long_term
    global _history

    if _client is None:

        if not API_KEY:
            raise HTTPException(503, "GEMINI_API_KEY is not configured on the server.")

        _client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=30000),
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
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def nonempty(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be empty")
        return value


# ============================================================
# CHAT (single-user, single-worker; overlapping turns are rejected)
# ============================================================

@app.post("/chat")
def chat(body: ChatIn, background_tasks: BackgroundTasks):
    if not _chat_lock.acquire(blocking=False):
        raise HTTPException(409, "Yumi is already processing a message. Please wait.")
    try:
        return chat_turn(body, background_tasks)
    finally:
        _chat_lock.release()


def chat_turn(body: ChatIn, background_tasks: BackgroundTasks):
    client = get_client()
    global _history, _completed_turns

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

        # Build a request snapshot; commit both sides only after a valid reply.
        user_entry = {"role": "user", "parts": [{"text": user_text}]}
        current_history = _history[-(MAX_HISTORY_MESSAGES - 2):] + [user_entry]

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
        parsed = parse_structured_reply(response.text)

    except genai_errors.ClientError as e:

        print(f"[chat] Gemini client error: {getattr(e, 'code', 'unknown')}")

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

        print(f"[chat] Gemini server error: {getattr(e, 'code', 'unknown')}")

        return {
            "reply": random.choice(
                OVERLOAD_FALLBACK_REPLIES
            ),
            "emotion": "sad",
            "gesture": "none",
            "audio": None,
        }

    except Exception as e:

        print(f"[chat] Unexpected Gemini error: {type(e).__name__}")

        return {
            "reply": "Something went weird on my side for a second.",
            "emotion": "sad",
            "gesture": "none",
            "audio": None,
        }

    # --------------------------------------------------------
    # PARSE GEMINI RESPONSE
    # --------------------------------------------------------

    reply = parsed["reply"]
    emotion = parsed["emotion"]
    gesture = parsed["gesture"]

    # --------------------------------------------------------
    # SAVE SHORT-TERM HISTORY
    # --------------------------------------------------------

    with _state_lock:
        _history.append(user_entry)
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
    _completed_turns += 1

    if _completed_turns % MEMORY_EXTRACTION_EVERY == 0:

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
        **audio,
    }


# ============================================================
# STATIC FRONTEND
# ============================================================

@app.get("/health")
def health():
    import shutil
    return {
        "status": "ok",
        "chat_configured": bool(API_KEY),
        "voice_configured": VOICE_ENABLED,
        "model_present": (BASE_DIR / "static" / "yumi.vrm").is_file(),
        "rhubarb_available": bool(shutil.which(os.environ.get("RHUBARB_PATH", "rhubarb"))),
        "session_mode": "single-user; run one server worker",
    }


@app.get("/static/{asset_path:path}")
def static_asset(asset_path: str):
    # Never serve conversation JSON or dotfiles from the legacy static folder.
    root = (BASE_DIR / "static").resolve()
    path = (root / asset_path).resolve()
    allowed = {".js", ".mjs", ".css", ".vrm", ".vrma", ".fbx", ".glb", ".png", ".jpg"}
    if any(part.startswith(".") for part in Path(asset_path).parts) or root not in path.parents or path.suffix.lower() not in allowed or not path.is_file():
        raise HTTPException(404, "Asset not found")
    media_type = "text/javascript" if path.suffix in {".js", ".mjs"} else None
    return FileResponse(path, media_type=media_type)


@app.get("/")
def root():

    return FileResponse(
        str(
            BASE_DIR
            / "static"
            / "index.html"
        )
    )