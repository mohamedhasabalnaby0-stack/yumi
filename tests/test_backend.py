import base64
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Tests never call paid services or use developer credentials.
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("ELEVENLABS_API_KEY", None)
with patch("dotenv.load_dotenv"):
    import app
from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient
from memory_store import MemoryStore, atomic_json, read_history
from speech import aligned_cues, pcm_wave, synthesize


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_atomic_history_restores_only_complete_pairs(self):
        path = self.root / "private" / "memory.json"
        entries = [
            {"role": "user", "parts": [{"text": "hello"}]},
            {"role": "model", "parts": [{"text": "hi"}]},
            {"role": "user", "parts": [{"text": "interrupted"}]},
        ]
        atomic_json(path, entries)
        self.assertEqual(read_history(path), entries[:2])
        self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_normalizes_persists_and_deduplicates(self):
        embed = Mock(return_value=[3.0, 4.0])
        memory = MemoryStore(self.root, embed, 2)
        memory.add("A durable fact")
        memory.add("A durable fact")
        self.assertEqual(embed.call_count, 1)
        self.assertAlmostEqual(sum(x * x for x in memory.vectors[0]), 1.0, places=5)
        reloaded = MemoryStore(self.root, embed, 2)
        self.assertEqual(reloaded.recall("query"), ["A durable fact"])
        self.assertEqual(reloaded.index.ntotal, len(reloaded.facts))

    def test_wrong_dimensions_do_not_corrupt_index(self):
        memory = MemoryStore(self.root, lambda text: [1, 2, 3], 2)
        memory.add("bad dimension")
        self.assertEqual(memory.facts, [])
        self.assertEqual(memory.index.ntotal, 0)

    def test_migrates_legacy_facts_without_old_faiss_file(self):
        legacy = self.root / "legacy"
        atomic_json(legacy / "long_term.json", ["fact one", "fact two"])
        memory = MemoryStore(self.root / "new", lambda text: [1, 0], 2, legacy)
        self.assertEqual(len(memory.pending), 2)
        self.assertEqual(set(memory.recall("fact")), {"fact one", "fact two"})
        self.assertEqual(memory.pending, [])

    def test_invalid_stored_vector_is_reembedded(self):
        atomic_json(self.root / "long_term.json", {"facts": ["fact"], "vectors": [[1, 2, 3]]})
        memory = MemoryStore(self.root, lambda text: [1, 0], 2)
        self.assertEqual(memory.recall("query"), ["fact"])

    def test_failed_write_can_be_retried_without_index_corruption(self):
        memory = MemoryStore(self.root, lambda text: [1, 0], 2)
        with patch("memory_store.atomic_json", side_effect=OSError("disk unavailable")):
            memory.add("fact")
        self.assertEqual(memory.index.ntotal, 0)
        self.assertEqual(memory.facts, [])
        memory.add("fact")
        self.assertEqual(memory.facts, ["fact"])

    def test_gemini_embedding_explicitly_requests_index_dimension(self):
        client = Mock()
        client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0] * app.EMBED_DIM)])
        with patch("app.DATA_DIR", self.root), patch("app.BASE_DIR", self.root):
            memory = app.LongTermMemory(client)
            memory.add("fact")
        config = client.models.embed_content.call_args.kwargs["config"]
        self.assertEqual(config.output_dimensionality, app.EMBED_DIM)
        self.assertEqual(memory.index.ntotal, 1)


class SpeechTests(unittest.TestCase):
    def test_installed_sdk_exposes_timestamp_endpoint(self):
        import inspect
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key="test-only-not-a-real-key", timeout=1)
        method = client.text_to_speech.convert_with_timestamps
        signature = inspect.signature(method)
        for parameter in ["voice_id", "text", "model_id", "output_format"]:
            self.assertIn(parameter, signature.parameters)

    def test_pcm_wav_header_and_duration(self):
        with wave.open(io.BytesIO(pcm_wave(b"\x00\x00" * 16000)), "rb") as wav:
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnframes(), 16000)
            self.assertEqual(wav.getnchannels(), 1)
        with self.assertRaises(ValueError):
            pcm_wave(b"x")

    def test_aligned_cues_close_lips_on_bilabials_and_spaces(self):
        alignment = {"characters": ["a", "m", " "], "character_start_times_seconds": [0, .2, .4], "character_end_times_seconds": [.2, .4, .6]}
        cues = aligned_cues(alignment, 1)
        self.assertGreater(cues[0]["weights"]["aa"], 0)
        self.assertEqual(cues[1]["weights"], {})
        self.assertEqual(cues[2]["weights"], {})

    def test_invalid_alignment_times_are_ignored(self):
        alignment = {"characters": ["a", "e"], "character_start_times_seconds": [float("nan"), .1], "character_end_times_seconds": [.1, .4]}
        self.assertEqual(len(aligned_cues(alignment, .3)), 1)
        self.assertEqual(aligned_cues(alignment, .3)[0]["end"], .3)
        self.assertEqual(aligned_cues({"characters": ["a"]}, 1), [])

    def test_optional_voice_and_provider_failure(self):
        self.assertIsNone(synthesize(None, "voice", "text")["audio"])
        client = Mock()
        client.text_to_speech.convert_with_timestamps.side_effect = RuntimeError("provider down")
        result = synthesize(client, "voice", "text")
        self.assertIsNone(result["audio"])
        self.assertTrue(result["voice_error"])

    def test_timestamp_fallback_is_labeled_not_phoneme_recognition(self):
        client = Mock()
        client.text_to_speech.convert_with_timestamps.return_value = SimpleNamespace(
            audio_base_64=base64.b64encode(b"\x00\x00" * 16000).decode(),
            normalized_alignment=None,
            alignment=SimpleNamespace(characters=["a"], character_start_times_seconds=[0], character_end_times_seconds=[.5]),
        )
        with patch("speech.rhubarb_cues", return_value=[]):
            result = synthesize(client, "voice", "a")
        self.assertEqual(result["lip_sync_mode"], "aligned-text")
        self.assertTrue(result["audio"].startswith("data:audio/wav;base64,"))
        self.assertEqual(len(result["mouth_cues"]), 1)

    def test_recognition_failure_preserves_audio(self):
        client = Mock()
        client.text_to_speech.convert_with_timestamps.return_value = {"audio_base_64": base64.b64encode(b"\x00\x00" * 100).decode()}
        with patch("speech.rhubarb_cues", side_effect=OSError("unavailable")):
            result = synthesize(client, "voice", "hi")
        self.assertTrue(result["audio"])
        self.assertEqual(result["lip_sync_mode"], "audio-energy")


class ChatTests(unittest.TestCase):
    def setUp(self):
        app._history = []
        app._completed_turns = 0
        self.client = Mock()
        self.client.models.generate_content.return_value = SimpleNamespace(text=json.dumps({"reply": "hello", "emotion": "happy", "gesture": "wave"}))
        for target, kwargs in [
            ("app.get_client", {"return_value": self.client}),
            ("app.save_memory", {}),
            ("app.synthesize_speech", {"return_value": {"audio": None, "mouth_cues": [], "lip_sync_mode": "silent", "voice_error": None}}),
        ]:
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_memory_extraction_continues_after_history_fills(self):
        tasks = BackgroundTasks()
        for _ in range(12):
            result = app.chat(app.ChatIn(message="hi"), tasks)
            self.assertEqual(result["reply"], "hello")
        self.assertEqual(app._completed_turns, 12)
        self.assertEqual(len(app._history), app.MAX_HISTORY_MESSAGES)
        self.assertEqual(len(tasks.tasks), 3)
        self.assertEqual([x["role"] for x in app._history], ["user", "model"] * 6)

    def test_failed_reply_does_not_modify_history(self):
        app.chat(app.ChatIn(message="hi"), BackgroundTasks())
        saved = list(app._history)
        self.client.models.generate_content.return_value = SimpleNamespace(text="[]")
        app.chat(app.ChatIn(message="hi"), BackgroundTasks())
        self.assertEqual(app._history, saved)
        self.assertEqual(app._completed_turns, 1)

    def test_concurrent_turn_is_rejected_and_lock_remains_held(self):
        with app._chat_lock:
            with self.assertRaises(HTTPException) as raised:
                app.chat(app.ChatIn(message="hi"), BackgroundTasks())
            self.assertEqual(raised.exception.status_code, 409)
            self.assertTrue(app._chat_lock.locked())

    def test_lock_released_on_exception(self):
        with patch("app.chat_turn", side_effect=RuntimeError("failure")):
            with self.assertRaises(RuntimeError):
                app.chat(app.ChatIn(message="hi"), BackgroundTasks())
        self.assertFalse(app._chat_lock.locked())

    def test_parser_rejects_nonobjects_and_invalid_reply(self):
        for raw in ["[]", "null", "", None, '{"reply": null}', '{"reply": 42}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                app.parse_structured_reply(raw)
        self.assertEqual(app.parse_structured_reply('```json\n{"reply":"hi","emotion":"unknown"}\n```')["emotion"], "neutral")


class HttpTests(unittest.TestCase):
    def test_routes_and_private_files(self):
        with TestClient(app.app) as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/static/avatar.mjs").status_code, 200)
            self.assertIn("javascript", client.get("/static/avatar.mjs").headers["content-type"])
            self.assertEqual(client.get("/static/memory.json").status_code, 404)
            self.assertEqual(client.get("/static/../app.py").status_code, 404)
            health = client.get("/health").json()
            self.assertTrue(health["model_present"])
            for message in ["", "   ", "x" * 2001]:
                self.assertEqual(client.post("/chat", json={"message": message}).status_code, 422)
            with patch("app.API_KEY", None), patch("app._client", None):
                self.assertEqual(client.post("/chat", json={"message": "hi"}).status_code, 503)

    def test_vrm_is_a_real_gltf_binary_with_humanoid_metadata(self):
        path = app.BASE_DIR / "static" / "yumi.vrm"
        with path.open("rb") as stream:
            magic, version, length = struct.unpack("<4sII", stream.read(12))
            self.assertEqual(magic, b"glTF")
            self.assertEqual(version, 2)
            self.assertEqual(length, path.stat().st_size)
            chunk_length, chunk_type = struct.unpack("<I4s", stream.read(8))
            self.assertEqual(chunk_type, b"JSON")
            data = json.loads(stream.read(chunk_length))
        extensions = data.get("extensions", {})
        self.assertTrue("VRM" in extensions or "VRMC_vrm" in extensions)


if __name__ == "__main__":
    unittest.main()
