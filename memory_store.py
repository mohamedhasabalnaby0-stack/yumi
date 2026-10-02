"""Private, atomic persistence; FAISS indexes are rebuilt from validated vectors."""
import json
import math
import os
from pathlib import Path
import tempfile
import threading

import faiss
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".memory-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_history(path, limit=12):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        # Only restore complete alternating turns, not interrupted requests.
        result = []
        pending = None
        for entry in data:
            if not isinstance(entry, dict):
                continue
            parts = entry.get("parts")
            if not isinstance(parts, list) or not parts or not all(
                isinstance(p, dict) and isinstance(p.get("text"), str) for p in parts
            ):
                continue
            clean = {"role": entry.get("role"), "parts": [{"text": p["text"]} for p in parts]}
            if clean["role"] == "user":
                pending = clean
            elif clean["role"] == "model" and pending:
                result.extend([pending, clean])
                pending = None
        return result[-(limit // 2 * 2):] if limit >= 2 else []
    except (OSError, ValueError):
        return []


class MemoryStore:
    def __init__(self, directory, embed, dimension=768, legacy_dir=None):
        self.path = Path(directory) / "long_term.json"
        self.embed = embed
        self.dimension = dimension
        self.lock = threading.Lock()
        self.facts = []
        self.vectors = []
        self.pending = []
        self.index = faiss.IndexFlatL2(dimension)
        source = self.path
        if not source.exists() and legacy_dir:
            source = Path(legacy_dir) / "long_term.json"
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.pending = list(dict.fromkeys(x.strip() for x in data if isinstance(x, str) and x.strip()))
            elif isinstance(data, dict):
                facts, vectors = data.get("facts", []), data.get("vectors", [])
                if not isinstance(facts, list) or not isinstance(vectors, list):
                    raise ValueError("Invalid memory format")
                for i, fact in enumerate(facts):
                    if not isinstance(fact, str) or not fact.strip() or fact in self.facts or fact in self.pending:
                        continue
                    try:
                        vector = self._vector(vectors[i])
                    except (IndexError, TypeError, ValueError):
                        self.pending.append(fact)
                        continue
                    self.index.add(vector)
                    self.facts.append(fact)
                    self.vectors.append(vector[0].tolist())
                self.pending.extend(x for x in data.get("pending", []) if isinstance(x, str) and x.strip() and x not in self.facts and x not in self.pending)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as error:
            print(f"[memory load skipped] {error}")

    def _vector(self, values):
        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all():
            raise ValueError("Embedding dimensions or values do not match the memory index")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError("Embedding has no finite magnitude")
        return (vector / norm).reshape(1, -1)

    def _save(self):
        # Facts and vectors share one atomic file: no mismatched index/fact pair.
        atomic_json(self.path, {"facts": self.facts, "vectors": self.vectors, "pending": self.pending})

    def add(self, fact):
        if not isinstance(fact, str) or not fact.strip():
            return
        fact = fact.strip()
        try:
            with self.lock:
                if fact in self.facts:
                    return
            vector = self._vector(self.embed(fact))
            with self.lock:
                if fact in self.facts:
                    return
                facts = self.facts + [fact]
                vectors = self.vectors + [vector[0].tolist()]
                pending = [item for item in self.pending if item != fact]
                atomic_json(self.path, {"facts": facts, "vectors": vectors, "pending": pending})
                self.index.add(vector)
                self.facts, self.vectors, self.pending = facts, vectors, pending
        except Exception as error:
            print(f"[memory add skipped] {error}")

    def recall(self, query, k=3):
        try:
            with self.lock:
                pending = self.pending[:2]
            # Migrate legacy facts gradually; never index with an incompatible vector size.
            for fact in pending:
                self.add(fact)
            with self.lock:
                if not self.facts or k <= 0:
                    return []
            vector = self._vector(self.embed(query))
            with self.lock:
                distances, indices = self.index.search(vector, min(k, len(self.facts)))
                return [self.facts[i] for d, i in zip(distances[0], indices[0]) if 0 <= i < len(self.facts) and d <= 1.8]
        except Exception as error:
            print(f"[memory recall skipped] {error}")
            return []
