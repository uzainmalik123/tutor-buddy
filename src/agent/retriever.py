from pathlib import Path
import json
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class CurriculumRetriever:
    """Loads curriculum notes, embeds them with sentence-transformers, and
    retrieves relevant notes using FAISS similarity search."""

    def __init__(self, notes_path: str | Path = "data/curriculum/notes.json",
                 index_path: str | Path = "data/faiss_index.bin"):
        self.notes_path = Path(notes_path)
        self.index_path = Path(index_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.notes: list[dict] = []
        self.index: faiss.Index | None = None

        self._load_or_build_index()

    def _load_notes(self) -> list[dict]:
        """Load notes from the JSON file."""
        with open(self.notes_path) as f:
            return json.load(f)

    def _build_index(self) -> faiss.Index:
        """Embed all notes and build a FAISS index."""
        self.notes = self._load_notes()
        texts = [note["content"] for note in self.notes]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype(np.float32))
        return index

    def _save_index(self):
        """Save the FAISS index and notes to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        meta_path = self.index_path.with_suffix(".pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(self.notes, f)

    def _load_index(self) -> bool:
        """Load a saved FAISS index and notes from disk. Returns True on success."""
        meta_path = self.index_path.with_suffix(".pkl")
        if self.index_path.exists() and meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(meta_path, "rb") as f:
                    self.notes = pickle.load(f)
                return True
            except Exception:
                return False
        return False

    def _load_or_build_index(self):
        """Load a cached index or build and save a new one."""
        if not self._load_index():
            self.index = self._build_index()
            self._save_index()

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Return the top-k most relevant notes as a single plain-text string."""
        query_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for i in indices[0]:
            note = self.notes[i]
            results.append(
                f"Topic: {note['topic']}\nGrade: {note['grade']}\n{note['content']}"
            )
        return "\n\n".join(results)
