from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from processing.chunking import chunk_documents


class KnowledgeBase:
    def __init__(self, persist_dir: str | os.PathLike[str], embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store: FAISS | None = None

    def reset(self) -> None:
        self.vector_store = None
        for path in [self.persist_dir / "index.faiss", self.persist_dir / "index.pkl"]:
            if path.exists():
                path.unlink()
        for path in self.persist_dir.glob("*.faiss"):
            path.unlink()
        for path in self.persist_dir.glob("*.pkl"):
            path.unlink()

    def _load_existing(self) -> FAISS | None:
        if (self.persist_dir / "index.faiss").exists() and (self.persist_dir / "index.pkl").exists():
            return FAISS.load_local(str(self.persist_dir), embeddings=self.embedding_model, allow_dangerous_deserialization=True)
        return None

    def add_documents(self, documents: list[Document], *, source: str) -> FAISS:
        chunks = chunk_documents(documents)
        if not chunks:
            raise ValueError("No document chunks were produced.")
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            metadata.setdefault("source", source)
            chunk.metadata = metadata

        existing = self._load_existing() if self.vector_store is None else None
        if self.vector_store is None:
            if existing is not None:
                self.vector_store = existing
            else:
                self.vector_store = FAISS.from_documents(documents=chunks, embedding=self.embedding_model)
        else:
            self.vector_store.add_documents(chunks)

        self.vector_store.save_local(str(self.persist_dir))
        return self.vector_store

    def as_retriever(self, k: int = 4) -> Any:
        if self.vector_store is None:
            self.vector_store = self._load_existing()
        if self.vector_store is None:
            raise ValueError("No vector store has been built yet.")
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k})
