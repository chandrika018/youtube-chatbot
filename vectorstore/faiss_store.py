from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from processing.chunking import chunk_documents


class KnowledgeBase:
    def __init__(self, persist_dir: str | os.PathLike[str], embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store: FAISS | None = None
        self._documents: list[Document] = []

    def reset(self) -> None:
        self.vector_store = None
        # Pehle ke redundant loop ko hata kar ek single optimized loop use kiya hai
        for path in self.persist_dir.glob("*"):
            if path.suffix in {".faiss", ".pkl"}:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def _load_existing(self) -> FAISS | None:
        if (self.persist_dir / "index.faiss").exists() and (self.persist_dir / "index.pkl").exists():
            try:
                return FAISS.load_local(
                    str(self.persist_dir), 
                    embeddings=self.embedding_model, 
                    allow_dangerous_deserialization=True
                )
            except Exception:
                return None
        return None

    def add_documents(self, documents: list[Document], *, source: str) -> FAISS:
        chunks = chunk_documents(documents, chunk_size=450, chunk_overlap=80)
        if not chunks:
            raise ValueError("No document chunks were produced.")
        
        # Metadata dictionary lookup ko fast kiya gaya hai
        for chunk in chunks:
            metadata = chunk.metadata or {}
            metadata.setdefault("source", source)
            metadata.setdefault("chunk_id", 0)
            chunk.metadata = metadata
            
        self._documents.extend(chunks)

        if self.vector_store is None:
            existing = self._load_existing()
            self.vector_store = existing if existing is not None else FAISS.from_documents(documents=chunks, embedding=self.embedding_model)
        else:
            self.vector_store.add_documents(chunks)

        self.vector_store.save_local(str(self.persist_dir))
        return self.vector_store

    def as_retriever(self, k: int = 4, **kwargs: Any) -> Any:
        if self.vector_store is None:
            self.vector_store = self._load_existing()
            
        if self.vector_store is None:
            return HybridRetriever(self._documents, k=k)
            
        search_kwargs = {"k": k, "lambda_mult": 0.5}
        search_kwargs.update(kwargs)
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)


class HybridRetriever:
    # Regex pattern ko baar-baar compile hone se bachane ke liye class-level par define kiya hai
    _TERM_PATTERN = re.compile(r"[a-z0-9]+")

    def __init__(self, documents: list[Document], *, k: int = 4) -> None:
        self.documents = documents
        self.k = k

    def invoke(self, question: str) -> list[Document]:
        if not self.documents:
            return []

        # Pre-compiled regex pattern use kiya hai speed ke liye
        query_terms = {term for term in self._TERM_PATTERN.findall(question.lower()) if len(term) > 2}
        
        if not query_terms:
            return self.documents[: self.k]

        scored: list[tuple[float, Document]] = []
        for doc in self.documents:
            content = doc.page_content.lower()
            # Loop inline sum() generator ki jagah list comprehension ya generator ko optimize kiya
            overlap = sum(term in content for term in query_terms)
            score = float(overlap) if overlap > 0 else -1.0
            scored.append((score, doc))

        # Pure list ko copy karne ki jagah sorting direct execute hogi
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[: self.k]]