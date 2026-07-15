from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from vectorstore.faiss_store import KnowledgeBase


class CombinedRetriever:
    def __init__(self, youtube_store: KnowledgeBase | None = None, document_store: KnowledgeBase | None = None) -> None:
        self.youtube_store = youtube_store
        self.document_store = document_store

    def retrieve(self, question: str, *, source: str) -> list[Document]:
        if source in {"youtube", "both"} and self.youtube_store is not None:
            youtube_docs = self.youtube_store.as_retriever(k=3).invoke(question)
        else:
            youtube_docs = []

        if source in {"documents", "both"} and self.document_store is not None:
            document_docs = self.document_store.as_retriever(k=3).invoke(question)
        else:
            document_docs = []

        combined = youtube_docs + document_docs
        deduped: list[Document] = []
        seen: set[tuple[Any, ...]] = set()
        for doc in combined:
            key = (doc.page_content, tuple(sorted(doc.metadata.items())))
            if key not in seen:
                deduped.append(doc)
                seen.add(key)
        return deduped[:6]
