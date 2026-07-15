from __future__ import annotations

import math
import re
from typing import Any

from langchain_core.documents import Document

from vectorstore.faiss_store import KnowledgeBase


class CombinedRetriever:
    def __init__(self, youtube_store: KnowledgeBase | None = None, document_store: KnowledgeBase | None = None) -> None:
        self.youtube_store = youtube_store
        self.document_store = document_store

    def retrieve(self, question: str, *, source: str) -> list[Document]:
        if source in {"youtube", "both"} and self.youtube_store is not None:
            youtube_docs = self.youtube_store.as_retriever(
                k=6,
                fetch_k=20,
                lambda_mult=0.7,
            ).invoke(question)
        else:
            youtube_docs = []

        if source in {"documents", "both"} and self.document_store is not None:
            document_docs = self.document_store.as_retriever(
                k=6,
                fetch_k=20,
                lambda_mult=0.7,
            ).invoke(question)
        else:
            document_docs = []

        combined = youtube_docs + document_docs
        scored: list[tuple[float, Document]] = []
        for doc in combined:
            score = self._score_document(question, doc)
            scored.append((score, doc))

        deduped: list[Document] = []
        seen: set[tuple[Any, ...]] = set()
        for _, doc in sorted(scored, key=lambda item: item[0], reverse=True):
            key = (doc.page_content, tuple(sorted(doc.metadata.items())))
            if key not in seen:
                deduped.append(doc)
                seen.add(key)
        return deduped[:6]

    def _score_document(self, question: str, doc: Document) -> float:
        question_terms = [term.lower() for term in re.findall(r"[a-z0-9]+", question.lower()) if len(term) > 2]
        content_terms = [term.lower() for term in re.findall(r"[a-z0-9]+", doc.page_content.lower()) if len(term) > 2]

        if not question_terms:
            return 0.0

        term_counts = {}
        for term in content_terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        doc_freq = 0
        for term in set(question_terms):
            if term in term_counts:
                doc_freq += 1

        bm25_score = 0.0
        for term in set(question_terms):
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log((1 + 1) / (1 + doc_freq)) + 1.0
            bm25_score += idf * ((tf * (1.5 + 1)) / (tf + 1.5 * (1 - 0.75 + 0.75 * 1)))

        overlap = len(set(question_terms) & set(content_terms))
        title_bonus = 0.3 if any(term in (doc.metadata or {}).get("title", "").lower() for term in set(question_terms)) else 0.0
        source_bonus = 0.1 if (doc.metadata or {}).get("source") in {"document", "youtube"} else 0.0
        return bm25_score + overlap + title_bonus + source_bonus
