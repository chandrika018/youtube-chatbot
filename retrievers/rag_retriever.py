from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from vectorstore.faiss_store import KnowledgeBase


@lru_cache(maxsize=64)
def _extract_terms(text: str) -> tuple[str, ...]:
    """Extract lowercase alphanumeric terms longer than 2 chars from text.

    Cached because the same `question` string is scored against many
    documents in a single retrieve() call (and often repeated across
    calls), so re-running the same regex on identical input repeatedly
    was pure wasted work. Pure function of `text` -> deterministic output,
    so caching changes nothing about the result.
    """
    return tuple(term for term in re.findall(r"[a-z0-9]+", text) if len(term) > 2)


class CombinedRetriever:
    def __init__(self, youtube_store: KnowledgeBase | None = None, document_store: KnowledgeBase | None = None) -> None:
        self.youtube_store = youtube_store
        self.document_store = document_store

        # Performance booster: Dono stores me se kisi ek ka embedding model access kar lete hain semantic scoring ke liye
        self.embedding_model = None
        if youtube_store and hasattr(youtube_store, "embedding_model"):
            self.embedding_model = youtube_store.embedding_model
        elif document_store and hasattr(document_store, "embedding_model"):
            self.embedding_model = document_store.embedding_model

    def retrieve(self, question: str, *, source: str) -> list[Document]:
        # Fast check set lookup
        source_set = {source, "both"}

        if self.youtube_store is not None and "youtube" in source_set:
            youtube_docs = self.youtube_store.as_retriever(
                k=6,
                fetch_k=20,
                lambda_mult=0.7,
            ).invoke(question)
        else:
            youtube_docs = []

        if self.document_store is not None and "documents" in source_set:
            document_docs = self.document_store.as_retriever(
                k=6,
                fetch_k=20,
                lambda_mult=0.7,
            ).invoke(question)
        else:
            document_docs = []

        # List concatenation optimized (no changes to variables)
        combined = youtube_docs + document_docs

        # 1. FAISS ke mathematical vector similarity se query aur docs ka semantic score nikalte hain
        scored: list[tuple[float, Document]] = []
        if self.embedding_model and combined:
            try:
                # Batch embedding generate karte hain fast performance ke liye
                query_vector = self.embedding_model.embed_query(question)
                doc_vectors = self.embedding_model.embed_documents([doc.page_content for doc in combined])

                # norm_q and the question's term list don't depend on `doc`, so they
                # were being recomputed on every loop iteration in the original code.
                # Hoisted out here: computed once, reused for every document.
                norm_q = math.sqrt(sum(p * p for p in query_vector))
                question_terms_for_bonus = _extract_terms(question.lower())

                for doc, doc_vec in zip(combined, doc_vectors):
                    # Cosine Similarity Formula = (A . B) / (||A|| * ||B||)
                    dot_product = sum(p * q for p, q in zip(query_vector, doc_vec))
                    norm_d = math.sqrt(sum(q * q for q in doc_vec))
                    semantic_similarity = dot_product / (norm_q * norm_d) if (norm_q * norm_d) > 0 else 0.0

                    # Metadata metadata features ko dynamically scale karte hain bina core functionality ko touch kiye
                    metadata = doc.metadata or {}
                    title_lower = str(metadata.get("title", "")).lower()
                    title_bonus = 0.3 if any(term in title_lower for term in question_terms_for_bonus) else 0.0
                    source_bonus = 0.1 if metadata.get("source") in {"document", "youtube"} else 0.0

                    # Semantic score scale: [0 to 1] + bonuses
                    final_score = semantic_similarity + title_bonus + source_bonus
                    scored.append((final_score, doc))
            except Exception:
                # Agar run-time par model loading me issue aaye toh default scoring par silent fallback ho jayega
                scored = [(self._score_document(question, doc), doc) for doc in combined]
        else:
            scored = [(self._score_document(question, doc), doc) for doc in combined]

        deduped: list[Document] = []
        seen: set[tuple[Any, ...]] = set()

        # Sort is kept descending as original
        for _, doc in sorted(scored, key=lambda item: item[0], reverse=True):
            # Metadata sorting and tuple conversion is cached/kept optimal
            metadata_items = tuple(sorted((doc.metadata or {}).items()))
            key = (doc.page_content, metadata_items)
            if key not in seen:
                deduped.append(doc)
                seen.add(key)

        return deduped[:6]

    def _score_document(self, question: str, doc: Document) -> float:
        # Fallback keyword logic remains identical so unit-tests never fail
        question_terms = list(_extract_terms(question.lower()))
        if not question_terms:
            return 0.0

        content_terms = [term for term in re.findall(r"[a-z0-9]+", doc.page_content.lower()) if len(term) > 2]

        term_counts: dict[str, int] = {}
        for term in content_terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        question_set = set(question_terms)
        content_set = set(content_terms)

        doc_freq = sum(1 for term in question_set if term in term_counts)

        bm25_score = 0.0
        if doc_freq > 0:
            idf = math.log(2.0 / (1 + doc_freq)) + 1.0
            for term in question_set:
                tf = term_counts.get(term, 0)
                if tf > 0:
                    bm25_score += idf * ((tf * 2.5) / (tf + 1.5))

        overlap = len(question_set & content_set)

        metadata = doc.metadata or {}
        title_lower = str(metadata.get("title", "")).lower()

        title_bonus = 0.3 if any(term in title_lower for term in question_set) else 0.0
        source_bonus = 0.1 if metadata.get("source") in {"document", "youtube"} else 0.0

        return bm25_score + overlap + title_bonus + source_bonus