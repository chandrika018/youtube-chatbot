from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: Iterable[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    prepared_documents: list[Document] = []
    for document in documents:
        text = (document.page_content or "").strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        prepared_document = Document(page_content=text, metadata=dict(document.metadata or {}))
        prepared_documents.append(prepared_document)

    if not prepared_documents:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " "])
    chunks = splitter.split_documents(prepared_documents)
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata.setdefault("chunk_id", index)
        metadata.setdefault("chunk_size", len(chunk.page_content.split()))
        chunk.metadata = metadata
    return chunks
