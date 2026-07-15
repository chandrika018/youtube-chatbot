from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: Iterable[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(list(documents))
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata.setdefault("chunk_id", index)
        chunk.metadata = metadata
    return chunks
