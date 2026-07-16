from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(documents: Iterable[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    prepared_documents: list[Document] = []
    
    # Pre-compile regex taaki loop ke andar har baar parse na karna pade (Fast Execution)
    whitespace_pattern = re.compile(r"\s+")
    
    for document in documents:
        text = (document.page_content or "").strip()
        if not text:
            continue
            
        # Regex substitution using pre-compiled pattern
        text = whitespace_pattern.sub(" ", text)
        if not text:  # Extra safety check in case string becomes empty after space cleanup
            continue
            
        # Metadata copying fallback fast dictionary creation
        prepared_document = Document(page_content=text, metadata=dict(document.metadata or {}))
        prepared_documents.append(prepared_document)

    if not prepared_documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = splitter.split_documents(prepared_documents)
    
    # Loop optimization: inline updates bina baar-baar key lookups ke
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        
        # setdefault internally evaluate hota hai, explicit checking fast lookup dega
        if "chunk_id" not in metadata:
            metadata["chunk_id"] = index
        if "chunk_size" not in metadata:
            # Word count optimize karne ke liye split() ko bina argument ke pass kiya (Fast performance)
            metadata["chunk_size"] = len(chunk.page_content.split())
            
        chunk.metadata = metadata
        
    return chunks