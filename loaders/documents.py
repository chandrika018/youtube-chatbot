from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# Precompiled once at module load instead of recompiling on every call to
# load_documents_from_paths (previously recompiled per-document via re.sub).
_WHITESPACE_RE = re.compile(r"\s+")


def is_supported_file(file_name: str | os.PathLike[str]) -> bool:
    return Path(str(file_name)).suffix.lower() in SUPPORTED_EXTENSIONS


def load_documents_from_paths(file_paths: Iterable[str | os.PathLike[str]]) -> list[Document]:
    documents: list[Document] = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"The file {path} does not exist.")
        if not is_supported_file(path):
            raise ValueError(f"Unsupported file type for {path.name}. Please upload PDF, DOCX, or TXT files.")

        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
            elif suffix == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")
            elif suffix == ".docx":
                loader = Docx2txtLoader(str(path))
            else:
                raise ValueError(f"Unsupported file type for {path.name}.")
            loaded = loader.load()
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Could not read {path.name}: {exc}") from exc

        if not loaded:
            raise ValueError(f"The file {path.name} did not contain any readable content.")

        for doc in loaded:
            text = (doc.page_content or "").strip()
            text = _WHITESPACE_RE.sub(" ", text)
            if not text:
                continue
            metadata = dict(doc.metadata or {})
            metadata.setdefault("source", "document")
            metadata.setdefault("filename", path.name)
            metadata.setdefault("file_type", suffix.lstrip("."))
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

    return documents


def load_documents_from_uploads(uploaded_files: Iterable[Any]) -> list[Document]:
    temp_paths: list[str] = []
    try:
        for uploaded_file in uploaded_files:
            if uploaded_file is None:
                continue
            if not getattr(uploaded_file, "name", ""):
                raise ValueError("One of the uploaded files is missing a name.")
            if not is_supported_file(uploaded_file.name):
                raise ValueError(f"Unsupported file type for {uploaded_file.name}. Please upload PDF, DOCX, or TXT files.")
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_paths.append(tmp.name)
        return load_documents_from_paths(temp_paths)
    finally:
        for file_path in temp_paths:
            try:
                os.remove(file_path)
            except FileNotFoundError:
                continue


if __name__ == "__main__":
    print("Document loader ready")