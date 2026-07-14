from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

import os


def load_document(file_path):

    # Get file extension
    suffix = os.path.splitext(file_path)[1].lower()

    # Select loader based on file type
    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)

    elif suffix == ".txt":
        loader = TextLoader(file_path)

    elif suffix == ".docx":
        loader = Docx2txtLoader(file_path)

    else:
        raise ValueError("Unsupported file format.")

    # Load documents
    documents = loader.load()

    return documents