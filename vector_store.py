import os

from embedding import get_documents, get_embedding_model
from langchain_community.vectorstores import FAISS
from transcript import extract_video_id
from embedding import get_text


def get_vector_store(video_url: str):

    video_id = extract_video_id(video_url)

    persist_dir = f"./vector_store/{video_id}"

    embedding_model = get_embedding_model()

    # -----------------------------
    # Load Existing FAISS Index
    # -----------------------------
    if os.path.exists(persist_dir):
        print("Loading Existing FAISS...")

        return FAISS.load_local(
            folder_path=persist_dir,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )

    print("Creating New FAISS Index...")

    docs = get_documents(video_url)

    if docs is None:
        return None

    # -----------------------------
    # Create FAISS Index
    # -----------------------------
    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embedding_model
    )

    # -----------------------------
    # Save Index
    # -----------------------------
    vector_store.save_local(persist_dir)

    return vector_store


def get_vector_file(uploaded_file):

    text_docs = get_text(uploaded_file)

    persist_dir = './vectore_store'

    embedding_model = get_embedding_model()

    # -----------------------------
    # Load Existing FAISS Index
    # -----------------------------
    if os.path.exists(persist_dir):
        print("Loading Existing FAISS...")

        return FAISS.load_local(
            folder_path=persist_dir,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )

    print("Creating New FAISS Index...")


    if text_docs is None:
        return None

    # -----------------------------
    # Create FAISS Index
    # -----------------------------
    vector_text = FAISS.from_documents(
        documents=text_docs,
        embedding=embedding_model
    )

    # -----------------------------
    # Save Index
    # -----------------------------
    vector_text.save_local(persist_dir)

    return vector_text
