import os

from embedding import get_documents, get_embedding_model
from langchain_chroma import Chroma
from transcript import extract_video_id


def get_vector_store(video_url: str):

    video_id = extract_video_id(video_url)

    persist_dir = f"./vector_store/{video_id}"

    embedding_model = get_embedding_model()

    # Already exists
    if os.path.exists(persist_dir):
        print("Loading Existing Chroma...")

        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
            collection_name=video_id
        )

    print("Creating New Chroma...")

    docs = get_documents(video_url)

    if docs is None:
        return None

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_name=video_id
    )

    return vector_store