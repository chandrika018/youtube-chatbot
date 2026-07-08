from embedding import get_embeddings
from langchain_chroma import Chroma

def  get_vector_store(video_id: str):
    docs, embedding_model = get_embeddings(video_id)

    if docs is None:
        return None
    
    vector_store = Chroma.from_documents(
        documents = docs,
        embedding = embedding_model,
        persist_directory = './vector_store',
        collection_name = video_id
    )
    
    return vector_store

if __name__ == "__main__":

    video_id = input("Enter Video ID: ")

    vector_db = get_vector_store(video_id)

    if vector_db is not None:
        print("✅ Chroma Vector Database Created Successfully!")