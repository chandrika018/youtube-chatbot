
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transcript import get_transcript

def get_embeddings(video_id: str):

    transcript_text = get_transcript(video_id)

    if transcript_text is None:
        return None, None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    docs = text_splitter.create_documents([transcript_text])

    embedding_model = HuggingFaceEmbeddings(
        model_name = "BAAI/bge-small-en-v1.5"
        )
    
    print(len(docs))
    
    return docs, embedding_model
     
     
if __name__ == "__main__":

    video_id = input("Enter Video ID: ")

    docs, embedding_model = get_embeddings(video_id)

    if docs is not None:
        print(f"Total Chunks: {len(docs)}")
        print("\nFirst Chunk:\n")
        print(docs[0].page_content[:500])
    else:
        print("transcript is not found")