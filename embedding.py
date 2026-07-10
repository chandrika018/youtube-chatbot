from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transcript import get_transcript

# Load embedding model only once
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


def get_documents(video_url: str):

    transcript_text = get_transcript(video_url)

    if transcript_text is None:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([transcript_text])

    print("Chunks:", len(docs))

    return docs


def get_embedding_model():
    return embedding_model


if __name__ == "__main__":

    video_url = input("Enter Video URL: ")

    docs = get_documents(video_url)

    if docs:
        print(f"Chunks: {len(docs)}")