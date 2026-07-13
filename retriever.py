from vector_store import get_vector_store
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_groq import ChatGroq
import os
def get_retriever(video_url: str):

    vector_db = get_vector_store(video_url)

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        streaming=True
    )

    if vector_db is None:
        return None

    compressor = LLMChainExtractor.from_llm(llm)

    retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
)
    return compression_retriever



# Testing
if __name__ == "__main__":

    video_url = input("Enter Video URL: ")

    retriever = get_retriever(video_url)


    if retriever is not None:

        query = "What is brain?"

        results = retriever.invoke(query)


        print("Retrieved Chunks:", len(results))


        for i, doc in enumerate(results):

            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:500])


    else:
        print("Retriever not created")