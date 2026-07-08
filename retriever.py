from vector_store import get_vector_store


def get_retriever(video_id: str):

    vector_db = get_vector_store(video_id)

    if vector_db is None:
        return None


    retriever = vector_db.as_retriever(
        search_type = "mmr",
        search_kwargs={
            "k": 3,
            "fetch_k":10
        }
    )

    return retriever



# Testing
if __name__ == "__main__":

    video_id = input("Enter Video ID: ")

    retriever = get_retriever(video_id)


    if retriever is not None:

        query = "What is brain?"

        results = retriever.invoke(query)


        print("Retrieved Chunks:", len(results))


        for i, doc in enumerate(results):

            print(f"\n--- Chunk {i+1} ---")
            print(doc.page_content[:500])


    else:
        print("Retriever not created")