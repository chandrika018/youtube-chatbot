from vector_store import get_vector_store
from vector_store import get_vector_file

def get_retriever(video_url: str):
     
    vector_db = get_vector_store(video_url)
    if vector_db is None:
        return None
    

    retriever=vector_db.as_retriever(
        search_type = "mmr",
        search_kwargs={"k": 3}
        )
    print(type(retriever))
    return retriever
   
def get_retri_file(uploaded_file):
    vector_db = get_vector_file(uploaded_file)

    if vector_db is None:
        return None
    
    retri_text=vector_db.as_retriever(
        search_type = "mmr",
        search_kwargs={"k": 3}
        )
    print(type(retri_text))
    return retri_text

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