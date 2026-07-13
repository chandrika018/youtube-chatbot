
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import time
from retriever import get_retriever


# Load Environment Variables
load_dotenv()


    # Gemini Model
llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        streaming=True
    )


    # Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
You are a Retrieval-Augmented Generation (RAG) assistant.

Your knowledge is limited ONLY to the retrieved transcript.

Instructions:

- Read every retrieved chunk carefully.
- If one or more chunks contain information related to the user's question, answer using ONLY those chunks.
- Never reject a question if relevant information exists in the retrieved context.
- Even if the information is partial, provide the partial answer.
- Do not invent facts.
- Respond with "I couldn't find that information in the video." ONLY when every retrieved chunk is completely unrelated to the question.

Retrieved Chunks:
{context}

Question:
{question}

Answer:
    """
    )


    # Create Chain
chain = prompt | llm

def ask_question(video_url: str, question: str):

    total_start = time.time()

    # Retriever
    t1 = time.time()
    retriever = get_retriever(video_url)
    print(f"Retriever Time: {time.time()-t1:.2f} sec")

    if retriever is None:
        return "Transcript not found."

    # Retrieval
    t2 = time.time()
    docs = retriever.invoke(question)
    print(f"Search Time: {time.time()-t2:.2f} sec")

    if not docs:
        return "I couldn't find relevant information in the video."

    context = "\n\n".join(doc.page_content for doc in docs)

    # LLM
    t3 = time.time()
   
    messages = prompt.format_messages(
            context= context,
            question= question
        )
    for chunk in llm.stream(messages):
        yield chunk.content

   
    print(f"LLM Time: {time.time()-t3:.2f} sec")

    print(f"Total Time: {time.time()-total_start:.2f} sec")

    # return response.content





# ---------------- Testing ----------------

if __name__ == "__main__":

    video_url = input("Enter Video url: ")


    while True:

        question = input(
            "\nAsk a Question (type 'exit' to quit): "
        )


        if question.lower() == "exit":
            break


        answer = ask_question(
            video_url,
            question
        )


        print("\nAnswer:")
        print(answer)

