
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from retriever import get_retriever


# Load Environment Variables
load_dotenv()


def ask_question(video_id: str, question: str):
    """
    Answer user question based on YouTube transcript.
    """

    # Get Retriever
    retriever = get_retriever(video_id)

    if retriever is None:
        return "Transcript not found."


    # Retrieve relevant documents
    docs = retriever.invoke(question)


    if not docs:
        return "I couldn't find relevant information in the video."


    print("Retrieved Chunks:", len(docs))

    for i, doc in enumerate(docs, start=1):
        print(f"\nChunk {i}:")
        print(doc.page_content[:200])


    # Create Context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )


    print("\nContext Length:", len(context))


    # Gemini Model
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )


    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
    """
    You are a YouTube video assistant.

Use ONLY the context below to answer.

If context does not contain the answer,
say:
"I couldn't find that information in the video."

Context:
{context}

Question:
{question}

Answer:
"""

    )


    # Create Chain
    chain = prompt | llm


    # Generate Answer
    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )


    return response.content



# ---------------- Testing ----------------

if __name__ == "__main__":

    video_id = input("Enter Video ID: ")


    while True:

        question = input(
            "\nAsk a Question (type 'exit' to quit): "
        )


        if question.lower() == "exit":
            break


        answer = ask_question(
            video_id,
            question
        )


        print("\nAnswer:")
        print(answer)

