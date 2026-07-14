
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
# Load Environment Variables
load_dotenv()


def get_llm():
       # Gemini Model
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        streaming=True
    )
    return llm
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

def get_chain():
    llm = get_llm()
      # Create Chain
    chain = prompt | llm
    return chain




# ---------------- Testing ----------------

if __name__ == "__main__":

    video_url = input("Enter Video url: ")


    while True:

        question = input(
            "\nAsk a Question (type 'exit' to quit): "
        )


        if question.lower() == "exit":
            break

 