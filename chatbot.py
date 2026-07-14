
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
# Load Environment Variables
load_dotenv()


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

prompt_text = ChatPromptTemplate.from_template(
    """
You are a Retrieval-Augmented Generation (RAG) assistant.

Your knowledge is limited ONLY to the retrieved document content.

Instructions:

- Read every retrieved chunk carefully.
- Answer the user's question using ONLY the retrieved document content.
- If the answer is spread across multiple chunks, combine the relevant information into a clear and concise response.
- Never use outside knowledge or make assumptions.
- Do not invent or hallucinate any facts.
- If the retrieved content provides only a partial answer, return the available information and do not add anything beyond it.
- If none of the retrieved chunks contain information related to the user's question, respond exactly with:
  "I couldn't find that information in the uploaded document."

Retrieved Document Chunks:
{documents}

Question:
{question}

Answer:
    """
)

def get_chain_text():
    chain_text = prompt_text | llm
    return chain_text

def get_chain():
      # Create Chain
    chain = prompt | llm
    return chain

print("chatbot.py loaded successfully")


# ---------------- Testing ----------------

if __name__ == "__main__":

    video_url = input("Enter Video url: ")


    while True:

        question = input(
            "\nAsk a Question (type 'exit' to quit): "
        )


        if question.lower() == "exit":
            break

 