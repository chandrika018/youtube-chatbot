from langchain_core.prompts import ChatPromptTemplate

CHAT_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful assistant answering questions from the provided retrieved context.
Use only the retrieved context to answer. If the context is insufficient, say so clearly.

Context:
{context}

Question:
{question}

Answer:
"""
)
