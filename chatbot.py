import os
import re
from typing import Callable, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# Precompiled once at module load instead of inside _fallback_response, where
# they were previously recompiled on every call (every fallback invocation).
# Same patterns, same behavior — just compiled once and reused.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_KEYWORD_RE = re.compile(r"[A-Za-z0-9']+")


class SimpleResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class PromptChain:
    def __init__(self, prompt, llm) -> None:
        self.prompt = prompt
        self.llm = llm

    def invoke(self, payload, *, stream: bool = False, on_token: Optional[Callable[[str], None]] = None):
        if isinstance(self.llm, ChatGroq):
            if stream:
                return self._stream_response(payload, on_token=on_token)
            return (self.prompt | self.llm).invoke(payload)
        return SimpleResponse(self._fallback_response(payload))

    async def ainvoke(self, payload, *, stream: bool = False, on_token: Optional[Callable[[str], None]] = None):
        if isinstance(self.llm, ChatGroq):
            if stream:
                return await self._astream_response(payload, on_token=on_token)
            return await (self.prompt | self.llm).ainvoke(payload)
        return SimpleResponse(self._fallback_response(payload))

    def _stream_response(self, payload, *, on_token: Optional[Callable[[str], None]] = None):
        response = (self.prompt | self.llm).invoke(payload)
        content = getattr(response, "content", "") or ""
        if on_token is not None:
            for chunk in content:
                on_token(chunk)
        return SimpleResponse(content)

    async def _astream_response(self, payload, *, on_token: Optional[Callable[[str], None]] = None):
        response = await (self.prompt | self.llm).ainvoke(payload)
        content = getattr(response, "content", "") or ""
        if on_token is not None:
            for chunk in content:
                on_token(chunk)
        return SimpleResponse(content)

    def _fallback_response(self, payload) -> str:
        context = payload.get("context") or payload.get("documents") or ""
        question = payload.get("question", "")
        if not context.strip():
            return "I couldn't find that information in the provided context."

        sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(context.strip()) if segment.strip()]
        keywords = [word.lower() for word in _KEYWORD_RE.findall(question) if len(word) > 2]
        if keywords:
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords):
                    return sentence
        if sentences:
            return sentences[0]
        return "I couldn't find that information in the provided context."


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            return ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key, temperature=0.3, streaming=True)
        except Exception:
            pass
    return object()


llm = get_llm()

prompt = ChatPromptTemplate.from_template(
    """
You are a Retrieval-Augmented Generation (RAG) assistant.
Your knowledge is limited ONLY to the retrieved context.

Instructions:
- Read every retrieved chunk carefully.
- Answer using ONLY those chunks.
- If the information is partial, share what you can.
- Do not invent facts.
- If nothing relevant is found, say that clearly.

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
- Answer using ONLY the retrieved document content.
- If the answer spans multiple chunks, combine them clearly.
- Do not invent or hallucinate facts.
- If none of the retrieved content is relevant, say that clearly.

Retrieved Document Chunks:
{documents}

Question:
{question}

Answer:
"""
)


def get_chain_text():
    return PromptChain(prompt_text, llm)


def get_chain():
    return PromptChain(prompt, llm)


print("chatbot.py loaded successfully")