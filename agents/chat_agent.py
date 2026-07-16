from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from chatbot import get_chain
from retrievers.rag_retriever import CombinedRetriever
from vectorstore.faiss_store import KnowledgeBase


class ChatState(TypedDict):
    question: str
    context: str
    answer: str
    source: str
    sources: list[dict[str, object]]


class ChatAgent:
    """LangGraph-based chat agent that retrieves context and generates answers."""

    def __init__(self, youtube_store: KnowledgeBase | None = None, document_store: KnowledgeBase | None = None) -> None:
        self.retriever = CombinedRetriever(youtube_store=youtube_store, document_store=document_store)
        self.chain = get_chain()

    def retrieve(self, state: ChatState) -> dict[str, str | list[dict[str, object]]]:
        """Retrieve relevant documents and build the context string and source metadata."""
        docs = self.retriever.retrieve(state["question"], source=state.get("source", "both"))

        # Single pass over docs: build context chunks and source metadata together
        # instead of iterating the docs list twice (once for context, once for sources).
        context_parts: list[str] = []
        sources: list[dict[str, object]] = []
        for doc in docs:
            context_parts.append(doc.page_content)
            # Read-only access, so no need to copy metadata into a new dict.
            metadata = doc.metadata or {}
            sources.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "title": metadata.get("title"),
                    "url": metadata.get("url"),
                    "filename": metadata.get("filename"),
                }
            )

        context = "\n\n".join(context_parts)
        return {"context": context, "sources": sources}

    def generate(self, state: ChatState) -> dict[str, str | list[dict[str, object]]]:
        """Generate an answer from the LLM chain using the retrieved context."""
        response = asyncio.run(self.chain.ainvoke({"context": state["context"], "question": state["question"]}))
        return {"answer": response.content, "sources": state.get("sources", [])}

    def build_graph(self):
        """Build and compile the LangGraph state graph for retrieve -> generate flow."""
        builder = StateGraph(ChatState)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("generate", self.generate)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()


if __name__ == "__main__":
    agent = ChatAgent()
    print(agent.build_graph())