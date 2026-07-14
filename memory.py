from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from chatbot import get_chain
from retriever import get_retriever


class ChatState(TypedDict):
    video_url: str
    question: str
    context: str
    answer: str


def retrieve(state: ChatState):

    retriever = get_retriever(state["video_url"])

    docs = retriever.ainvoke(state["question"])

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "context": context
    }

def generate(state: ChatState):
    chain = get_chain()
    response = chain.ainvoke(
        {
            "context": state["context"],
            "question": state["question"]
        }
    )

    return {
        "answer": response.content
    }


def get_graph():
    builder = StateGraph(ChatState)

    builder.add_node("retrieve", retrieve)

    builder.add_node("generate", generate)

    builder.add_edge(START,"retrieve")
    builder.add_edge("retrieve","generate")
    builder.add_edge("generate",END)

    memory = MemorySaver()

    graph = builder.compile(
        checkpointer=memory
    )
    return graph 