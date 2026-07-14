from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from chatbot import get_chain_text
from retriever import get_retri_file


class ChatState(TypedDict):
    file_path: str
    question: str
    documents: str
    answer: str


def retrieve_file(state: ChatState):

    retriever = get_retri_file(state["file_path"])

    docs = retriever.invoke(state["question"])

    documents = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "documents": documents
    }
async def generate_file(state: ChatState):
    chain = get_chain_text()
    response = await chain.ainvoke(
        {
            "documents": state["documents"],
            "question": state["question"]
        }
    )

    return {
        "answer": response.content
    }



def get_graph_file():
    builder = StateGraph(ChatState)

    builder.add_node("retrieve", retrieve_file)

    builder.add_node("generate", generate_file)

    builder.add_edge(START,"retrieve")
    builder.add_edge("retrieve","generate")
    builder.add_edge("generate",END)

    memory = MemorySaver()

    graph_file = builder.compile(
        checkpointer=memory
    )
    return graph_file 